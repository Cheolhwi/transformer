import torch
import os
import math
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from transformers import BertTokenizer

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Tokenizer class
class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> list:
        return self.tokenizer(text, return_tensors='pt')['input_ids'][0].tolist()

    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)


tokenizer = Tokenizer()


# Dataset class
class Transformer_Dataset(Dataset):
    def __init__(self, files: list):
        self.tokens = []
        self.vocab_size = tokenizer.vocab_size
        self.data = self.read_files(files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def read_files(self, files: list):
        data = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.tokens.extend(tokenizer.encode(file.read()))
        input_ids = torch.tensor(self.tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(self.tokens[1:], dtype=torch.long)
        data.append((input_ids, target_ids))
        return data


# Define the current working directory path
path = os.getcwd()

# List of text files for training
file_paths = [
    os.path.join(path, 'file1.txt'),
]

data = Transformer_Dataset(file_paths)

# DataLoader
data_loader = DataLoader(data, batch_size=8, shuffle=True)


# Self-Attention mechanism
class self_attention(nn.Module):
    def __init__(self, d, dk):
        super().__init__()
        self.dk = dk
        self.dk = dk
        self.q = nn.Linear(d, dk)
        self.k = nn.Linear(d, dk)
        self.v = nn.Linear(d, dk)

    def attention(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor):
        return torch.softmax((Q @ K.transpose(-2, -1)) / self.dk ** 0.5 + mask, dim=-1) @ V

    def forward(self, x: tuple):
        x, mask = x
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        return self.attention(Q, K, V, mask)


# Decoder class
class decoder(nn.Module):
    def __init__(self, head_num, d, dk, dff, dropout=0.1):  # 增加 Dropout 率
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(head_num):
            self.heads.append(self_attention(d, dk))
        self.o = nn.Linear(head_num * dk, d)
        self.norm1 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, dff),
            nn.ReLU(),
            nn.Linear(dff, d),
        )
        self.norm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: tuple):
        x, mask = x
        heads_res = []
        for head in self.heads:
            heads_res.append(head((x, mask)))
        a = self.dropout1(self.o(torch.concat(heads_res, dim=-1)))
        b = self.norm1(a + x)
        y = self.norm2(self.dropout2(self.ffn(b)) + b)
        return (y, mask)


# Transformer class
class transformer(nn.Module):
    def __init__(self, decoder_num=3, head_num=4, d=256, dk=64, dff=512, vocab_size=30522):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d)
        self.pos_code = Tensor()
        self.mask = Tensor()
        self.zero_mask = Tensor()
        self.decoders = nn.Sequential()
        for _ in range(decoder_num):
            self.decoders.append(decoder(head_num, d, dk, dff))
        self.last_linear = nn.Linear(d, self.vocab_size)

    def get_mask(self, sequence_len):
        if not self.training:
            if sequence_len != len(self.zero_mask):
                self.zero_mask = torch.zeros(sequence_len, sequence_len).to(device)
            return self.zero_mask
        if sequence_len != len(self.mask):
            self.mask = torch.zeros(sequence_len, sequence_len).to(device)
            for i in range(sequence_len):
                for j in range(sequence_len):
                    if j > i:
                        self.mask[i][j] = -1e9
        return self.mask

    def pos_encode(self, sequence_len):
        if len(self.pos_code) == sequence_len:
            return self.pos_code
        self.pos_code = []
        for pos in range(sequence_len):
            buf = []
            for i in range(self.d):
                value = math.sin(pos / 1e4 ** (i / self.d)) if i % 2 == 0 else math.cos(pos / 1e4 ** ((i - 1) / self.d))
                buf.append(value)
            self.pos_code.append(torch.tensor(buf).to(device))
        self.pos_code = torch.stack(self.pos_code)
        return self.pos_code

    def forward(self, x: Tensor):
        sequence_len = x.shape[1]
        x = self.embedding(x) * self.d ** 0.5 + self.pos_encode(sequence_len).to(device)
        y, _ = self.decoders((x, self.get_mask(sequence_len)))
        y = self.last_linear(y)
        return y


# Define the model
model = transformer(decoder_num=8, head_num=10, d=512, dk=128, dff=1024).to(device)


# Training function
def train(epochs: int, lr: float, weight_decay: float):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            pred = pred.view(-1, pred.size(-1))
            y = y.view(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')


train(epochs=400, lr=1e-4, weight_decay=1e-4)
torch.save(model.state_dict(), 'weights.pth')
