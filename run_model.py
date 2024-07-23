import torch
import os
import math
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


# Self-Attention mechanism
class self_attention(nn.Module):
    def __init__(self, d, dk):
        super().__init__()
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

# Load the trained model weights
model.load_state_dict(torch.load('weights.pth'))
model.eval()


# Function to generate text
def generate_text(prompt, max_length=1000, temperature=1.0, top_k=50):
    tokens = tokenizer.encode(prompt)
    generated_tokens = tokens.copy()

    for _ in range(max_length):
        input_ids = torch.tensor([generated_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_ids)
        next_token_logits = output[0, -1, :] / temperature
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        sorted_logits = sorted_logits[:top_k]
        sorted_indices = sorted_indices[:top_k]
        next_token_probs = torch.softmax(sorted_logits, dim=-1)
        next_token_id = sorted_indices[torch.multinomial(next_token_probs, num_samples=1)].item()
        generated_tokens.append(next_token_id)
        if next_token_id == tokenizer.tokenizer.sep_token_id:
            break

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


# User input prompt
user_prompt = input("Enter your prompt: ")
generated_text = generate_text(user_prompt, max_length=100, temperature=0.7, top_k=50)
print("\nGenerated Text:\n", generated_text)
# save the generated text to a file
with open('generated_text.txt', 'w') as f:
    f.write(generated_text)
    print("Generated text saved to 'generated_text.txt'")

