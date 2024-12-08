import os
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Dataset class
class EnigmaDataset(Dataset):
    def __init__(self, file_path):
        self.data = ""
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    self.data += line
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {i}: {e}")
        self.data = json.loads(self.data)
        all_characters = set("".join([item['plain'] + item['encoded'] for item in self.data]))
        self.char_encoder = LabelEncoder()
        self.char_encoder.fit(list(all_characters))
        self.vocab_size = len(self.char_encoder.classes_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        plain_text = item['plain']
        encoded_text = item['encoded']
        encoded_ids = self.char_encoder.transform(list(encoded_text))
        plain_ids = self.char_encoder.transform(list(plain_text))
        return torch.tensor(encoded_ids, dtype=torch.long), torch.tensor(plain_ids, dtype=torch.long)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# ResNet Decoder
class ResNetDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_blocks, max_length):
        super(ResNetDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.proj(embedded)
        out = self.res_blocks(embedded)
        out = out.permute(0, 2, 1)
        logits = self.fc(out)
        return logits

# Training loop
def train_model(model, dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for encoded, plain in dataloader:
            encoded = encoded.to(device)
            plain = plain.to(device)
            logits = model(encoded)
            logits = logits.view(-1, logits.size(-1))
            plain = plain.view(-1)
            loss = F.cross_entropy(logits, plain, ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Decode function
def decode_word(model, encoded_word, char_encoder, device):
    model.eval()
    with torch.no_grad():
        encoded_ids = np.array(char_encoder.transform(list(encoded_word.replace(" ", ""))))
        input_tensor = torch.tensor(encoded_ids[np.newaxis, :], dtype=torch.long).to(device)
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        decoded_word = "".join(char_encoder.inverse_transform(predictions))
    return decoded_word

# Plot loss
def plot_loss_curve(training_losses, validation_losses, save_path=None):
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_losses, label="Training Loss", marker='o')
    plt.plot(epochs, validation_losses, label="Validation Loss", marker='x')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "loss_curve.png")

    plt.savefig(save_path)

# Parameters
file_path = "/home/smadejski/mathforml/enigma/random_data.json"
batch_size = 16
embed_dim = 64
hidden_dim = 64
num_blocks = 3
num_epochs = 15
max_length = 25

# Dataset and DataLoader
dataset = EnigmaDataset(file_path)
def collate_fn(batch):
    encoded, plain = zip(*batch)
    encoded_padded = pad_sequence(encoded, batch_first=True, padding_value=0)
    plain_padded = pad_sequence(plain, batch_first=True, padding_value=0)
    return encoded_padded, plain_padded
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model, optimizer, and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetDecoder(vocab_size=dataset.vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, max_length=max_length).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

# Train the model
train_model(model, dataloader, optimizer, num_epochs, device)

# Example decoding
encoded_word = "ZQFAQ LA"
decoded_word = decode_word(model, encoded_word, dataset.char_encoder, device)
print(f"Encoded: {encoded_word}, Decoded: {decoded_word}")
