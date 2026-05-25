import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, tokens: list[int], context_length):
        super().__init__()
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length - 1

    def __getitem__(self, idx):
        return self.tokens[idx:idx + self.context_length], self.tokens[idx + 1:idx + self.context_length + 1]


# Define config
CONTEXT_LENGTH = 8
EMBEDDING_DIM = 32
BATCH_SIZE = 32
rows = 1000


# Process data
df = pd.read_csv('./data/recipes_data.csv', nrows=rows)
df = df.dropna(subset=["title", "ingredients", "directions"])
df['text'] = (
    "<RECIPE>\n"
    "Title: " + df["title"].astype(str) + "\n\n"
    "Ingredients:\n" + df["ingredients"].astype(str) + "\n\n"
    "Instructions:\n" + df["directions"].astype(str) + "\n"
    "</RECIPE>"
)


# Tokenize data
char_to_token = {}
token_to_char = {}
for i in range(rows):
    recipe = df.iloc[i]["text"]
    for char in recipe:
        if char not in char_to_token:
            token = len(char_to_token)
            char_to_token[char] = token
            token_to_char[token] = char


# Define encoding/decoding functions using tokenized data
def encode(text):
    tokens = []
    for char in text:
        tokens.append(char_to_token[char])
    return tokens


def decode(tokens):
    text = ""
    for token in tokens:
        text += token_to_char[int(token)]
    return text



full_text = "".join(df['text'])
full_tokens = encode(full_text)
dataset = Data(full_tokens, CONTEXT_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(len(dataset))
print(dataset[5])
