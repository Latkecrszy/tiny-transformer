import pandas as pd
import torch


# Define config
CONTEXT_LENGTH = 8
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


def create_tensors(text):
    input_window: list[list] = []
    output_window: list[list] = []
    start = 0
    end = CONTEXT_LENGTH
    tokens = encode(text)
    while end < len(tokens):
        input_window.append(tokens[start:end])
        output_window.append(tokens[start+1:end+1])
        start += 1
        end += 1
    input_tensor = torch.tensor(input_window)
    output_tensor = torch.tensor(output_window)
    return input_tensor, output_tensor


input_tensor, output_tensor = create_tensors('sweet potato')
print([decode(i) for i in input_tensor])
print([decode(i) for i in output_tensor])
print(input_tensor.shape)
print(output_tensor.shape)
