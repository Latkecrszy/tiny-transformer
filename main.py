import pandas as pd


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
        text += token_to_char[token]
    return text




