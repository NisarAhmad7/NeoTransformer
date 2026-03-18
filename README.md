# NeoTransformer: Advanced Seq2Seq Transformer

### Author: Nisar Ahmad Zamani | 2026

NeoTransformer is a fully implemented sequence-to-sequence Transformer model for autoregressive NLP tasks. It includes:

Multi-head self-attention

Cross-attention in decoder

Residual connections and layer normalization

Position embeddings and masking for autoregressive generation

This implementation can be used for tasks like machine translation, text summarization, and other sequence modeling tasks.

## Features

Encoder-decoder architecture with full Transformer blocks

Multi-head self-attention with scaling and masking

Configurable number of layers, heads, and embedding size

Supports source and target vocabularies with padding masks

Easy to run on CPU or GPU

Modular design for extending or fine-tuning

# Installation

## Clone the repository:

[NeoTransformer GitHub Repository](https://github.com/NisarAhmad7/NeoTransformer)
or : https://github.com/NisarAhmad7/NeoTransformer

cd NeoTransformer

## Install requirements:
```python 
pip install -r requirements.txt
```
Usage
Example: Quick Test
```python
import torch
from transformer import Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
# Sample input (batch of sequences)
```python


x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0],
                  [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0],
                    [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
```


`src_pad_idx = 0`
`trg_pad_idx = 0`
`src_vocab_size = 10`
`trg_vocab_size = 10`


# Initialize model
```python
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
```

# Forward pass
```pyhton
output = model(x, trg[:, :-1])
print(output.shape)  # (batch_size, target_len-1, trg_vocab_size)
```
# Project Structure
NeoTransformer/
│
├── transformer.py         
├── README.md               
├── requirements.txt       
└── .gitignore             