# wodehouse-gpt

A decoder-only transformer built from scratch in PyTorch, trained on P.G. Wodehouse's novels to generate text in his style.

## What is this?

A learning project that builds every component of a GPT-style transformer from the ground up - no huggingface transformers library, no pre-trained models, just raw PyTorch. The model learns to write like Wodehouse by predicting the next character in his books.

## Training Data

30 P.G. Wodehouse novels (11.2 million characters) sourced from Project Gutenberg via the [edwardjross/wodehouse](https://huggingface.co/datasets/edwardjross/wodehouse) dataset. Includes Jeeves & Wooster stories, Psmith novels, and more.

## Architecture

- **Type**: Decoder-only transformer (same family as GPT)
- **Tokenization**: Character-level (76 unique characters)
- **Embeddings**: Learned character + positional embeddings
- **Device**: Trains on Apple MPS (Metal GPU) or CPU

## Project Structure

```
tokenizer.py          # Character-level tokenizer (encode/decode)
embedding_demo.py     # Demo: token IDs -> learned vectors
positional_demo.py    # Demo: adding position information to embeddings
data.txt              # Combined training text (generated from parquet)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch datasets pyarrow
```

### Download the data

```bash
curl -kL -o wodehouse_train.parquet \
  "https://huggingface.co/datasets/edwardjross/wodehouse/resolve/main/data/train-00000-of-00001.parquet"
```

### Prepare training text

```python
import pyarrow.parquet as pq

table = pq.read_table('wodehouse_train.parquet')
df = table.to_pandas()

with open('data.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(row['content'])
        f.write('\n\n')
```

## Status

Work in progress. Building each transformer component step by step.
