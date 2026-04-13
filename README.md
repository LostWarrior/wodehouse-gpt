# wodehouse-gpt

A GPT-style transformer built from the ground up - no huggingface transformers library, no pre-trained models, just raw PyTorch. The model learns to write like Wodehouse by predicting the next character in his books.

## Quick Start

```bash
git clone git@github.com:LostWarrior/wodehouse-gpt.git
cd wodehouse-gpt
```

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and prepare data

```bash
curl -kL -o wodehouse_train.parquet \
  "https://huggingface.co/datasets/edwardjross/wodehouse/resolve/main/data/train-00000-of-00001.parquet"

python3 -c "
import pyarrow.parquet as pq
table = pq.read_table('wodehouse_train.parquet')
df = table.to_pandas()
with open('data.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(row['content'])
        f.write('\n\n')
print(f'Created data.txt')
"
```

30 Wodehouse novels, 11.2 million characters from Project Gutenberg via [edwardjross/wodehouse](https://huggingface.co/datasets/edwardjross/wodehouse).

### 3. Train

```bash
python3 train.py            # start fresh
python3 train.py --resume   # pick up from last checkpoint
```

Takes 30-60 minutes on Apple MPS, longer on CPU. Saves to `model.pt` when done.

### 4. Generate

```bash
./jeeves                                    # interactive mode
./jeeves "Jeeves entered the room"          # single prompt
./jeeves "It was a" --chars 300 --temp 0.5  # with options
./jeeves --help
```

To use `jeeves` from anywhere, either symlink it:

```bash
ln -s "$(pwd)/jeeves" /usr/local/bin/jeeves
```

Or add the project to your PATH in `~/.zshrc` (or `~/.bashrc`):

```bash
export PATH="$HOME/path/to/wodehouse-gpt:$PATH"
```

## Sample Output (v4, 17.9M parameters)

```
> Jeeves
Jeeves . ."

Jill thoughtfully. In she had never done with a feeling that she
fended that there was a bit on the street and subserved on and
all that sort of life had been leaned for a splendid composer of
the leader Certain Cambeth to her that the house on the mouth.

> Hello Bertie
Hello Bertie, where he was feeling with the Reggie, but in which
an absolute creature of the netting of the local dinner was still
bunitoring.

"Here's a bit. I suppose I went to the house as well. He waited
here when I see somewhom repried in Amar to me found through one
again and refuge to the little tree and correct when I have
expended the sort of potted speech fate in the necklace
```

Not Shakespeare (or Wodehouse), but it learned English structure, dialogue patterns, and character names from scratch - one character at a time.

## Training Data

30 P.G. Wodehouse novels (11.2 million characters) sourced from Project Gutenberg via the [edwardjross/wodehouse](https://huggingface.co/datasets/edwardjross/wodehouse) dataset. Includes Jeeves & Wooster stories, Psmith novels, and more.

## Architecture

Decoder-only transformer, same family as GPT. Every component built from scratch.

```
Input text
  |
  v
Tokenizer              character -> integer ID (76 unique characters)
  |
  v
Character Embedding    ID -> vector of 64-384 learned features
  +
Position Embedding     position -> vector (learned, not sinusoidal)
  |
  v
Transformer Block x N
  |-- LayerNorm -> Multi-Head Self-Attention (causal mask) -> + residual
  |-- LayerNorm -> Feed-Forward (expand 4x, ReLU, compress) -> + residual
  |
  v
Final LayerNorm -> Linear -> 76 scores (one per character)
  |
  v
Next character prediction
```

- **Tokenization**: Character-level (76 unique characters, no subword/BPE)
- **Attention**: Multi-head self-attention with causal mask (each position only sees past characters)
- **Feed-Forward**: Expand to 4x embed_dim, ReLU, compress back
- **Normalization**: Pre-norm (LayerNorm before each sublayer, GPT-2 style)
- **Device**: Apple MPS, CUDA, or CPU

## Model Versions

| Version | embed_dim | Layers | Heads | Params | Val Loss | Quality |
|---------|-----------|--------|-------|--------|----------|---------|
| v1 | 64 | 4 | 4 | 226K | 1.86 | Gibberish |
| v2 | 128 | 6 | 4 | 1.2M | 1.42 | English-ish |
| v3 | 256 | 8 | 8 | 6.4M | 1.27 | Recognizable Wodehouse |
| v4 | 384 | 10 | 8 | 17.9M | 1.25 | Wodehouse-ish prose |

## Project Structure

```
model.py           # complete transformer (MultiHeadAttention, FeedForward, TransformerBlock, WodehouseGPT)
tokenizer.py       # character-level tokenizer (build_vocab, encode, decode)
train.py           # training loop (data loading, batching, loss, optimization)
generate.py        # text generation (temperature sampling, interactive/CLI modes)
jeeves             # CLI wrapper - run ./jeeves "prompt" from anywhere
demos/             # step-by-step learning demos
  attention.py           # single-head self-attention
  multihead_attention.py # multi-head attention
  feedforward.py         # expand/ReLU/compress
  embedding_demo.py      # token IDs -> learned vectors
  positional_demo.py     # adding position information
  layernorm_demo.py      # normalization + residual connections
```

## Configuration

Edit `config.py` - both `train.py` and `generate.py` read from it:

```python
embed_dim = 384
num_heads = 8
num_layers = 10
max_seq_len = 256
batch_size = 16
learning_rate = 3e-4
max_steps = 10000
```

Bigger embed_dim and more layers = better output but slower training.
