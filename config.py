"""
Model and training configuration. Edit this file to change settings.
Both train.py and generate.py read from here.
"""

# Model
vocab_size = 4000
embed_dim = 384
num_heads = 8
num_layers = 10
max_seq_len = 256
dropout = 0.1

# Training
batch_size = 16
learning_rate = 3e-4
max_steps = 10000
eval_interval = 1000
