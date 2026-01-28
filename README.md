# DAb - Antibody Language Model

DAb is a masked language model (MLM) transformer designed for antibody sequence analysis and generation. It uses chain-aware attention to model paired heavy/light chain antibodies, enabling sophisticated understanding of antibody structure and function.

## Features

- **Chain-aware attention**: MINT-style hybrid self/cross-attention for modeling paired heavy/light chains
- **Information-weighted masking**: Preferentially masks CDR and non-germline positions during training
- **Flexible configuration**: Hydra-based config system with easy CLI overrides
- **Distributed training**: Multi-GPU support via Accelerate
- **Rich inference API**: Embedding extraction, masked token prediction, and logit access

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended for training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/your-org/DAb.git
cd DAb

# Install in development mode
pip install -e ".[dev]"

# Or production only (no dev dependencies)
pip install -e .
```

### Verify installation

```bash
# Check CLI is available
dab --help

# Run tests
pytest tests/
```

## Quick Start

### Training a model

```bash
# Basic training with default configuration
dab train data.train=/path/to/antibodies.csv

# Use a smaller model for testing
dab train data.train=/path/to/data.csv model=small

# Enable mixed precision training
dab train data.train=/path/to/data.csv train.mixed_precision=bf16
```

### Encoding sequences

```bash
# Extract embeddings from sequences
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.pt
```

### Python API

```python
from dab.encoding import DAbEncoder

# Load a trained model
encoder = DAbEncoder.from_pretrained("checkpoint.pt", device="cuda")

# Get sequence embeddings
embedding = encoder.encode(
    heavy_chain="EVQLVQSGAEVKKPGASVKVSCKAS",
    light_chain="DIQMTQSPSSLSASVGDRVTITC"
)
```

## Training

### Basic Training

The `dab train` command handles all training orchestration:

```bash
# Minimal training command
dab train data.train=/path/to/train.csv

# With validation data
dab train data.train=/path/to/train.csv data.eval.validation=/path/to/val.csv

# Custom output directory and experiment name
dab train \
    data.train=/path/to/train.csv \
    --output-dir ./experiments \
    --name my_experiment
```

### Training with Custom Configuration

```bash
# Use a predefined model size
dab train data.train=/path/to/data.csv model=small   # 19M params
dab train data.train=/path/to/data.csv model=base    # 99M params
dab train data.train=/path/to/data.csv model=large   # 411M params

# Use debug config for quick iteration
dab train data.train=/path/to/data.csv train=debug

# Override specific parameters
dab train \
    data.train=/path/to/data.csv \
    model=small \
    train.batch_size=64 \
    train.learning_rate=1e-4 \
    train.max_steps=50000

# Use custom config file
dab train -c /path/to/my_config.yaml data.train=/path/to/data.csv
```

### Multi-GPU Training with Accelerate

For distributed training across multiple GPUs, use `accelerate`:

```bash
# First, configure accelerate (one-time setup)
accelerate config

# Launch distributed training
accelerate launch -m dab.train data.train=/path/to/data.csv

# Specify number of GPUs
accelerate launch --num_processes=4 -m dab.train \
    data.train=/path/to/data.csv \
    model=base \
    train.batch_size=32

# Multi-node training
accelerate launch \
    --multi_gpu \
    --num_machines=2 \
    --num_processes=8 \
    -m dab.train \
    data.train=/path/to/data.csv
```

### Mixed Precision Training

```bash
# BFloat16 (recommended for Ampere+ GPUs)
dab train data.train=/path/to/data.csv train.mixed_precision=bf16

# Float16
dab train data.train=/path/to/data.csv train.mixed_precision=fp16

# With accelerate
accelerate launch --mixed_precision=bf16 -m dab.train data.train=/path/to/data.csv
```

### Resuming from Checkpoint

```bash
# Resume training from a checkpoint
dab train \
    data.train=/path/to/data.csv \
    --resume /path/to/checkpoint.pt
```

### Training with Multiple Datasets

Create a config file or use CLI overrides for weighted multi-dataset training:

```yaml
# multi_dataset.yaml
data:
  train:
    oas_paired:
      path: /data/oas_paired.parquet
      fraction: 0.7
    proprietary:
      path: /data/internal.parquet
      fraction: 0.3
```

```bash
dab train -c multi_dataset.yaml
```

### Logging with Weights & Biases

```bash
# Enable W&B logging (default)
dab train data.train=/path/to/data.csv --wandb

# Disable W&B logging
dab train data.train=/path/to/data.csv --no-wandb

# Set W&B project
dab train data.train=/path/to/data.csv log.wandb_project=my_project
```

## Inference & Encoding

### Loading a Pretrained Model

```python
from dab.encoding import DAbEncoder

# Load from checkpoint
encoder = DAbEncoder.from_pretrained(
    checkpoint_path="path/to/checkpoint.pt",
    device="cuda",  # or "cpu"
    pooling="mean"  # Pooling strategy for embeddings
)

# Check embedding dimension
print(f"Embedding dimension: {encoder.get_embedding_dim()}")
```

### Extracting Embeddings

```python
# Single sequence
embedding = encoder.encode(
    heavy_chain="EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR",
    light_chain="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLT"
)
print(f"Embedding shape: {embedding.shape}")  # (d_model,)

# Batch encoding
heavy_chains = [
    "EVQLVQSGAEVKKPGASVKVSCKAS...",
    "QVQLVQSGAEVKKPGSSVKVSCKAS...",
    "EVQLLESGGGLVQPGGSLRLSCAAS...",
]
light_chains = [
    "DIQMTQSPSSLSASVGDRVTITC...",
    "EIVLTQSPATLSLSPGERATLSC...",
    "DIQMTQSPSSLSASVGDRVTITC...",
]

embeddings = encoder.encode_batch(
    heavy_chains=heavy_chains,
    light_chains=light_chains,
    batch_size=32,
    return_numpy=False  # Return PyTorch tensor
)
print(f"Batch embeddings shape: {embeddings.shape}")  # (n_sequences, d_model)
```

### Pooling Strategies

```python
# Mean pooling (default) - average over all positions
encoder = DAbEncoder.from_pretrained("checkpoint.pt", pooling="mean")

# CLS token embedding
encoder = DAbEncoder.from_pretrained("checkpoint.pt", pooling="cls")

# Max pooling
encoder = DAbEncoder.from_pretrained("checkpoint.pt", pooling="max")

# Concatenated mean and max (2x embedding dimension)
encoder = DAbEncoder.from_pretrained("checkpoint.pt", pooling="mean_max")

# No pooling - returns full sequence embeddings
encoder = DAbEncoder.from_pretrained("checkpoint.pt", pooling="none")
embedding = encoder.encode(heavy_chain="...", light_chain="...")
# Shape: (sequence_length, d_model)
```

### Masked Token Prediction

Predict amino acids at masked positions:

```python
# Mask unknown positions with <mask>
result = encoder.predict(
    heavy_chain="EVQLVQ<mask><mask>AEVKKPGAS",
    light_chain="DIQMTQSPSSLSASVGDR"
)

print(f"Predicted heavy chain: {result['heavy_chain']}")
print(f"Predicted light chain: {result['light_chain']}")

# Get prediction probabilities
result = encoder.predict(
    heavy_chain="EVQLVQ<mask><mask>AEVKKPGAS",
    light_chain="DIQMTQSPSSLSASVGDR",
    return_probs=True
)
print(f"Heavy chain probabilities shape: {result['heavy_probs'].shape}")
```

### Getting Raw Logits

Access model logits for custom analysis:

```python
logits = encoder.get_logits(
    heavy_chain="EVQLVQSGAEVKKPGAS",
    light_chain="DIQMTQSPSSLSASVGDR"
)

print(f"Full logits shape: {logits['logits'].shape}")        # (seq_len, vocab_size)
print(f"Heavy logits shape: {logits['heavy_logits'].shape}") # (heavy_len, vocab_size)
print(f"Light logits shape: {logits['light_logits'].shape}") # (light_len, vocab_size)
```

### CLI Batch Encoding

```bash
# Encode to PyTorch format
dab encode \
    -c checkpoint.pt \
    -i sequences.csv \
    -o embeddings.pt \
    --pooling mean \
    --batch-size 64

# Encode to NumPy format
dab encode \
    -c checkpoint.pt \
    -i sequences.parquet \
    -o embeddings.npy \
    --pooling cls

# Use GPU
dab encode \
    -c checkpoint.pt \
    -i sequences.csv \
    -o embeddings.pt \
    --device cuda
```

Input file must contain `heavy_chain` and `light_chain` columns.

## Data Format

### Training Data

DAb accepts CSV, TSV, or Parquet files with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `sequence_aa:0` | Yes | Heavy chain amino acid sequence |
| `sequence_aa:1` | Yes | Light chain amino acid sequence |
| `cdr_mask_aa:0` | No | Heavy chain CDR annotation mask |
| `cdr_mask_aa:1` | No | Light chain CDR annotation mask |
| `nongermline_mask_aa:0` | No | Heavy chain non-germline positions |
| `nongermline_mask_aa:1` | No | Light chain non-germline positions |

**Example CSV:**

```csv
sequence_aa:0,sequence_aa:1,cdr_mask_aa:0,cdr_mask_aa:1
EVQLVQSGAEVKKPGAS,DIQMTQSPSSLSASVGDR,00000011100000000,000000000000000000
QVQLVQSGAEVKKPGSS,EIVLTQSPATLSLSPGER,00000011100000000,000000000000000000
```

### CDR Mask Format

CDR masks use numeric labels for each position:
- `0`: Framework region
- `1`: CDR1
- `2`: CDR2
- `3`: CDR3

### Supported Amino Acids

The tokenizer supports:
- 20 standard amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- 5 non-standard amino acids (B, J, O, U, Z)
- Gap character (`-`)
- Insertion character (`.`)

## Configuration

### Model Configurations

| Config | d_model | Layers | Heads | Parameters |
|--------|---------|--------|-------|------------|
| `small` | 256 | 24 | 4 | ~19M |
| `base` | 384 | 56 | 6 | ~99M |
| `large` | 512 | 128 | 8 | ~411M |

### Key Training Parameters

```bash
# Learning rate and optimization
train.learning_rate=3e-4
train.weight_decay=0.01
train.warmup_steps=10000

# Batch size and accumulation
train.batch_size=32
train.gradient_accumulation_steps=4  # Effective batch = 32 * 4 = 128

# Training duration
train.max_steps=100000

# Checkpointing
train.checkpoint_steps=50000
train.keep_last_n_checkpoints=5
train.save_best=true

# Evaluation
train.eval_steps=5000
```

### Masking Configuration

```bash
# Mask rate (fraction of tokens to mask)
masking.mask_rate=0.15

# Information-weighted masking
masking.use_information_weighted_masking=true
masking.cdr_weight_multiplier=2.0      # 2x weight for CDR positions
masking.nongermline_weight_multiplier=1.5  # 1.5x weight for non-germline
```

### View Model Size

```bash
# Check parameter count for different configurations
dab model-size              # Default (base) model
dab model-size model=small  # Small model
dab model-size model=large  # Large model
dab model-size model.n_layers=32 model.d_model=512  # Custom config
```

## Model Architecture

DAb uses a pre-norm transformer architecture with:

- **Rotary Position Embeddings (RoPE)**: Relative position encoding applied to attention queries and keys
- **SwiGLU Feed-Forward Networks**: Gated linear units with SiLU activation
- **Chain-Aware Attention**: Optional MINT-style hybrid attention that separately models intra-chain and inter-chain relationships
- **Pre-norm**: Layer normalization before attention and FFN blocks

### Sequence Format

Sequences are formatted as:
```
[CLS] heavy_chain light_chain [EOS]
```

Chain IDs are used internally to distinguish heavy (0) and light (1) chains for chain-aware attention.

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# With coverage
pytest tests/ --cov=dab
```

### Code Quality

```bash
# Linting
ruff check src/dab tests/

# Formatting
ruff format src/dab tests/

# Type checking
mypy src/dab/
```

## License

[Add license information here]

## Citation

[Add citation information here]
