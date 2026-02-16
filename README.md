# DAb

**Discrete Diffusion Antibody Language Model**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-orange)

## Overview

DAb is a discrete diffusion transformer for antibody sequence generation and analysis. It models paired heavy and light chain antibodies using masked diffusion — a process where amino acid tokens are progressively masked and the model learns to predict the original tokens at each noise level.

DAb introduces **chain-aware attention**, a hybrid self/cross-attention mechanism inspired by [MINT](https://arxiv.org/abs/2501.06755) that treats intra-chain and inter-chain interactions differently. Within each chain, self-attention uses rotary position embeddings (RoPE) to encode sequential relationships. Between chains, cross-attention operates without positional encoding, allowing the model to learn pairing-specific interactions without imposing sequential ordering across chains. Both attention paths share a single softmax normalization, enabling the model to jointly weigh intra- and inter-chain context.

DAb also supports **information-weighted masking**, which biases the diffusion noise toward CDR (complementarity-determining region) and non-germline positions — the highly variable regions of antibodies that determine antigen binding specificity. This encourages the model to allocate more capacity to the most functionally important positions.

## Key Features

- Chain-aware attention for paired heavy/light chain antibody modeling
- Multiple noise schedules (cosine, linear, sqrt, power, static/MLM)
- Information-weighted masking with CDR and non-germline prioritization
- Multiple pooling strategies for embeddings (mean, CLS, max, mean+max)
- Multi-dataset training with weighted sampling
- Distributed training and mixed precision via HuggingFace Accelerate
- Hydra-based configuration with composable presets (small / base / large)
- Region-level evaluation (per-CDR, per-framework accuracy)

## Installation

Requires Python 3.10+ and PyTorch 2.0+.

```bash
# Install from source
git clone https://github.com/brineylab/DAb.git
cd DAb
pip install -e .

# With development dependencies (testing, linting, type checking)
pip install -e ".[dev]"
```

## Quick Start

**Train a model:**

```bash
dab train data.train=data/antibodies.csv
```

**Extract embeddings from a trained model:**

```bash
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.pt --pooling mean
```

## Usage

### Data Format

DAb expects paired antibody sequence data with heavy and light chain columns. Supported file formats are CSV, TSV, and Parquet.

**Required columns:**

| Column | Description |
|--------|-------------|
| `sequence_aa:0` | Heavy chain amino acid sequence |
| `sequence_aa:1` | Light chain amino acid sequence |

**Optional columns:**

| Column | Description |
|--------|-------------|
| `cdr_mask_aa:0` / `cdr_mask_aa:1` | CDR region mask (0=framework, 1=CDR1, 2=CDR2, 3=CDR3) |
| `nongermline_mask_aa:0` / `nongermline_mask_aa:1` | Non-germline mask (0=germline, 1=non-germline) |
| `heavy_coords` / `light_coords` | CA atom coordinates for structure-aware evaluation |

Column names are configurable via the `data.*_col` config options. Masks are strings of contiguous digits with one digit per residue position (e.g., `"000011110000022200003333"`).

**Example CSV:**

```csv
sequence_aa:0,sequence_aa:1,cdr_mask_aa:0,cdr_mask_aa:1
EVQLVESGGGLVQPGGSLRL...,DIQMTQSPSSLSASVGDR...,000000000111110000000...,000000000111100000...
```

### Training

**Basic training:**

```bash
dab train data.train=/path/to/train.csv
```

**Choose a model size:**

```bash
dab train data.train=data.csv model=small    # ~24M parameters
dab train data.train=data.csv model=base     # ~124M parameters (default)
dab train data.train=data.csv model=large    # ~512M parameters
```

**Custom config overrides:**

```bash
# Use a custom config file
dab train -c my_config.yaml data.train=data.csv

# Override individual parameters
dab train data.train=data.csv train.batch_size=64 train.optimizer.learning_rate=1e-4
```

**Multi-dataset training with weighted sampling:**

Configure in a YAML file or via CLI overrides:

```yaml
# data config
data:
  train:
    oas:
      path: /data/oas_paired.parquet
      fraction: 0.7
    proprietary:
      path: /data/internal.parquet
      fraction: 0.3
```

**Resume from checkpoint:**

```bash
dab train data.train=data.csv --resume checkpoints/step_50000.pt
```

**Distributed training with Accelerate:**

```bash
accelerate launch -m dab.cli train data.train=data.csv
```

### Extracting Embeddings

**CLI:**

The input file must contain `heavy_chain` and `light_chain` columns.

```bash
# Save as PyTorch tensor
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.pt

# Save as NumPy array with mean pooling
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.npy --pooling mean

# Specify batch size and device
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.pt -b 64 -d cuda
```

**Python API:**

```python
from dab.encoding import DAbEncoder

# Load encoder from checkpoint
encoder = DAbEncoder.from_pretrained("checkpoint.pt", device="cuda", pooling="mean")

# Encode a single antibody
embedding = encoder.encode("EVQLVESGGGLVQ...", "DIQMTQSPSSLS...")

# Encode a batch
embeddings = encoder.encode_batch(
    heavy_chains=["EVQLV...", "QVQLQ..."],
    light_chains=["DIQMT...", "EIVLT..."],
    batch_size=32,
)
```

**Pooling strategies:**

| Strategy | Output Dim | Description |
|----------|-----------|-------------|
| `none` | `(seq_len, d_model)` | Full per-token embeddings |
| `mean` | `(d_model,)` | Mean over sequence positions |
| `cls` | `(d_model,)` | CLS token (first position) |
| `max` | `(d_model,)` | Max over sequence positions |
| `mean_max` | `(d_model * 2,)` | Concatenation of mean and max |

### Masked Prediction

Predict the most likely amino acids at masked positions:

```python
from dab.encoding import DAbEncoder

encoder = DAbEncoder.from_pretrained("checkpoint.pt")

result = encoder.predict(
    heavy_chain="EVQLV<mask><mask>SGGG...",
    light_chain="DIQMT...",
    return_probs=True,
)

print(result["heavy_chain"])   # Full predicted heavy chain sequence
print(result["light_chain"])   # Full predicted light chain sequence
print(result["heavy_probs"])   # Per-position prediction probabilities
```

### Likelihood Scoring

Get raw logits for sequence likelihood or perplexity computation:

```python
logits = encoder.get_logits("EVQLVESGGGLVQ...", "DIQMTQSPSSLS...")
logits["logits"]         # Full sequence logits (seq_len, vocab_size)
logits["heavy_logits"]   # Heavy chain only (heavy_len, vocab_size)
logits["light_logits"]   # Light chain only (light_len, vocab_size)
```

### Configuration

DAb uses [Hydra](https://hydra.cc/) for configuration management. The main config composes sub-configs for model, training, data, diffusion, logging, and evaluation:

```yaml
# configs/config.yaml
defaults:
  - model: base
  - train: default
  - data: default
  - diffusion: default
  - log: default
  - eval: default
```

**Model presets:**

| Preset | `d_model` | `n_layers` | `n_heads` | Parameters |
|--------|-----------|------------|-----------|------------|
| `small` | 256 | 24 | 4 | ~24M |
| `base` | 384 | 56 | 6 | ~124M |
| `large` | 512 | 128 | 8 | ~512M |

Check parameter count for any configuration:

```bash
dab model-size                                  # Default (base)
dab model-size model=small                      # Small preset
dab model-size model.n_layers=32 model.d_model=512  # Custom
```

**Common overrides:**

```bash
# Training hyperparameters
dab train data.train=data.csv train.batch_size=64 train.max_steps=200000

# Diffusion schedule
dab train data.train=data.csv diffusion.schedule_type=power diffusion.power=4.0

# Use NELBO loss instead of standard MLM loss
dab train data.train=data.csv diffusion.loss_objective=nelbo

# Disable WandB logging
dab train data.train=data.csv --no-wandb
```

## Architecture

DAb is a pre-norm transformer with tied input/output embeddings, using SwiGLU feed-forward layers and RoPE for position encoding.

```
  Heavy chain         Light chain
       |                   |
       v                   v
  [ Tokenizer: character-level amino acid tokenization ]
       |                   |
       +----> [CLS] H H H H L L L L [EOS] <----+
                       |
              chain_ids: 0 0 0 0 0 1 1 1 1 1
                       |
              [ Token + Position Embedding ]
                       |
              +--------+--------+
              |                 |
        [ Intra-chain     [ Inter-chain
          Self-Attn         Cross-Attn
          (with RoPE) ]     (no RoPE) ]
              |                 |
              +--> [ Merged Softmax ] --> [ Weighted Values ]
                       |
              [ SwiGLU Feed-Forward ]
                       |
                  x N layers
                       |
              [ LM Head (tied weights) ]
                       |
              [ Token Predictions / Embeddings ]
```

The chain-aware attention mechanism uses separate projection weights for the self-attention (intra-chain) and cross-attention (inter-chain) paths. After computing scores independently, they are merged into a single attention matrix via chain masking before the softmax, then the resulting weights are split again to select the appropriate value vectors for each path. This allows joint normalization of intra- and inter-chain attention while maintaining distinct learned representations for each type of interaction.

## Tokenizer

DAb uses a character-level tokenizer with a 32-token vocabulary (padded to a GPU-friendly multiple of 8):

- **Special tokens:** `<cls>` (0), `<pad>` (1), `<eos>` (2), `<unk>` (3), `<mask>` (31)
- **Amino acids:** 20 standard + 5 non-standard (X, B, U, O, Z) at positions 4-28
- **Structural markers:** `.` (insertion, 29), `-` (gap, 30)

Paired sequences are encoded as `[CLS] heavy light [EOS]` with a `chain_ids` tensor that marks heavy chain positions as 0 and light chain positions as 1:

```python
from dab.tokenizer import tokenizer

result = tokenizer.encode_paired("AC", "DE")
result["input_ids"]    # [0, 5, 23, 13, 9, 2]  ->  <cls> A C D E <eos>
result["chain_ids"]    # [0, 0, 0,  1, 1, 1]
```

## Citation

If you use DAb in your research, please cite:

```bibtex
@software{dab2025,
  title={DAb: Discrete Diffusion Antibody Language Model},
  author={Briney, Bryan},
  year={2025},
  url={https://github.com/brineylab/DAb}
}
```

## License

MIT
