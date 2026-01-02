# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DAb (Diffusion Antibody language model) is a discrete diffusion transformer for antibody sequence generation and analysis. It uses masked diffusion with chain-aware attention for modeling paired heavy/light chain antibodies.

## Build & Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests (19)
pytest tests/integration/    # Integration tests (5)
pytest tests/e2e/            # End-to-end tests (3)

# Run single test file
pytest tests/unit/test_tokenizer.py -v

# Run with coverage
pytest tests/ --cov=dab

# Linting and formatting
ruff check src/dab tests/
ruff format src/dab tests/

# Type checking
mypy src/dab/
```

## CLI Commands

```bash
# Training (data path specified via config override)
dab train data.train=data/train.csv                    # Basic training
dab train data.train=data.csv model=small              # Use small model config
dab train -c my_config.yaml data.train=data.csv        # Use custom config file
dab train -c /path/to/configs data.train=data.csv      # Use custom config dir

# Model size (trainable parameters)
dab model-size                                         # Default (base) model
dab model-size model=small                             # Small model config
dab model-size model.n_layers=32 model.d_model=512     # Custom overrides

# Encoding/inference
dab encode -c checkpoint.pt -i sequences.csv -o embeddings.pt
```

## Architecture

### Core Components

- **`src/dab/model/`** - Pre-norm transformer with RoPE, SwiGLU FFN, and chain-aware attention (MINT-style hybrid self/cross-attention for antibody chain modeling)
- **`src/dab/diffusion/`** - Discrete diffusion: noise schedules (linear/cosine/sqrt), masking strategies (uniform/information-weighted), sampling
- **`src/dab/data/`** - Multi-dataset support with weighted sampling, handles CSV/Parquet/PDB formats
- **`src/dab/training/`** - Trainer with Accelerate integration for distributed/mixed-precision training
- **`src/dab/encoding/`** - DAbEncoder for sequence embedding extraction with pooling strategies
- **`src/dab/eval/`** - Modular evaluation framework with metric registry

### Configuration (Hydra)

Configs in `configs/` use Hydra's compose pattern:
- `configs/config.yaml` - Main entry point, composes sub-configs
- Override via CLI: `dab train data.train=data.csv model=large train=debug`

### Key Entry Points

- `src/dab/train.py` - Training orchestration (Hydra config management)
- `src/dab/cli.py` - Click CLI with train/encode commands
- `src/dab/model/transformer.py` - DAbModel and DAbConfig

### Tokenizer

HuggingFace-style tokenizer in `src/dab/tokenizer.py` extending `PreTrainedTokenizerFast`:
- 32-token vocabulary with special tokens: `<cls>` (0), `<pad>` (1), `<eos>` (2), `<unk>` (3), `<mask>` (31)
- 20 standard amino acids + 5 non-standard + gap/insertion markers
- Module-level `tokenizer` instance for convenience

## Code Style

- Line length: 100 characters
- Python 3.10+ with type hints
- Ruff for linting/formatting, mypy for type checking
- NumPy-style docstrings

## Testing

Test fixtures in `tests/conftest.py` provide `small_config`, `small_model`, and `sample_batch` for quick testing. Use `-m "not slow"` to skip slow tests.
