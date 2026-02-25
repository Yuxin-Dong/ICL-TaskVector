# Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper "Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations" accepted at ICLR 2026.

## Overview

In-context learning (ICL) enables large language models (LLMs) to learn new tasks from a few examples provided in the prompt. Recent work has shown that this capability can be attributed to **task vectors**—hidden state representations that encode task-specific information. This repository provides a comprehensive framework for:

- **Extracting task vectors** from LLMs during ICL
- **Analyzing task vector functionality** across different model architectures and tasks
- **Investigating the emergence and limitations** of task vectors
- **Linear transformer experiments** for theoretical insights

## Key Features

- 🔬 **Task Vector Extraction**: Extract task vectors from various LLM architectures (LLaMA, GPT-J, Pythia)
- 📊 **Comprehensive Evaluation**: Test on 34 diverse tasks spanning knowledge, algorithmic, translation, and linguistic domains
- 🧮 **Linear Transformer Analysis**: Theoretical and empirical analysis using simplified linear transformers
- 🎨 **Visualization Tools**: Generate figures and tables for analysis

## Repository Structure

```
ICL-TaskVector/
├── icl_task_vectors/           # Main implementation
│   ├── core/                   # Core functionality
│   │   ├── task_vectors.py     # Task vector extraction and manipulation
│   │   ├── attention_saliency.py  # Attention analysis
│   │   ├── config.py           # Configuration
│   │   ├── experiments_config.py  # Experiment settings
│   │   ├── data/               # Data handling
│   │   │   ├── datasets/       # Few-shot dataset implementations
│   │   │   ├── tasks/          # Task definitions
│   │   │   └── preparation/    # Data preparation scripts
│   │   └── models/             # Model utilities
│   │       ├── context_managers/  # Forward pass modification
│   │       └── utils/          # Model loading and inference
│   └── scripts/                # Experiment scripts
│       ├── experiments/        # Main experiment runners
│       ├── figures/            # Figure generation
│       └── models/             # Model downloading/caching
├── LinearTransformer/          # Linear transformer experiments
│   ├── icl_regression.py       # Training linear transformers
│   ├── collect_result.py       # Result collection and plotting
│   └── weight_sum.py           # Weight analysis utilities
└── Explaining_Task_Vector.pdf  # Paper PDF
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for LLM experiments)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yuxin-Dong/ICL-TaskVector.git
cd ICL-TaskVector
```

2. Install dependencies:
```bash
pip install torch transformers numpy pandas matplotlib scikit-learn
```

3. Set up environment variables (optional):
```bash
# Create a .env file with your HuggingFace token if needed
HF_TOKEN=your_token_here
```

## Quick Start

### 1. Linear Transformer Experiments

Train and evaluate linear transformers to understand task vector emergence in a simplified setting:

```bash
cd LinearTransformer
python icl_regression.py --n_layer 2 --N 10 --d 4 --mode pair --seed 1
```

Key arguments:
- `--n_layer`: Number of transformer layers
- `--N`: Number of in-context examples
- `--d`: Input dimension
- `--mode`: Data processing mode (`single`, `pair`, or `triple`)

### 2. Task Vector Evaluation on LLMs

Run the main experiments on large language models:

```bash
cd icl_task_vectors
python scripts/experiments/main.py --model_type llama --model_variant 7B --num_train 10
```

This will:
1. Load the specified model and tokenizer
2. Evaluate ICL and task vector performance on all tasks
3. Save results to `outputs/results/`

### 3. Generate Figures

Create visualizations from experimental results:

```bash
cd icl_task_vectors
python scripts/figures/main.py
```

## Task Categories

The framework evaluates task vectors on 34 tasks across 5 categories:

| Category | Tasks | Description |
|----------|-------|-------------|
| **Knowledge** | 4 tasks | Factual knowledge (country capitals, languages, etc.) |
| **Algorithmic** | 4 tasks | Symbolic manipulation (next/prev letter, list operations) |
| **Translation** | 6 tasks | Language translation (EN↔FR, EN↔IT, EN↔ES) |
| **Linguistic** | 9 tasks | Linguistic transformations (tense, number, antonyms) |
| **Bijection** | 11 tasks | Composition of multiple transformations |

## Supported Models

The framework supports the following model architectures:

| Model | Variants |
|-------|----------|
| LLaMA | 7B, 13B, 30B |
| GPT-J | 6B |
| Pythia | 2.8B, 6.9B, 12B |

## Core API

### Task Vector Extraction

```python
from core.task_vectors import get_task_hiddens, modulated_generate

# Extract task vectors from few-shot examples
task_hiddens = get_task_hiddens(model, tokenizer, task, datasets)

# Generate predictions using task vectors
predictions = modulated_generate(
    model, tokenizer, task, test_datasets, 
    task_hiddens=task_hiddens, 
    intermediate_layer=best_layer
)
```

### Running ICL Baseline

```python
from core.task_vectors import run_icl

# Standard in-context learning
predictions = run_icl(model, tokenizer, task, test_datasets)
```

### Task Vector with Multiple Contexts

```python
from core.task_vectors import run_multi_task_vector

# Use multiple task vectors for improved performance
predictions, accuracy_by_layer, task_hiddens = run_multi_task_vector(
    model, tokenizer, task, test_datasets, dev_datasets, multiple_dataset=True
)
```

## Reproducing Paper Results

To reproduce the main results from the paper:

1. **Download models** (if not already cached):
```bash
python icl_task_vectors/scripts/models/download.py
```

2. **Run main experiments** for all models:
```bash
cd icl_task_vectors
python scripts/experiments/main.py --experiment_id camera_ready --num_train 10 --num_valid 0
```

3. **Collect and plot results**:
```bash
python scripts/figures/main.py
```

Results will be saved to `outputs/figures/`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{dong2026understanding,
  title={Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations},
  author={Yuxin Dong and Jiachen Jiang and Zhihui Zhu and Xia Ning},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=CLBVilFk7N}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work was accepted at ICLR 2026
- We thank the open-source community for the transformers library and pre-trained models

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
