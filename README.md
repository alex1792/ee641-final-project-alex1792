# EE641 Final Project

**Quick Video Demo:** [Click Me](https://www.youtube.com/watch?v=CA7O84SmFRs&t=15s)

This project implements and compares **ViT-DeCV (Vision Transformer with Decorrelation)** performance across different datasets, integrating CLIP text decomposition techniques for visual representation analysis.

## Project Core

**The main contributions of this project are located in:**

1. **`ipynb/`** - **Core experimental notebooks** containing the main ViT-DeCV implementations and experiments
2. **`clip_text_span-main/`** - **Extended codebase** with custom implementations for decorrelation training and ablation strategies

## Project Structure

```text
.
├── ipynb/                        # CORE: Main experimental notebooks
│   ├── cifar10_and_cifar100_script.ipynb  # ViT-DeCV on CIFAR-10/100
│   └── waterbird_script.ipynb              # ViT-DeCV on Waterbirds with ablation
│
├── clip_text_span-main/          # Extended CLIP codebase with custom additions
│   ├── train_with_decorrelation.py        # Custom: Training with decorrelation loss
│   ├── my_strategy_ablation.py            # Custom: Ablation strategy implementation
│   ├── prs_hook.py                        # Custom: Modified PRS hook for training mode
│   ├── cluster_attention_heads.py         # Custom: Attention head clustering
│   ├── analyze_attention_head_embeddings.py  # Custom: Attention head analysis
│   ├── compute_*.py                        # Original CLIP decomposition scripts
│   └── utils/                              # Utility functions
│
├── figures/                      # Project-related images and visualization results
├── slides_and_report/            # Presentation and report documents
└── README.md                     # This file
```

## Quick Start

### Main Experiments (Start Here!)

The core experiments are in the Jupyter notebooks:

#### 1. CIFAR-10/100 Experiments

```bash
# Open Jupyter notebook
jupyter notebook ipynb/cifar10_and_cifar100_script.ipynb
```

**What it does:**

- Implements ViT-DeCV (Vision Transformer with decorrelation loss)
- Compares performance between base model and decorrelation model
- Supports CIFAR-10 (10 classes) and CIFAR-100 (100 classes) classification tasks
- Adjustable decorrelation loss weight (`lambda_decorr`)
- Configurable number of unfrozen layers (`defreeze_layer`)

#### 2. Waterbirds Experiments

```bash
# Download dataset first (if not already downloaded)
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xf waterbird_complete95_forest2water2.tar.gz

# Open Jupyter notebook
jupyter notebook ipynb/waterbird_script.ipynb
```

**What it does:**

- Classification task on Waterbirds dataset
- Integration of diversity loss
- User-guided attention head ablation
- Integration with CLIP text decomposition techniques

## Custom Code Contributions

The following files in `clip_text_span-main/` are **custom implementations** added for this project:

### Core Training Components

- **`train_with_decorrelation.py`**:
  - Main training script with decorrelation loss support
  - Memory-efficient training with mixed precision support
  - Supports CIFAR-10, CIFAR-100, and Waterbirds datasets
  - Configurable layer freezing and decorrelation loss weights

- **`prs_hook.py`**:
  - Modified PRS (Projected Residual Stream) logger
  - Added training mode support with gradient preservation
  - Memory optimization: only saves last N layers during training
  - Enables decorrelation loss computation during training

### Ablation and Analysis Tools

- **`my_strategy_ablation.py`**:
  - Custom ablation strategy implementation
  - Loads and applies head ablation strategies
  - Integrates with CLIP text decomposition framework

- **`cluster_attention_heads.py`**:
  - Clusters attention heads based on projected features
  - Uses SVD for dimensionality reduction
  - Generates ablation head lists based on clustering results

- **`analyze_attention_head_embeddings.py`**:
  - Analyzes attention head embeddings
  - Provides insights into head representations
  - Supports ablation strategy selection

### Usage of Custom Training Script

```bash
cd clip_text_span-main

# Train with decorrelation loss on CIFAR-10
python train_with_decorrelation.py \
    --dataset CIFAR10 \
    --device cuda:0 \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k \
    --lambda_decorr 0.1 \
    --last_n_layers 2 \
    --train_last_n_layers 4 \
    --batch_size 8 \
    --epochs 10 \
    --use_amp

# Train on Waterbirds dataset
python train_with_decorrelation.py \
    --dataset binary_waterbirds \
    --data_path /path/to/waterbirds \
    --device cuda:0 \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k \
    --lambda_decorr 0.1 \
    --last_n_layers 2
```

## Environment Setup

### 1. Install Dependencies

The project uses Conda for environment management. Please install Conda first, then run:

```bash
cd clip_text_span-main
conda env create -f environment.yml
conda activate prsclip
```

### 2. Install Additional Packages (for Jupyter notebooks)

```bash
pip install tqdm transformers datasets torchinfo
```

## Dataset Download

### ImageNet Segmentation Dataset (.mat file)

**Important**: The `gtsegs_ijcv.mat` file is very large and will not be included in the Git repository. If you need to run ImageNet segmentation evaluation, please download it manually:

```bash
cd clip_text_span-main
mkdir -p imagenet_seg
cd imagenet_seg
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
```

**Note**:

- This file is only required when running `compute_segmentations.py`
- If you only run other experiments (CIFAR-10/100, Waterbirds), you don't need to download this file
- `.gitignore` is configured to ignore all `.mat` files to prevent accidental uploads to GitHub

### Waterbirds Dataset

To run Waterbirds experiments, please download the dataset:

```bash
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xf waterbird_complete95_forest2water2.tar.gz
```

## Directory Usage Guide

### `ipynb/` - Core Experimental Notebooks

**This is the main focus of the project.** The notebooks contain:

- Complete ViT-DeCV implementation with decorrelation loss
- Training and evaluation pipelines
- Model comparison and analysis
- Integration with CLIP text decomposition
- Ablation studies and experiments

### `clip_text_span-main/` - Extended Codebase

This directory contains both original CLIP decomposition code and custom additions:

#### Custom Files (Added for this project)

- `train_with_decorrelation.py` - Training script with decorrelation loss
- `my_strategy_ablation.py` - Custom ablation strategies
- `prs_hook.py` - Modified PRS hook for training
- `cluster_attention_heads.py` - Attention head clustering
- `analyze_attention_head_embeddings.py` - Head embedding analysis

#### Original CLIP Decomposition Scripts

- `compute_prs.py`: Compute Projected Residual Stream (PRS) components
- `compute_text_projection.py`: Convert text labels to CLIP text representations
- `compute_segmentations.py`: ImageNet segmentation evaluation
- `compute_ablations.py`: Verify mean-ablations effects
- `compute_complete_text_set.py`: Find meaningful directions for all attention heads
- `compare_models.py`: Compare performance across different models

#### Supporting Directories

- **`utils/`**: Model configurations, dataset processing, CLIP implementations
- **`text_descriptions/`**: Text description files for CLIP
- **`output_dir/`**: Stores computation results and output files
- **`images/`**: Project-related images and visualization results

### `figures/`

Stores project-related images and visualization results:

- Architecture diagrams (`architecture_and_loss.jpg/pdf`)
- Loss calculation explanations (`loss.jpg`, `loss_detail.jpg`)
- Other experimental result visualizations

### `slides_and_report/`

Stores project reports and presentations:

- `EE641 Final Project Report.docx`: Project report
- `EE641_final_project_presentation.pdf/pptx`: Project presentation

## Important Notes

1. **Large Files**: `.mat` files (e.g., `gtsegs_ijcv.mat`) are not uploaded to GitHub. Please download them manually when needed.
2. **GPU Requirements**: Most experiments require GPU support. Please ensure CUDA environment is properly configured.
3. **Data Paths**: Please verify data path settings are correct when running scripts.
4. **Environment Variables**: Some scripts may require specific environment variables or data paths to be set.

## References

- CLIP Text Decomposition Original Project: [GitHub](https://github.com/yossigandelsman/clip_text_span)
- Paper: [Interpreting CLIP's Image Representation via Text-Based Decomposition](https://arxiv.org/abs/2310.05916)

## License

Please refer to `clip_text_span-main/LICENSE.txt` for license information.
