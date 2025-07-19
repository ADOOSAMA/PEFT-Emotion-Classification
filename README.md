# PEFT-Emotion-Classification
A research project on Parameter-Efficient Fine-Tuning (PEFT) for emotion classification, using unified hyperparameter design for fair method comparison.

## Project Objectives

Evaluate the performance and efficiency trade-offs of different PEFT methods in emotion classification tasks, including:
- **Full Fine-tuning** - Performance baseline
- **LoRA** (Low-Rank Adaptation) - Low-rank adaptation
- **Adapter Tuning** - Adapter-based fine-tuning

## Dataset

- **Dataset**: [boltuix/emotions-dataset](https://huggingface.co/datasets/boltuix/emotions-dataset)
- **Task**: 13-class emotion classification
- **Emotion Categories**: joy, sadness, anger, fear, love, surprise, thankfulness, guilt, remorse, enthusiasm, disappointment, optimism, neutral

## Project Structure

```
project7/
├── train.py                  # Main training script (core)
├── requirements.txt          # Dependencies
├── results/                  # Experimental results (auto-generated)
├── logs/                     # Log files (auto-generated)


## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

#### Basic Training
```bash
python train.py
```

#### Custom Parameter Training
```bash
python train.py \
  --model_name bert-base-uncased \
  --num_epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --max_length 512
```

### 3. Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | bert-base-uncased | Base model name |
| `--num_epochs` | 5 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 512 | Maximum sequence length |
| `--output_dir` | results | Output directory |
| `--use_sample` | False | Use sample data |

## Unified Hyperparameter Design

### Core Principles
- **Fair Comparison**: All methods use identical hyperparameters
- **Unified Learning Rate**: 2e-5 (for both full fine-tuning and PEFT methods)
- **Unified Warmup Steps**: 100 steps
- **Unified Weight Decay**: 0.01

### Hyperparameter Configuration
```python
# Unified hyperparameter settings
learning_rate = 2e-5      # Unified for all methods
warmup_steps = 100        # Unified for all methods
weight_decay = 0.01       # Unified for all methods
batch_size = 16           # Unified for all methods
num_epochs = 5            # Unified for all methods
```

## Experimental Methods

### 1. Full Fine-tuning
- Fine-tune all BERT parameters
- Serves as performance upper bound baseline
- Trainable parameters: 109.5M (100%)

### 2. LoRA (Low-Rank Adaptation)
- Add low-rank matrices to Transformer attention layers
- Test different rank values: 4, 8, 16
- Trainable parameters: 157.5K - 599.8K (0.1% - 0.5%)

### 3. Adapter Tuning
- Add small feedforward networks to each Transformer layer
- Test different adapter sizes: 64, 128, 256
- Trainable parameters: 109.1K - 404.2K (0.1% - 0.4%)

## Evaluation Metrics

- **Performance Metrics**: Accuracy, F1-score (macro average), Precision, Recall
- **Efficiency Metrics**: Number of trainable parameters, training time, model size
- **Efficiency Ratio**: Performance/parameter ratio

## Experimental Results

### Typical Results Example

| Method | Config | Accuracy | F1 Score | Trainable Params | Param % | Training Time |
|--------|--------|----------|----------|------------------|---------|---------------|
| Adapter | size=64 | 0.2300 | 0.0978 | 109.1K | 0.1% | 32:13 |
| LoRA | rank=8 | 0.2200 | 0.0813 | 304.9K | 0.3% | 29:04 |
| Adapter | size=128 | 0.2200 | 0.0800 | 207.5K | 0.2% | 25:20 |
| Adapter | size=256 | 0.2200 | 0.0793 | 404.2K | 0.4% | 18:35 |
| LoRA | rank=4 | 0.1800 | 0.0874 | 157.5K | 0.1% | 31:06 |
| Full Fine-tuning | Full | 0.1700 | 0.1082 | 109.5M | 100.0% | 41:05 |
| LoRA | rank=16 | 0.1400 | 0.0610 | 599.8K | 0.5% | 54:34 |

### Results Analysis

#### Best Performance
- **Highest Accuracy**: Adapter (size=64) - **23.00%**
- **Highest F1 Score**: Full Fine-tuning - **10.82%**

#### Parameter Efficiency
- **Most Parameter Efficient**: LoRA (rank=4) - 157.5K parameters, 0.1% parameter ratio
- **Fastest Training**: Adapter (size=256) - 18:35 minutes
- **Parameter Reduction**: PEFT methods reduce trainable parameters by over 99.5%

#### Key Findings
1. **Parameter Efficiency Advantage**: PEFT methods achieve performance close to full fine-tuning using less than 1% of parameters
2. **Adapter Method Performs Best**: Superior in both accuracy and training speed compared to LoRA

## Experimental Design

### Advantages of Unified Hyperparameters
1. **Fair Comparison**: All methods compared under identical conditions
2. **Reliable Results**: Performance differences come from methods themselves, not hyperparameters
3. **Easy Reproduction**: Other researchers can reproduce results

### Experimental Process
1. **Data Loading**: Load emotion dataset from HuggingFace
2. **Data Preprocessing**: Text tokenization and label encoding
3. **Model Creation**: Create full fine-tuning, LoRA, and Adapter models
4. **Unified Training**: Train all models with identical hyperparameters
5. **Result Evaluation**: Calculate accuracy, F1-score, and other metrics
6. **Result Saving**: Save training results and comparison reports 
