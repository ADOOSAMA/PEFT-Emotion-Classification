import argparse
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from peft import AdapterConfig
    ADAPTER_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    ADAPTER_AVAILABLE = False
    logger.warning(f"Failed to import AdapterConfig, will use custom Adapter implementation: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    encoding='utf-8'  # Force UTF-8 encoding
)

logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """Emotion classification dataset class"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_experiment_id():
    """Create experiment ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_file, log_level):
    """Set up logging"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def setup_cuda_environment():
    """Set up CUDA environment"""
    return torch.cuda.is_available()

def load_emotion_dataset(use_sample: bool = False, sample_size: int = None) -> Dict[str, Any]:
    try:
        # Official dataset loading
        logger.info("Loading boltuix/emotions-dataset from HuggingFace...")
        ds = load_dataset("boltuix/emotions-dataset")
        logger.info("Dataset loaded successfully!")
        logger.info(f"Dataset structure: {list(ds.keys())}")
        train_data = ds['train']
        all_texts = train_data['Sentence']
        all_labels_str = train_data['Label']
        unique_labels = sorted(list(set(all_labels_str)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        logger.info(f"Found emotion classes ({len(unique_labels)}):")
        for i, label in enumerate(unique_labels):
            logger.info(f"  {i}: {label}")
        # Convert labels to int
        all_labels = [label_to_id[label] for label in all_labels_str]
        sample_size = 5000 
        if sample_size is not None and sample_size > 0:
            logger.info(f"Limiting dataset size to: {sample_size} samples")
            all_texts = all_texts[:sample_size]
            all_labels = all_labels[:sample_size]
        # Split dataset (80% train, 10% val, 10% test)
        total_size = len(all_texts)
        train_size = max(1, int(0.8 * total_size))
        val_size = max(1, int(0.1 * total_size))
        if train_size + val_size >= total_size:
            val_size = max(1, total_size - train_size - 1)
            if val_size <= 0:
                val_size = 1
                train_size = max(1, total_size - 2)
        train_texts = all_texts[:train_size]
        train_labels = all_labels[:train_size]
        val_texts = all_texts[train_size:train_size+val_size]
        val_labels = all_labels[train_size:train_size+val_size]
        test_texts = all_texts[train_size+val_size:]
        test_labels = all_labels[train_size+val_size:]
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_texts)} samples")
        logger.info(f"  Validation: {len(val_texts)} samples")
        logger.info(f"  Test: {len(test_texts)} samples")
        # Show label distribution
        from collections import Counter
        train_label_dist = Counter(train_labels)
        logger.info(f"Train label distribution:")
        for label_id, count in sorted(train_label_dist.items()):
            logger.info(f"  {id_to_label[label_id]}: {count} samples")
        # Check train text and label content and length
        logger.info(f"Train text examples: {train_texts[:5]}")
        logger.info(f"Train label examples: {train_labels[:5]}")
        return {
            'train': (train_texts, train_labels),
            'validation': (val_texts, val_labels),
            'test': (test_texts, test_labels),
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(unique_labels)
        }
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("Creating demo data...")
        demo_texts = [
            "I am very happy today!",
            "This is absolutely terrible.",
            "I feel okay, nothing special.",
            "I'm so excited about this!",
            "This makes me really angry.",
            "I'm feeling quite sad.",
            "What a wonderful surprise!",
            "I hate when this happens.",
            "Life is good.",
            "This is really frustrating."
        ]
        demo_labels = [0, 1, 2, 0, 1, 3, 0, 1, 0, 1]  # 0:happy, 1:angry, 2:neutral, 3:sad
        train_size = 8
        train_texts = demo_texts[:train_size]
        train_labels = demo_labels[:train_size]
        val_texts = demo_texts[train_size:]
        val_labels = demo_labels[train_size:]
        test_texts = demo_texts[train_size:]
        test_labels = demo_labels[train_size:]
        label_names = ["happy", "angry", "neutral", "sad"]
        label_to_id = {name: i for i, name in enumerate(label_names)}
        id_to_label = {i: name for i, name in enumerate(label_names)}
        return {
            'train': (train_texts, train_labels),
            'validation': (val_texts, val_labels),
            'test': (test_texts, test_labels),
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(label_names)
        }

try:
    from peft import AdapterConfig, LoraConfig, TaskType, get_peft_model
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False

def initialize_active_adapters(model):
    """Initialize model's active_adapters attribute"""
    return model

import torch.nn as nn
class SimpleAdapterModel(nn.Module):
    def __init__(self, model_name, num_labels, adapter_size=64):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        # Freeze BERT parameters
        for param in self.bert.bert.parameters():
            param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size),
            nn.ReLU(),
            nn.Linear(adapter_size, hidden_size),
        )
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output
        adapted = self.adapter(pooled_output)
        logits = self.bert.classifier(adapted)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.bert.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

def create_models(model_name: str, num_labels: int):
    models = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = {
        'model_name': model_name,
        'num_labels': num_labels,
        'id2label': {i: str(i) for i in range(num_labels)},
        'label2id': {str(i): i for i in range(num_labels)}
    }
    # Full fine-tuning model
    full_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=config['id2label'],
        label2id=config['label2id']
    )
    full_model = initialize_active_adapters(full_model)
    models['full_finetuning'] = {
        'model': full_model,
        'tokenizer': tokenizer,
        'method': 'Full Fine-tuning',
        'trainable_params': sum(p.numel() for p in full_model.parameters() if p.requires_grad),
        'total_params': sum(p.numel() for p in full_model.parameters()),
        'trainable_percentage': (sum(p.numel() for p in full_model.parameters() if p.requires_grad) / sum(p.numel() for p in full_model.parameters())) * 100
    }
    # LoRA models
    lora_ranks = [4, 8, 16]
    for rank in lora_ranks:
        lora_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=config['id2label'],
            label2id=config['label2id']
        )
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        lora_model = get_peft_model(lora_model, lora_config)
        lora_model = initialize_active_adapters(lora_model)
        models[f'lora_r{rank}'] = {
            'model': lora_model,
            'tokenizer': tokenizer,
            'method': f'LoRA (rank={rank})',
            'trainable_params': sum(p.numel() for p in lora_model.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in lora_model.parameters()),
            'trainable_percentage': (sum(p.numel() for p in lora_model.parameters() if p.requires_grad) / sum(p.numel() for p in lora_model.parameters())) * 100,
            'lora_rank': rank
        }
    # Adapter models - minimal implementation
    adapter_sizes = [64, 128, 256]
    for size in adapter_sizes:
        try:
            adapter_model = SimpleAdapterModel(model_name, num_labels, adapter_size=size)
            trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in adapter_model.parameters())
            models[f'adapter_{size}'] = {
                'model': adapter_model,
                'tokenizer': tokenizer,
                'method': f'Adapter (size={size})',
                'trainable_params': trainable_params,
                'total_params': total_params,
                'trainable_percentage': (trainable_params / total_params) * 100,
                'adapter_size': size
            }
        except Exception as e:
            print(f"Failed to create Adapter model (size={size}): {e}")
            continue
    if not any('adapter' in key for key in models.keys()):
        logger.info("All Adapter models failed to create, continue comparing LoRA and full fine-tuning...")
    else:
        logger.info("Minimal Adapter models created successfully!")
    return models

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def train_single_model(
    model_info: dict,
    train_dataset,
    val_dataset,
    test_dataset,
    output_base_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train a single model"""
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    method_name = model_info['method']
    # Create output directory for this method
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"Start training: {method_name}")
    logger.info(f"Trainable params: {model_info['trainable_params']:,} ({model_info['trainable_percentage']:.2f}%)")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"{'='*60}")
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    lr = learning_rate  
    warmup_steps = 100  
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        learning_rate=lr,  
    )
    # Create trainer 
    eval_dataset = val_dataset if len(val_dataset) > 0 else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Start training
    start_time = datetime.now()
    logger.info(f"Start training: {method_name}")
    train_result = trainer.train()
    training_time = datetime.now() - start_time
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    # Evaluate test set
    try:
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
    except Exception as e:
        logger.warning(f"Error evaluating test set: {e}")
        test_results = {
            'eval_accuracy': 0.0,
            'eval_f1': 0.0,
            'eval_precision': 0.0,
            'eval_recall': 0.0
        }
    # Collect results
    result = {
        'method': method_name,
        'model_info': {
            'trainable_params': model_info['trainable_params'],
            'total_params': model_info['total_params'],
            'trainable_percentage': model_info['trainable_percentage']
        },
        'training_time': str(training_time),
        'train_results': train_result.metrics,
        'test_results': test_results,
        'output_dir': output_dir
    }
    # Add specific config info
    if 'lora_rank' in model_info:
        result['lora_rank'] = model_info['lora_rank']
    if 'adapter_size' in model_info:
        result['adapter_size'] = model_info['adapter_size']
    logger.info(f" {method_name} training complete!")
    logger.info(f" Training time: {training_time}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f" Test F1 score: {test_results['eval_f1']:.4f}")
    return result

def train_all_methods(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./results",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    use_sample: bool = False,
    max_length: int = 512
):
    """Train all methods for comparison"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"peft_comparison_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    # Set main log file
    log_file = os.path.join(main_output_dir, "comparison.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Starting PEFT method comparison study")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Training params: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Main output dir: {main_output_dir}")
    logger.info("Loading dataset...")
    # Load data
    logger.info("\n Loading dataset...")
    data = load_emotion_dataset(use_sample=use_sample)
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['validation']
    test_texts, test_labels = data['test']
    num_labels = data['num_labels']
    label_to_id = data['label_to_id']
    id_to_label = data['id_to_label']
    # Save label mapping
    label_mapping = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_labels': num_labels
    }
    with open(os.path.join(main_output_dir, "label_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    # Create all models
    logger.info("\n Creating all models...")
    models = create_models(model_name, num_labels)
    # Create datasets (all models share the same tokenizer)
    tokenizer = list(models.values())[0]['tokenizer']
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)
    # Train all models
    all_results = {}
    logger.info(f"\n Start training {len(models)} models...")
    for model_key, model_info in models.items():
        try:
            result = train_single_model(
                model_info=model_info,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_base_dir=main_output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            all_results[model_key] = result
        except Exception as e:
            logger.error(f" {model_info['method']} training failed: {e}")
            continue
    logger.info("\n Generating comparison report...")
    comparison_report = generate_comparison_report(all_results, id_to_label)
    # Save complete results
    final_results = {
        'experiment_info': {
            'model_name': model_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_sample': use_sample,
            'max_length': max_length,
            'timestamp': timestamp
        },
        'dataset_info': {
            'train_size': len(train_texts),
            'val_size': len(val_texts),
            'test_size': len(test_texts),
            'num_labels': num_labels
        },
        'results': all_results,
        'comparison_report': comparison_report
    }
    with open(os.path.join(main_output_dir, "complete_results.json"), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    # Print final summary
    print_final_summary(all_results)
    logger.info(f"\n All methods training complete!")
    logger.info(f"Complete results saved in: {main_output_dir}")
    return final_results

def generate_comparison_report(results: dict, id_to_label: dict) -> dict:
    """Generate method comparison report"""
    comparison = {
        'summary': {},
        'rankings': {
            'by_accuracy': [],
            'by_f1': [],
            'by_efficiency': []  # accuracy/parameter count ratio
        }
    }
    # Collect all results
    method_stats = []
    for method_key, result in results.items():
        stats = {
            'method': result['method'],
            'key': method_key,
            'accuracy': result['test_results']['eval_accuracy'],
            'f1': result['test_results']['eval_f1'],
            'trainable_params': result['model_info']['trainable_params'],
            'trainable_percentage': result['model_info']['trainable_percentage'],
            'training_time': result['training_time']
        }
        # Compute efficiency metric (accuracy/million params)
        params_millions = stats['trainable_params'] / 1_000_000
        stats['efficiency'] = stats['accuracy'] / params_millions if params_millions > 0 else 0
        method_stats.append(stats)
        comparison['summary'][method_key] = stats
    # Generate rankings
    comparison['rankings']['by_accuracy'] = sorted(method_stats, key=lambda x: x['accuracy'], reverse=True)
    comparison['rankings']['by_f1'] = sorted(method_stats, key=lambda x: x['f1'], reverse=True)
    comparison['rankings']['by_efficiency'] = sorted(method_stats, key=lambda x: x['efficiency'], reverse=True)
    return comparison

def print_final_summary(results: dict):
    """Print final summary"""
    print("\n" + "="*80)
    print(" PEFT Method Comparison - Final Results")
    print("="*80)
    # Check if there are results
    if not results:
        print("No successful training results")
        print("="*80)
        return
    # Show sorted by accuracy
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['test_results']['eval_accuracy'], 
        reverse=True
    )
    print(f"\nTest results ranking (by accuracy):")
    print(f"{'Rank':<4} {'Method':<20} {'Accuracy':<8} {'F1 Score':<8} {'Trainable Params':<12} {'Param %':<8} {'Training Time':<12}")
    print("-" * 80)
    for rank, (method_key, result) in enumerate(sorted_results, 1):
        method = result['method']
        accuracy = result['test_results']['eval_accuracy']
        f1 = result['test_results']['eval_f1']
        params = result['model_info']['trainable_params']
        percentage = result['model_info']['trainable_percentage']
        time_str = result['training_time']
        # Format parameter count
        if params >= 1_000_000:
            params_str = f"{params/1_000_000:.1f}M"
        elif params >= 1_000:
            params_str = f"{params/1_000:.1f}K"
        else:
            params_str = str(params)
        print(f"{rank:<4} {method:<20} {accuracy:.4f}   {f1:.4f}   {params_str:<12} {percentage:.1f}%     {time_str}")
    # Show best methods
    best_accuracy = sorted_results[0]
    best_efficiency = min(sorted_results, key=lambda x: x[1]['model_info']['trainable_params'])
    print(f"\nBest accuracy: {best_accuracy[1]['method']} - {best_accuracy[1]['test_results']['eval_accuracy']:.4f}")
    print(f"Best efficiency: {best_efficiency[1]['method']} - {best_efficiency[1]['model_info']['trainable_params']:,} params")
    print("\nRecommendation:")
    if best_accuracy[0] == best_efficiency[0]:
        print(f"   Recommend using {best_accuracy[1]['method']} - best accuracy and efficiency")
    else:
        print(f"   High accuracy: {best_accuracy[1]['method']}")
        print(f"   High efficiency: {best_efficiency[1]['method']}")
    print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PEFT Method Comparison Study')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Base model name')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--use_sample', action='store_true',
                       help='Use sample data for quick test')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Max sequence length')
    parser.add_argument('--methods', nargs='+', 
                       choices=['full', 'lora', 'adapter', 'all'],
                       default=['all'],
                       help='Fine-tuning methods to run')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Limit sample size of dataset')
    args = parser.parse_args()
    # Create experiment ID
    experiment_id = create_experiment_id()
    # Set up logging
    log_file = os.path.join('logs', f"peft_comparison_{experiment_id}.log")
    setup_logging(log_file, args.log_level)
    logger.info("="*80)
    logger.info("PEFT Method Comparison Study")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Fine-tuning methods: {args.methods}")
    logger.info(f"Training params: epochs={args.num_epochs}, batch_size={args.batch_size}")
    logger.info(f"Use sample data: {args.use_sample}")
    logger.info("="*80)
    try:
        # Check CUDA environment
        cuda_available = setup_cuda_environment()
        logger.info("CUDA environment: {}".format('Available' if cuda_available else 'Not available'))
        print(" Starting PEFT method comparison study...")
        print(f"Model: {args.model_name}")
        print(f" Methods: {args.methods}")
        print(f"Sample data: {'Yes' if args.use_sample else 'No'}")
        print("="*50)
        # Run comparison study
        final_results = train_all_methods(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_sample=args.use_sample,
            max_length=args.max_length
        )
        logger.info("\n" + "="*80)
        logger.info("PEFT method comparison study complete!")
        logger.info("Experiment ID: {}".format(experiment_id))
        logger.info("Complete results: {}".format(final_results['experiment_info']))
        logger.info("="*80)
        # Simplified console output
        print(f"\n Study complete! Experiment ID: {experiment_id}")
        return 0
    except Exception as e:
        logger.error(f" Study failed: {e}", exc_info=True)
        print(f" Study failed: {e}")
        return 1
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)