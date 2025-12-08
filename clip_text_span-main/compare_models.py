import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import argparse
from pathlib import Path
import os
import json
import pandas as pd
from collections import defaultdict

from utils.factory import create_model_and_transforms, get_tokenizer
from utils.binary_waterbirds import BinaryWaterbirds
from utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from utils.cub_classes import waterbird_classes
from compute_text_projection import zero_shot_classifier
from torchvision.datasets import CIFAR100, CIFAR10


def get_args_parser():
    parser = argparse.ArgumentParser("Compare Two Model Checkpoints", add_help=False)
    
    # Model checkpoints to compare
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first model checkpoint")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second model checkpoint")
    parser.add_argument("--model1_name", type=str, default="Model 1", help="Display name for first model")
    parser.add_argument("--model2_name", type=str, default="Model 2", help="Display name for second model")
    
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-L-14",
        type=str,
        metavar="MODEL",
        help="Name of model architecture (must match both checkpoints)",
    )
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k", type=str, help="Pretrained weights name")
    
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/shared/group/ilsvrc", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="binary_waterbirds", help="binary_waterbirds, CIFAR10, CIFAR100"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save comparison results"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for evaluation")
    
    return parser


def load_model_from_checkpoint(checkpoint_path, model_name, model_arch, pretrained, device):
    """Load a model from checkpoint file"""
    print(f"\nLoading {model_name} from {checkpoint_path}...")
    
    # Create model
    model, _, preprocess = create_model_and_transforms(
        model_arch, pretrained=pretrained
    )
    model.to(device)
    model.eval()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # PyTorch 2.6+ requires weights_only=False when loading checkpoints with non-weight objects (e.g., argparse.Namespace)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val Accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
            if 'args' in checkpoint:
                args_dict = vars(checkpoint['args']) if hasattr(checkpoint['args'], '__dict__') else checkpoint['args']
                print(f"  Training args: lambda_decorr={args_dict.get('lambda_decorr', 'unknown')}, "
                      f"last_n_layers={args_dict.get('last_n_layers', 'unknown')}")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    print(f"  Model loaded successfully!")
    
    return model, preprocess


def full_accuracy(preds, labels, scene_attributes):
    """Compute accuracy for different scene combinations (like in compare_ablation_strategies.py)
    
    Args:
        preds: predictions array (1D, already argmax-ed)
        labels: true labels array (class: 0=landbird, 1=waterbird)
        scene_attributes: scene attributes array (0=land scene, 1=water scene)
    
    Returns:
        dict with accuracies for different combinations:
        - "full": overall accuracy
        - "(0, 0)": landbird on land scene
        - "(0, 1)": landbird on water scene (spurious correlation)
        - "(1, 0)": waterbird on land scene (spurious correlation)
        - "(1, 1)": waterbird on water scene
    """
    preds = np.array(preds) if not isinstance(preds, np.ndarray) else preds
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    scene_attributes = np.array(scene_attributes) if not isinstance(scene_attributes, np.ndarray) else scene_attributes
    
    accs = {}
    
    # Calculate accuracy for each combination
    for i in [0, 1]:  # class label
        for j in [0, 1]:  # scene attribute
            mask = np.logical_and(labels == i, scene_attributes == j)
            if mask.sum() > 0:
                # Direct accuracy calculation: (preds == labels).mean()
                correct = (preds[mask] == labels[mask]).sum()
                total = mask.sum()
                accs[f"({i}, {j})"] = (correct / total) * 100
            else:
                accs[f"({i}, {j})"] = 0.0
    
    # Overall accuracy
    correct_full = (preds == labels).sum()
    total_full = len(labels)
    accs["full"] = (correct_full / total_full) * 100
    
    return accs


def evaluate_model(model, dataloader, classifier, device, scene_attributes=None):
    """Evaluate a model on a dataset
    
    Args:
        model: model to evaluate
        dataloader: data loader
        classifier: classifier weights
        device: device to use
        scene_attributes: optional array of scene attributes (for binary_waterbirds)
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.inference_mode():
        for images, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            reps = model.encode_image(images, attn_method="direct", normalize=False)
            logits = reps @ classifier
            preds = logits.argmax(dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            del reps, logits, preds, images, labels
    
    accuracy = correct / total * 100
    
    # Calculate per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    per_class_acc = {}
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() > 0:
            per_class_acc[int(label)] = (all_predictions[mask] == all_labels[mask]).mean() * 100
    
    # Calculate scene combination accuracies if scene_attributes provided
    scene_accuracies = None
    if scene_attributes is not None:
        scene_accuracies = full_accuracy(all_predictions, all_labels, scene_attributes)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'correct': correct,
        'total': total,
        'scene_accuracies': scene_accuracies  # New: scene combination accuracies
    }


def compare_predictions(pred1, pred2, labels):
    """Compare predictions between two models"""
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    labels = np.array(labels)
    
    # Agreement rate
    agreement = (pred1 == pred2).mean() * 100
    
    # Both correct
    both_correct = ((pred1 == labels) & (pred2 == labels)).sum()
    
    # Model 1 correct but Model 2 wrong
    model1_only_correct = ((pred1 == labels) & (pred2 != labels)).sum()
    
    # Model 2 correct but Model 1 wrong
    model2_only_correct = ((pred1 != labels) & (pred2 == labels)).sum()
    
    # Both wrong
    both_wrong = ((pred1 != labels) & (pred2 != labels)).sum()
    
    # Cases where predictions disagree: (0, 1) and (1, 0)
    # (0, 1): Model 1 predicts 0, Model 2 predicts 1
    case_01_mask = (pred1 == 0) & (pred2 == 1)
    case_01_count = case_01_mask.sum()
    case_01_correct_model1 = ((pred1 == labels) & case_01_mask).sum()
    case_01_correct_model2 = ((pred2 == labels) & case_01_mask).sum()
    case_01_accuracy_model1 = (case_01_correct_model1 / case_01_count * 100) if case_01_count > 0 else 0.0
    case_01_accuracy_model2 = (case_01_correct_model2 / case_01_count * 100) if case_01_count > 0 else 0.0
    # Label distribution for (0, 1) cases
    case_01_label_0 = (labels[case_01_mask] == 0).sum() if case_01_count > 0 else 0
    case_01_label_1 = (labels[case_01_mask] == 1).sum() if case_01_count > 0 else 0
    
    # (1, 0): Model 1 predicts 1, Model 2 predicts 0
    case_10_mask = (pred1 == 1) & (pred2 == 0)
    case_10_count = case_10_mask.sum()
    case_10_correct_model1 = ((pred1 == labels) & case_10_mask).sum()
    case_10_correct_model2 = ((pred2 == labels) & case_10_mask).sum()
    case_10_accuracy_model1 = (case_10_correct_model1 / case_10_count * 100) if case_10_count > 0 else 0.0
    case_10_accuracy_model2 = (case_10_correct_model2 / case_10_count * 100) if case_10_count > 0 else 0.0
    # Label distribution for (1, 0) cases
    case_10_label_0 = (labels[case_10_mask] == 0).sum() if case_10_count > 0 else 0
    case_10_label_1 = (labels[case_10_mask] == 1).sum() if case_10_count > 0 else 0
    
    return {
        'agreement_rate': agreement,
        'both_correct': both_correct,
        'model1_only_correct': model1_only_correct,
        'model2_only_correct': model2_only_correct,
        'both_wrong': both_wrong,
        'total': len(labels),
        'case_01': {
            'count': case_01_count,
            'accuracy_model1': case_01_accuracy_model1,
            'accuracy_model2': case_01_accuracy_model2,
            'label_0_count': case_01_label_0,
            'label_1_count': case_01_label_1
        },
        'case_10': {
            'count': case_10_count,
            'accuracy_model1': case_10_accuracy_model1,
            'accuracy_model2': case_10_accuracy_model2,
            'label_0_count': case_10_label_0,
            'label_1_count': case_10_label_1
        }
    }


def convert_to_json_serializable(obj):
    """Convert NumPy types and other non-JSON-serializable types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values recursively
        return {convert_to_json_serializable(key): convert_to_json_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


def generate_comparison_report(results1, results2, comparison, model1_name, model2_name, output_dir):
    """Generate a detailed comparison report"""
    report = []
    report.append("=" * 80)
    report.append("MODEL COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"\nModel 1: {model1_name}")
    report.append(f"Model 2: {model2_name}\n")
    
    # Overall accuracy comparison
    report.append("-" * 80)
    report.append("OVERALL ACCURACY")
    report.append("-" * 80)
    report.append(f"{model1_name}: {results1['accuracy']:.2f}%")
    report.append(f"{model2_name}: {results2['accuracy']:.2f}%")
    diff = results2['accuracy'] - results1['accuracy']
    report.append(f"Difference: {diff:+.2f}% ({model2_name} - {model1_name})")
    report.append("")
    
    # Per-class accuracy comparison
    report.append("-" * 80)
    report.append("PER-CLASS ACCURACY")
    report.append("-" * 80)
    all_classes = set(results1['per_class_accuracy'].keys()) | set(results2['per_class_accuracy'].keys())
    for cls in sorted(all_classes):
        acc1 = results1['per_class_accuracy'].get(cls, 0.0)
        acc2 = results2['per_class_accuracy'].get(cls, 0.0)
        diff = acc2 - acc1
        report.append(f"Class {cls}: {model1_name}={acc1:.2f}%, {model2_name}={acc2:.2f}%, Diff={diff:+.2f}%")
    report.append("")
    
    # Scene combination accuracies (like in compare_ablation_strategies.py)
    if results1.get('scene_accuracies') is not None and results2.get('scene_accuracies') is not None:
        report.append("-" * 80)
        report.append("SCENE COMBINATION ACCURACIES")
        report.append("-" * 80)
        report.append("(For binary_waterbirds: Class, Scene combinations)")
        report.append("  (0, 0): Landbird on Land Scene")
        report.append("  (0, 1): Landbird on Water Scene (spurious correlation)")
        report.append("  (1, 0): Waterbird on Land Scene (spurious correlation)")
        report.append("  (1, 1): Waterbird on Water Scene")
        report.append("")
        
        scene_combinations = ["full", "(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
        combination_labels = {
            "full": "Full Accuracy",
            "(0, 0)": "Landbird on Land (0,0)",
            "(0, 1)": "Landbird on Water (0,1) - Spurious",
            "(1, 0)": "Waterbird on Land (1,0) - Spurious",
            "(1, 1)": "Waterbird on Water (1,1)"
        }
        
        report.append(f"{'Combination':<30} {model1_name:<20} {model2_name:<20} {'Difference':<15}")
        report.append("-" * 80)
        
        for combo in scene_combinations:
            acc1 = results1['scene_accuracies'].get(combo, 0.0)
            acc2 = results2['scene_accuracies'].get(combo, 0.0)
            diff = acc2 - acc1
            label = combination_labels.get(combo, combo)
            report.append(f"{label:<30} {acc1:>6.2f}%{'':<12} {acc2:>6.2f}%{'':<12} {diff:+6.2f}%")
        
        report.append("")
    
    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    if results2['accuracy'] > results1['accuracy']:
        report.append(f"✓ {model2_name} performs better by {abs(diff):.2f}%")
    elif results1['accuracy'] > results2['accuracy']:
        report.append(f"✓ {model1_name} performs better by {abs(diff):.2f}%")
    else:
        report.append("✓ Both models perform equally")
    
    report.append(f"✓ Models agree on {comparison['agreement_rate']:.2f}% of predictions")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to: {report_path}")
    
    # Save JSON results (convert NumPy types to Python native types)
    json_results = {
        'model1': {
            'name': model1_name,
            'accuracy': results1['accuracy'],
            'per_class_accuracy': results1['per_class_accuracy'],
            'scene_accuracies': results1.get('scene_accuracies')
        },
        'model2': {
            'name': model2_name,
            'accuracy': results2['accuracy'],
            'per_class_accuracy': results2['per_class_accuracy'],
            'scene_accuracies': results2.get('scene_accuracies')
        },
        'comparison': comparison
    }
    # Convert all NumPy types to Python native types for JSON serialization
    json_results = convert_to_json_serializable(json_results)
    
    json_path = os.path.join(output_dir, "model_comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")


def main(args):
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    model1, preprocess1 = load_model_from_checkpoint(
        args.model1_path, args.model1_name, args.model, args.pretrained, args.device
    )
    model2, preprocess2 = load_model_from_checkpoint(
        args.model2_path, args.model2_name, args.model, args.pretrained, args.device
    )
    
    # Use the same preprocess (should be identical)
    preprocess = preprocess1
    
    # Load classifier weights
    tokenizer = get_tokenizer(args.model)
    if args.dataset == "binary_waterbirds":
        classes = waterbird_classes
    elif args.dataset == "CIFAR10":
        classes = CIFAR10.classes
    elif args.dataset == "CIFAR100":
        classes = CIFAR100.classes
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    classifier_path = os.path.join(args.output_dir, f"{args.dataset}_classifier_{args.model}.npy")
    if os.path.exists(classifier_path):
        print(f"\nLoading classifier from {classifier_path}")
        classifier = torch.from_numpy(np.load(classifier_path)).to(args.device)
    else:
        print("\nComputing classifier weights...")
        # Use model1 to compute classifier (should be similar for both models)
        classifier = zero_shot_classifier(
            model1, tokenizer, classes, OPENAI_IMAGENET_TEMPLATES, args.device
        )
        np.save(classifier_path, classifier.detach().cpu().numpy())
        classifier = classifier.to(args.device)
    
    # Load dataset
    scene_attributes = None
    if args.dataset == "binary_waterbirds":
        val_ds = BinaryWaterbirds(root=args.data_path, split="test", transform=preprocess)
        # Load scene attributes from metadata.csv
        metadata_path = os.path.join(args.data_path, 'metadata.csv')
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            # Filter for test split (split == 2)
            test_metadata = metadata[metadata['split'] == 2]
            # Get scene attributes (place: 0=land, 1=water)
            scene_attributes = test_metadata['place'].values
            print(f"\nLoaded scene attributes: {len(scene_attributes)} samples")
            print(f"  Land scenes (0): {(scene_attributes == 0).sum()}")
            print(f"  Water scenes (1): {(scene_attributes == 1).sum()}")
        else:
            print(f"\n⚠️  Warning: metadata.csv not found at {metadata_path}")
            print("  Scene combination accuracies will not be computed")
    elif args.dataset == "CIFAR10":
        val_ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR100":
        val_ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"\nDataset: {args.dataset}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size: {args.batch_size}")
    
    # Evaluate both models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    results1 = evaluate_model(model1, val_loader, classifier, args.device, scene_attributes=scene_attributes)
    results2 = evaluate_model(model2, val_loader, classifier, args.device, scene_attributes=scene_attributes)
    
    # Compare predictions
    comparison = compare_predictions(
        results1['predictions'], 
        results2['predictions'], 
        results1['labels']
    )
    
    # Generate report
    generate_comparison_report(
        results1, results2, comparison, 
        args.model1_name, args.model2_name, 
        args.output_dir
    )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

