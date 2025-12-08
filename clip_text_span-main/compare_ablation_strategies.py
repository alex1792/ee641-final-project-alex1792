import numpy as np
import torch
import os
import argparse
import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from collections import defaultdict
from utils.misc import accuracy

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def full_accuracy(preds, labels, locs_attributes):
    """compute full accuracy (including different combinations)"""
    locs_labels = labels.detach().cpu().numpy()
    accs = {}
    for i in [0, 1]:
        for j in [0, 1]:
            locs = np.logical_and(locs_labels == i, locs_attributes == j)
            accs[f"({i}, {j})"] = accuracy(preds[locs], labels[locs])[0] * 100
    accs[f"full"] = accuracy(preds, labels)[0] * 100
    return accs


def run_single_ablation_experiment(args, attns, mlps, classifier, labels, experiment_name, extra_args):
    """Run a single ablation experiment and return accuracy results"""
    print(f"\n{'='*80}")
    print(f"Running: {experiment_name}")
    print(f"{'='*80}")
    
    # Import here to avoid circular imports
    from my_strategy_ablation import load_heads_from_file, get_args_parser as get_ablation_parser
    
    # Parse extra args to determine what to do
    ablation_args = get_ablation_parser().parse_args(extra_args)
    ablation_args.model = args.model
    ablation_args.dataset = args.dataset
    ablation_args.input_dir = args.input_dir
    
    # Determine heads file
    # If heads_file is explicitly provided, use it directly
    if ablation_args.heads_file is not None:
        # heads_file is already set, just verify it exists
        pass
    elif ablation_args.heads_file is None and ablation_args.n_clusters is not None:
        ablation_args.heads_file = os.path.join(
            args.input_dir,
            "heads_projected_results",
            f"heads_to_ablate_projected_n{ablation_args.n_clusters}_{args.model}.txt"
        )
    elif ablation_args.heads_file is None:
        ablation_args.heads_file = os.path.join(
            args.input_dir,
            "heads_projected_results",
            f"heads_to_ablate_projected_{args.model}.txt"
        )
    
    print(f"  Looking for heads file: {ablation_args.heads_file}")
    print(f"  Strategy: {ablation_args.strategy}")
    print(f"  N_clusters: {ablation_args.n_clusters}")
    
    # Load heads
    if not os.path.exists(ablation_args.heads_file):
        print(f"⚠️  Heads file not found: {ablation_args.heads_file}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Input dir: {args.input_dir}")
        print(f"   Input dir exists: {os.path.exists(args.input_dir)}")
        if os.path.exists(args.input_dir):
            print(f"   Contents of input_dir:")
            for item in os.listdir(args.input_dir):
                print(f"     - {item}")
            heads_dir = os.path.join(args.input_dir, "heads_projected_results")
            if os.path.exists(heads_dir):
                print(f"   Contents of heads_projected_results:")
                for item in os.listdir(heads_dir):
                    print(f"     - {item}")
        return None
    
    print(f"  ✓ Heads file found")
    
    # For manual selection files, strategy should be None to read all heads
    # For strategy-based files, use the specified strategy
    load_strategy = ablation_args.strategy if ablation_args.strategy is not None else None
    heads_to_ablate = load_heads_from_file(ablation_args.heads_file, strategy=load_strategy)
    
    if not heads_to_ablate:
        print(f"⚠️  No heads found in file {ablation_args.heads_file}")
        print(f"   Let's check what's in the file:")
        with open(ablation_args.heads_file, "r") as f:
            lines = f.readlines()
            print(f"   Total lines: {len(lines)}")
            print(f"   First 20 lines:")
            for i, line in enumerate(lines[:20]):
                print(f"     {i+1}: {line.rstrip()}")
        return None
    
    print(f"  Loaded {len(heads_to_ablate)} heads to ablate")
    
    # Perform ablation
    attns_ablated = attns.copy()
    mlps_ablated = mlps.copy()
    num_layers, num_heads = attns.shape[1], attns.shape[2]
    
    # Ablate specified heads
    for layer, head in heads_to_ablate:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            attns_ablated[:, layer, head, :] = np.mean(
                attns_ablated[:, layer, head, :], axis=0, keepdims=True
            )
    
    # Ablate early layers if requested
    if ablation_args.ablate_early_layers:
        ablate_until_layer = num_layers - ablation_args.keep_last_layers
        for layer in range(ablate_until_layer):
            for head in range(num_heads):
                if (layer, head) not in heads_to_ablate:
                    attns_ablated[:, layer, head, :] = np.mean(
                        attns_ablated[:, layer, head, :], axis=0, keepdims=True
                    )
    
    # Ablate MLPs if requested
    if ablation_args.ablate_all_mlps:
        for layer in range(mlps_ablated.shape[1]):
            mlps_ablated[:, layer] = np.mean(
                mlps_ablated[:, layer], axis=0, keepdims=True
            )
    
    # Compute accuracy
    ablated = attns_ablated.sum(axis=(1, 2)) + mlps_ablated.sum(axis=1)
    ablated_acc = full_accuracy(
        torch.from_numpy(ablated @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    
    print(f"✓ Completed: {experiment_name}")
    print(f"  Accuracy: {ablated_acc}")
    
    return ablated_acc


def compare_all_experiments(args):
    """Run all comparison experiments"""
    
    # Load baseline
    print(f"\n{'='*80}")
    print("Loading baseline...")
    print(f"{'='*80}")
    
    with open(os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb") as f:
        attns = np.load(f)
    with open(os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb") as f:
        mlps = np.load(f)
    with open(os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"), "rb") as f:
        classifier = np.load(f)
    
    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    else:
        with open(os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb") as f:
            labels = np.load(f)
            labels = labels[:, :, 0]
    
    baseline = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    baseline_acc = full_accuracy(
        torch.from_numpy(baseline @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    
    print("Baseline accuracy:", baseline_acc)
    
    # Define all experiments
    experiments = {
        # Group 1: Different strategies
        "Strategy1_heads_only": ["--strategy", "1", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy2_heads_only": ["--strategy", "2", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy3_heads_only": ["--strategy", "3", "--n_clusters", str(args.n_clusters or 5)],
        
        # Group 1.5: Manual selection (from comparison directory)
        "Manual_last2_heads_only": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt")],
        "Manual_last4_heads_only": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last4_{args.model}.txt")],
        "Manual_last6_heads_only": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last6_{args.model}.txt")],
        
        # Group 2: Strategy 2 with different ablation scopes
        "Strategy2_early_layers_keep2": ["--strategy", "2", "--ablate_early_layers", "--keep_last_layers", "2", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy2_early_layers_keep4": ["--strategy", "2", "--ablate_early_layers", "--keep_last_layers", "4", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy2_mlps": ["--strategy", "2", "--ablate_all_mlps", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy2_early_layers_mlps_keep2": ["--strategy", "2", "--ablate_early_layers", "--ablate_all_mlps", "--keep_last_layers", "2", "--n_clusters", str(args.n_clusters or 5)],
        "Strategy2_early_layers_mlps_keep4": ["--strategy", "2", "--ablate_early_layers", "--ablate_all_mlps", "--keep_last_layers", "4", "--n_clusters", str(args.n_clusters or 5)],
        
        # Group 2.5: Manual selection with different ablation scopes
        "Manual_last2_early_layers_keep2": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "2"],
        "Manual_last2_early_layers_keep4": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "4"],
        "Manual_last2_mlps": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt"), "--ablate_all_mlps"],
        "Manual_last2_early_layers_mlps_keep2": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt"), "--ablate_early_layers", "--ablate_all_mlps", "--keep_last_layers", "2"],
        
        # Group 3: Different keep_last_layers values
        "Strategy2_early_layers_keep6": ["--strategy", "2", "--ablate_early_layers", "--keep_last_layers", "6", "--n_clusters", str(args.n_clusters or 5)],
        
        # Group 3.5: Manual selection - different last layers with early layers ablation
        "Manual_last2_early_layers_keep6": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last2_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "6"],
        "Manual_last4_early_layers_keep2": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last4_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "2"],
        "Manual_last4_early_layers_keep4": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last4_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "4"],
        "Manual_last6_early_layers_keep2": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last6_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "2"],
        "Manual_last6_early_layers_keep4": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last6_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "4"],
        "Manual_last6_early_layers_keep6": ["--heads_file", os.path.join(args.input_dir, "comparison", f"manually_selected_heads_last6_{args.model}.txt"), "--ablate_early_layers", "--keep_last_layers", "6"],
    }
    
    # Run all experiments
    results = {"baseline": baseline_acc}
    
    print(f"\n{'='*80}")
    print(f"Running {len(experiments)} experiments...")
    print(f"{'='*80}")
    
    for exp_name, exp_args in experiments.items():
        acc = run_single_ablation_experiment(
            args, attns, mlps, classifier, labels, exp_name, exp_args
        )
        if acc is not None:
            results[exp_name] = acc
            print(f"✓ Added {exp_name} to results")
        else:
            print(f"✗ Failed to get results for {exp_name}")
    
    print(f"\n{'='*80}")
    print(f"Results collected: {len(results)} experiments (including baseline)")
    print(f"Experiment keys: {list(results.keys())}")
    print(f"{'='*80}")
    
    # Generate comparison report and visualizations
    generate_comparison_report(results, args.output_dir, args.model, args.dataset)
    generate_visualizations(results, args.output_dir, args.model, args.dataset)


def generate_visualizations(results, output_dir, model, dataset):
    """Generate visualization plots for ablation comparison - Focused Analysis"""
    
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")
    print(f"Available results: {list(results.keys())}")
    
    # Extract metrics
    metrics = ["full", "(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
    
    # Define strategies
    strategies = ["Baseline", "Strategy1", "Strategy2", "Strategy3", "Manual(last2)", "Manual(last4)", "Manual(last6)"]
    strategy_keys = ["baseline", "Strategy1_heads_only", "Strategy2_heads_only", "Strategy3_heads_only", 
                     "Manual_last2_heads_only", "Manual_last4_heads_only", "Manual_last6_heads_only"]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22', '#16a085']
    
    # Get valid strategies
    valid_strategies = []
    valid_keys = []
    for strategy, key in zip(strategies, strategy_keys):
        if key in results:
            valid_strategies.append(strategy)
            valid_keys.append(key)
    
    # ============================================================================
    # Analysis 1: Different strategies - Full accuracy
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    valid_values = [results[key].get("full", 0) for key in valid_keys]
    valid_colors = [colors[i % len(colors)] for i, key in enumerate(strategy_keys) if key in results]
    
    if valid_strategies:
        bars = ax.bar(valid_strategies, valid_values, color=valid_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Full Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Full Accuracy: Different Strategies ({model}, {dataset})', fontsize=15, fontweight='bold')
        if valid_values:
            ax.set_ylim([min(valid_values) * 0.95, max(valid_values) * 1.05])
        ax.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars, valid_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Rotate x-axis labels if too many
        if len(valid_strategies) > 4:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        else:
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=11)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"analysis1_full_accuracy_strategies_{model}_{dataset}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Analysis 1 saved to: {plot_file}")
    plt.close()
    
    # ============================================================================
    # Analysis 2: Each strategy's performance across different metrics (4 separate plots)
    # ============================================================================
    # For each metric, create a separate plot
    metric_labels = ["Full Accuracy", "Landbird on Land (0,0)", "Landbird on Water (0,1)", 
                     "Waterbird on Land (1,0)", "Waterbird on Water (1,1)"]
    metric_keys = ["full", "(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
    
    for metric_idx, (metric_label, metric_key) in enumerate(zip(metric_labels, metric_keys)):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_values = [results[key].get(metric_key, 0) for key in valid_keys]
        
        bars = ax.bar(valid_strategies, metric_values, color=valid_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(f'{metric_label} (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_label}: Different Strategies ({model}, {dataset})', fontsize=15, fontweight='bold')
        if metric_values:
            ax.set_ylim([min(metric_values) * 0.95, max(metric_values) * 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Rotate x-axis labels if too many
        if len(valid_strategies) > 4:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        else:
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=11)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"analysis2_metric_{metric_key.replace('(', '').replace(')', '').replace(', ', '_')}_{model}_{dataset}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis 2-{metric_idx+1} ({metric_label}) saved to: {plot_file}")
        plt.close()
    
    # ============================================================================
    # Analysis 3: Manual selection different layers vs Baseline
    # ============================================================================
    manual_experiments = [
        ("Baseline", "baseline"),
        ("Manual (last2)", "Manual_last2_heads_only"),
        ("Manual (last4)", "Manual_last4_heads_only"),
        ("Manual (last6)", "Manual_last6_heads_only"),
    ]
    
    valid_manual = [(label, key) for label, key in manual_experiments if key in results]
    
    if len(valid_manual) > 1:  # At least baseline + one manual
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        num_experiments = len(valid_manual)
        width = max(0.15, min(0.25, 0.7 / num_experiments))
        
        manual_colors = ['#2ecc71', '#9b59b6', '#e67e22', '#16a085']
        
        for i, (label, key) in enumerate(valid_manual):
            values = [results[key].get(m, 0) for m in metrics]
            offset = (i - num_experiments/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, 
                         color=manual_colors[i % len(manual_colors)], alpha=0.7, edgecolor='black', linewidth=1.5)
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_title(f'Manual Selection: Different Layers vs Baseline ({model}, {dataset})', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=11, loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"analysis3_manual_vs_baseline_{model}_{dataset}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis 3 saved to: {plot_file}")
        plt.close()
    else:
        print("⚠️  Skipping Analysis 3: Not enough manual experiments found")
    
    # ============================================================================
    # Analysis 4: Different ablation scope - Full accuracy and (0,0)-(1,1) difference
    # ============================================================================
    scope_experiments = [
        ("Heads Only", "Strategy2_heads_only"),
        ("+Early(keep2)", "Strategy2_early_layers_keep2"),
        ("+Early(keep4)", "Strategy2_early_layers_keep4"),
        ("+MLPs", "Strategy2_mlps"),
        ("+Both(keep2)", "Strategy2_early_layers_mlps_keep2"),
        ("+Both(keep4)", "Strategy2_early_layers_mlps_keep4"),
        # Add Manual comparisons
        ("Manual(last2)", "Manual_last2_heads_only"),
        ("Manual+Early(keep2)", "Manual_last2_early_layers_keep2"),
        ("Manual+MLPs", "Manual_last2_mlps"),
        ("Manual+Both(keep2)", "Manual_last2_early_layers_mlps_keep2"),
    ]
    
    valid_experiments = [(label, key) for label, key in scope_experiments if key in results]
    
    if valid_experiments:
        # Plot 4a: Full accuracy
        fig, ax = plt.subplots(figsize=(14, 8))
        
        experiment_labels = [label for label, _ in valid_experiments]
        full_values = [results[key].get("full", 0) for _, key in valid_experiments]
        
        num_experiments = len(valid_experiments)
        colors_scope = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#16a085', '#d35400', '#c0392b']
        
        bars = ax.bar(experiment_labels, full_values, 
                     color=[colors_scope[i % len(colors_scope)] for i in range(num_experiments)],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Full Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Ablation Scope: Full Accuracy ({model}, {dataset})', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, full_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"analysis4a_ablation_scope_full_accuracy_{model}_{dataset}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis 4a saved to: {plot_file}")
        plt.close()
        
        # Plot 4b: (0,0) - (1,1) difference
        fig, ax = plt.subplots(figsize=(14, 8))
        
        diff_values = []
        for _, key in valid_experiments:
            val_00 = results[key].get("(0, 0)", 0)
            val_11 = results[key].get("(1, 1)", 0)
            diff_values.append(val_00 - val_11)
        
        bars = ax.bar(experiment_labels, diff_values, 
                     color=[colors_scope[i % len(colors_scope)] for i in range(num_experiments)],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax.set_ylabel('Accuracy Difference (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Ablation Scope: (0,0) - (1,1) Difference ({model}, {dataset})', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, diff_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"analysis4b_ablation_scope_diff_00_11_{model}_{dataset}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis 4b saved to: {plot_file}")
        plt.close()
    else:
        print("⚠️  Skipping Analysis 4: No ablation scope experiments found")
    
    print(f"\n{'='*80}")
    print("Visualization generation completed!")
    print(f"{'='*80}")


def create_detailed_bar_charts(results, output_dir, model, dataset, metrics):
    """Create detailed bar charts for different experiment groups - ALL BAR CHARTS"""
    
    # Chart 1: Strategy comparison - all metrics (BAR CHART)
    fig, ax = plt.subplots(figsize=(14, 8))
    strategies = {
        "Baseline": "baseline",
        "Strategy 1": "Strategy1_heads_only",
        "Strategy 2": "Strategy2_heads_only",
        "Strategy 3": "Strategy3_heads_only",
        "Manual (last2)": "Manual_last2_heads_only",
        "Manual (last4)": "Manual_last4_heads_only",
        "Manual (last6)": "Manual_last6_heads_only",
    }
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22', '#16a085']
    x = np.arange(len(metrics))
    width = 0.12  # Reduce width to fit more strategies
    
    valid_strategies = [(label, key) for label, key in strategies.items() if key in results]
    
    for i, (label, key) in enumerate(valid_strategies):
        values = [results[key].get(m, 0) for m in metrics]
        offset = (i - len(valid_strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, 
                     color=colors[i % len(colors)], alpha=0.7, edgecolor='black')
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_title(f'Strategy Comparison - All Metrics ({model}, {dataset})', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    if valid_strategies:
        ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"strategy_comparison_bar_{model}_{dataset}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Strategy comparison bar chart saved to: {plot_file}")
    plt.close()
    
    # Chart 2: Ablation scope progression (BAR CHART)
    fig, ax = plt.subplots(figsize=(14, 8))
    scope_experiments = [
        ("Heads Only", "Strategy2_heads_only"),
        ("+ Early Layers\n(keep 2)", "Strategy2_early_layers_keep2"),
        ("+ Early Layers\n(keep 4)", "Strategy2_early_layers_keep4"),
        ("+ MLPs", "Strategy2_mlps"),
        ("+ Early Layers\n+ MLPs (keep 2)", "Strategy2_early_layers_mlps_keep2"),
        ("+ Early Layers\n+ MLPs (keep 4)", "Strategy2_early_layers_mlps_keep4"),
        # Add Manual comparisons
        ("Manual(last2)\nHeads Only", "Manual_last2_heads_only"),
        ("Manual(last2)\n+ Early (keep2)", "Manual_last2_early_layers_keep2"),
        ("Manual(last2)\n+ MLPs", "Manual_last2_mlps"),
        ("Manual(last2)\n+ Both (keep2)", "Manual_last2_early_layers_mlps_keep2"),
    ]
    
    x = np.arange(len(metrics))
    width = 0.10  # Reduce width to fit more experiments
    valid_experiments = [(label, key) for label, key in scope_experiments if key in results]
    
    colors_scope = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#16a085', '#d35400', '#c0392b']
    for i, (label, key) in enumerate(valid_experiments):
        values = [results[key].get(m, 0) for m in metrics]
        offset = (i - len(valid_experiments)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, 
                     color=colors_scope[i % len(colors_scope)], alpha=0.7, edgecolor='black')
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_title(f'Ablation Scope Progression - Strategy 2 vs Manual ({model}, {dataset})', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    if valid_experiments:
        ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"ablation_scope_progression_bar_{model}_{dataset}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Ablation scope progression bar chart saved to: {plot_file}")
    plt.close()
    
    # Chart 3: Keep last layers comparison (BAR CHART)
    fig, ax = plt.subplots(figsize=(14, 8))
    keep_experiments = [
        ("Strategy2\nKeep 2", "Strategy2_early_layers_keep2"),
        ("Strategy2\nKeep 4", "Strategy2_early_layers_keep4"),
        ("Strategy2\nKeep 6", "Strategy2_early_layers_keep6"),
        # Add Manual comparisons
        ("Manual(last2)\nKeep 2", "Manual_last2_early_layers_keep2"),
        ("Manual(last2)\nKeep 4", "Manual_last2_early_layers_keep4"),
        ("Manual(last2)\nKeep 6", "Manual_last2_early_layers_keep6"),
        ("Manual(last4)\nKeep 2", "Manual_last4_early_layers_keep2"),
        ("Manual(last4)\nKeep 4", "Manual_last4_early_layers_keep4"),
        ("Manual(last6)\nKeep 2", "Manual_last6_early_layers_keep2"),
        ("Manual(last6)\nKeep 4", "Manual_last6_early_layers_keep4"),
        ("Manual(last6)\nKeep 6", "Manual_last6_early_layers_keep6"),
    ]
    
    x = np.arange(len(metrics))
    width = 0.08  # Reduce width to fit more experiments
    valid_keep = [(label, key) for label, key in keep_experiments if key in results]
    
    colors_keep = ['#e74c3c', '#f39c12', '#9b59b6', '#e67e22', '#16a085', '#d35400', '#c0392b', '#8e44ad', '#27ae60', '#2980b9', '#f1c40f']
    for i, (label, key) in enumerate(valid_keep):
        values = [results[key].get(m, 0) for m in metrics]
        offset = (i - len(valid_keep)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, 
                     color=colors_keep[i % len(colors_keep)], alpha=0.7, edgecolor='black')
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_title(f'Keep Last Layers Comparison - Strategy 2 vs Manual ({model}, {dataset})', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    if valid_keep:
        ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"keep_layers_comparison_bar_{model}_{dataset}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Keep layers comparison bar chart saved to: {plot_file}")
    plt.close()


def generate_comparison_report(results, output_dir, model, dataset):
    """Generate a comprehensive comparison report"""
    
    report_file = os.path.join(output_dir, f"ablation_comparison_{model}_{dataset}.txt")
    
    with open(report_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"ABLATION STRATEGIES COMPARISON REPORT\n")
        f.write(f"Model: {model}, Dataset: {dataset}\n")
        f.write("=" * 100 + "\n\n")
        
        # Group 1: Strategy comparison
        f.write("GROUP 1: Different Selection Strategies (Heads Only)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Metric':<15} {'Baseline':<12} {'Strategy1':<12} {'Strategy2':<12} {'Strategy3':<12} {'Manual(last2)':<15} {'Manual(last4)':<15} {'Manual(last6)':<15}\n")
        f.write("-" * 100 + "\n")
        
        if "baseline" in results:
            baseline = results["baseline"]
            for key in baseline.keys():
                f.write(f"{key:<15} {baseline[key]:>6.2f}%     ")
                for strategy in ["Strategy1_heads_only", "Strategy2_heads_only", "Strategy3_heads_only",
                                "Manual_last2_heads_only", "Manual_last4_heads_only", "Manual_last6_heads_only"]:
                    if strategy in results:
                        diff = results[strategy][key] - baseline[key]
                        f.write(f"{results[strategy][key]:>6.2f}% ({diff:+5.2f})  ")
                    else:
                        f.write(f"{'N/A':<12}  ")
                f.write("\n")
        
        f.write("\n\n")
        
        # Group 1.5: Manual selection comparison
        f.write("GROUP 1.5: Manual Selection - Different Last Layers (Heads Only)\n")
        f.write("-" * 100 + "\n")
        manual_experiments = [
            ("Manual (last 2)", "Manual_last2_heads_only"),
            ("Manual (last 4)", "Manual_last4_heads_only"),
            ("Manual (last 6)", "Manual_last6_heads_only"),
        ]
        
        f.write(f"{'Metric':<15}")
        for label, _ in manual_experiments:
            f.write(f"{label:<20}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        if "baseline" in results:
            baseline = results["baseline"]
            for key in baseline.keys():
                f.write(f"{key:<15}")
                for _, exp_key in manual_experiments:
                    if exp_key in results:
                        diff = results[exp_key][key] - baseline[key]
                        f.write(f"{results[exp_key][key]:>6.2f}% ({diff:+5.2f})      ")
                    else:
                        f.write(f"{'N/A':<20}")
                f.write("\n")
        
        f.write("\n\n")

        # Group 2: Ablation scope comparison
        f.write("GROUP 2: Different Ablation Scopes (Strategy 2)\n")
        f.write("-" * 100 + "\n")
        scope_experiments = [
            ("Heads Only", "Strategy2_heads_only"),
            ("+Early(keep2)", "Strategy2_early_layers_keep2"),
            ("+Early(keep4)", "Strategy2_early_layers_keep4"),
            ("+MLPs", "Strategy2_mlps"),
            ("+Both(keep2)", "Strategy2_early_layers_mlps_keep2"),
            ("+Both(keep4)", "Strategy2_early_layers_mlps_keep4"),
        ]
        
        f.write(f"{'Metric':<15}")
        for label, _ in scope_experiments:
            f.write(f"{label:<15}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        if "baseline" in results:
            baseline = results["baseline"]
            for key in baseline.keys():
                f.write(f"{key:<15}")
                for _, exp_key in scope_experiments:
                    if exp_key in results:
                        diff = results[exp_key][key] - baseline[key]
                        f.write(f"{results[exp_key][key]:>6.2f}% ({diff:+5.2f})  ")
                    else:
                        f.write(f"{'N/A':<15}")
                f.write("\n")
        
        f.write("\n\n")
        
        # Group 3: Keep last layers comparison
        f.write("GROUP 3: Different Keep Last Layers (Strategy 2 + Early Layers)\n")
        f.write("-" * 100 + "\n")
        keep_experiments = [
            ("Keep 2", "Strategy2_early_layers_keep2"),
            ("Keep 4", "Strategy2_early_layers_keep4"),
            ("Keep 6", "Strategy2_early_layers_keep6"),
        ]
        
        f.write(f"{'Metric':<15}")
        for label, _ in keep_experiments:
            f.write(f"{label:<15}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        if "baseline" in results:
            baseline = results["baseline"]
            for key in baseline.keys():
                f.write(f"{key:<15}")
                for _, exp_key in keep_experiments:
                    if exp_key in results:
                        diff = results[exp_key][key] - baseline[key]
                        f.write(f"{results[exp_key][key]:>6.2f}% ({diff:+5.2f})  ")
                    else:
                        f.write(f"{'N/A':<15}")
                f.write("\n")
        
        f.write("\n\n")
        
        # Summary
        f.write("SUMMARY AND INSIGHTS\n")
        f.write("-" * 100 + "\n")
        # Add analysis and insights
    
    print(f"\n✓ Comparison report saved to: {report_file}")


def visualize_at_console():
    pass

def get_args_parser():
    parser = argparse.ArgumentParser("Compare Ablation Strategies", add_help=False)
    
    parser.add_argument("--model", default="ViT-L-14", type=str)
    parser.add_argument("--dataset", default="binary_waterbirds", type=str)
    parser.add_argument("--input_dir", default="./output_dir", type=str)
    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters (default: 5)")
    
    return parser   


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    compare_all_experiments(args)