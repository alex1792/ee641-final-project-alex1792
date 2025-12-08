import numpy as np
import torch
import os
import argparse
from pathlib import Path
from collections import Counter
import tqdm
from utils.misc import accuracy
from compute_complete_text_set import replace_with_iterative_removal


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


def get_args_parser():
    parser = argparse.ArgumentParser("Select Heads and Run Ablation", add_help=False)
    
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="Model name (default: ViT-L-14)",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary_waterbirds",
        help="Dataset name (default: binary_waterbirds)",
    )
    
    # path parameters
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output_dir",
        help="Input directory for data files (default: output_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Output directory for results (default: output_dir)",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default="text_descriptions",
        help="Text descriptions directory (default: text_descriptions)",
    )
    parser.add_argument(
        "--text_descriptions",
        type=str,
        default="image_descriptions_general",
        help="Text descriptions file name (default: image_descriptions_general)",
    )
    
    # analysis parameters
    parser.add_argument(
        "--num_last_layers",
        type=int,
        default=2,
        help="Number of last layers to analyze (default: 2)",
    )
    parser.add_argument(
        "--w_ov_rank",
        type=int,
        default=80,
        help="SVD rank for projection (default: 80)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    
    # Classifier file option (for fine-tuned models)
    parser.add_argument(
        "--classifier_file",
        type=str,
        default=None,
        help="Custom classifier file path (if not provided, will use {dataset}_classifier_{model}.npy)",
    )
    
    # Ablation options
    parser.add_argument(
        "--ablate_early_layers",
        action="store_true",
        help="Also ablate early layers (like original compute_use_specific_heads.py)",
    )
    parser.add_argument(
        "--ablate_all_mlps",
        action="store_true",
        help="Also ablate all MLP layers",
    )
    parser.add_argument(
        "--keep_last_layers",
        type=int,
        default=4,
        help="Keep last N layers when ablating early layers (default: 4)",
    )
    
    return parser


def main(args):
    print("=" * 80)
    print("SELECT HEADS AND RUN ABLATION")
    print("=" * 80)
    
    # 1. Load attention features
    attn_path = os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy")
    print(f"\nLoading attention features from: {attn_path}")
    with open(attn_path, "rb") as f:
        attns = np.load(f)  # [N, 24, 16, 1024]
    
    num_samples, num_layers, num_heads, embedding_dim = attns.shape
    print(f"Model structure: {num_layers} layers, {num_heads} heads per layer")
    
    # 2. Determine layers to analyze
    layers = list(range(num_layers - args.num_last_layers, num_layers))
    print(f"Analyzing last {args.num_last_layers} layers: {layers}")
    
    # Create dictionaries to store head features and text spans
    head_features_dict = {}
    head_text_spans = {}
    
    # 3. Load text features and text lines
    text_features_path = os.path.join(
        args.input_dir, f"{args.text_descriptions}_{args.model}.npy"
    )
    print(f"Loading text features from: {text_features_path}")
    with open(text_features_path, "rb") as f:
        text_features = np.load(f)  # [num_texts, 1024]
    
    text_file_path = os.path.join(args.text_dir, f"{args.text_descriptions}.txt")
    print(f"Loading text descriptions from: {text_file_path}")
    with open(text_file_path, "r") as f:
        text_lines = [line.replace("\n", "") for line in f.readlines()]
    
    # 4. Compute text span for each head
    print(f"\nComputing top 1 text span for all heads in last {args.num_last_layers} layers...")
    for layer in layers:
        for head in range(num_heads):
            # Extract features of this head
            head_features = attns[:, layer, head]  # [N, 1024]
            head_features_dict[(layer, head)] = head_features
            
            # Compute text span (only top 1)
            reconstruct, text_span = replace_with_iterative_removal(
                head_features,
                text_features,
                text_lines,
                1,  # iters: only top 1
                args.w_ov_rank,  # rank: SVD rank
                args.device
            )
            
            top1_text = text_span[0] if len(text_span) > 0 else "N/A"
            head_text_spans[(layer, head)] = top1_text
    
    print(f"✓ Computation completed, {len(head_text_spans)} heads")
    
    # 5. Interactive selection of heads
    print(f"\n{'='*80}")
    print("INTERACTIVE HEAD SELECTION")
    print(f"{'='*80}")
    print("You will be shown each head's text span and asked whether to ablate it.")
    print("Commands:")
    print("  'y' or 'yes' - Ablate this head")
    print("  'n' or 'no'  - Keep this head")
    print("  'q' or 'quit' - Quit and run ablation with current selection")
    print("  'a' or 'abort' - Abort and clear all selections")
    print("  's' or 'show' - Show all selected heads so far")
    print("=" * 80)
    
    selected_heads = []
    skipped_heads = []
    head_list = list(head_features_dict.keys())
    total_heads = len(head_list)
    
    print(f"\nReviewing {total_heads} heads...\n")
    
    for idx, (layer, head) in enumerate(head_list, 1):
        text_span = head_text_spans[(layer, head)]
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{total_heads}] Layer {layer}, Head {head}")
        print(f"{'='*80}")
        print(f"Text span: {text_span}")
        print(f"\nCurrent selection: {len(selected_heads)} heads selected, {len(skipped_heads)} skipped")
        
        # Wait for user input
        user_input = None
        confirm = None
        
        while True:
            try:
                user_input = input(f"\n  Ablate this head? [y/n/q/a/s]: ").strip().lower()
                
                if user_input in ['y', 'yes']:
                    selected_heads.append((layer, head))
                    print(f"  ✓ Added to ablation list (total: {len(selected_heads)})")
                    break
                elif user_input in ['n', 'no']:
                    skipped_heads.append((layer, head))
                    print(f"  - Skipped")
                    break
                elif user_input in ['q', 'quit']:
                    print(f"\n  Quitting selection.")
                    print(f"  Selected {len(selected_heads)} heads so far.")
                    confirm = input(f"  Run ablation with current selection? [y/n]: ").strip().lower()
                    if confirm in ['y', 'yes']:
                        break
                    else:
                        # Continue selection
                        continue
                elif user_input in ['a', 'abort']:
                    print(f"\n  Aborting all selections.")
                    confirm = input(f"  Clear all {len(selected_heads)} selected heads? [y/n]: ").strip().lower()
                    if confirm in ['y', 'yes']:
                        selected_heads = []
                        print(f"  ✓ All selections cleared")
                        break
                    else:
                        continue
                elif user_input in ['s', 'show']:
                    if selected_heads:
                        print(f"\n  Currently selected heads ({len(selected_heads)}):")
                        for l, h in selected_heads:
                            print(f"    Layer {l}, Head {h}: {head_text_spans[(l, h)]}")
                    else:
                        print(f"  No heads selected yet.")
                    continue
                else:
                    print(f"  Invalid input. Please enter 'y', 'n', 'q', 'a', or 's'")
            except KeyboardInterrupt:
                print(f"\n\n  ⚠️  Interrupted by user (Ctrl+C)")
                print(f"  Selected {len(selected_heads)} heads so far.")
                save_now = input(f"  Run ablation with current selection? [y/n]: ").strip().lower()
                if save_now in ['y', 'yes']:
                    break
                else:
                    print(f"  Continuing...")
                    continue
        
        # If user chose to quit, break the loop
        if user_input in ['q', 'quit'] and confirm in ['y', 'yes']:
            break
        if user_input in ['a', 'abort'] and confirm in ['y', 'yes']:
            # Continue, but selections are cleared
            pass
    
    # 6. Summary of selection
    print(f"\n{'='*80}")
    print(f"SELECTION SUMMARY")
    print(f"{'='*80}")
    print(f"  Selected for ablation: {len(selected_heads)} heads")
    print(f"  Skipped: {len(skipped_heads)} heads")
    if selected_heads:
        print(f"\n  Selected heads:")
        for layer, head in selected_heads:
            print(f"    Layer {layer}, Head {head}: {head_text_spans[(layer, head)]}")
    else:
        print("\n  ⚠️  No heads selected! Ablation will not be performed.")
        return
    print()
    
    # 7. Run ablation
    print(f"{'='*80}")
    print("RUNNING ABLATION")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns_ablated = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb"
    ) as f:
        mlps_ablated = np.load(f)  # [b, l+1, d]
    # Load classifier (use custom file if provided, otherwise use default naming)
    if args.classifier_file:
        classifier_path = args.classifier_file
        if not os.path.isabs(classifier_path):
            # If relative path, check in input_dir first, then current directory
            if os.path.exists(os.path.join(args.input_dir, classifier_path)):
                classifier_path = os.path.join(args.input_dir, classifier_path)
            elif not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
    else:
        classifier_path = os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy")
    
    print(f"Loading classifier from: {classifier_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
    
    with open(classifier_path, "rb") as f:
        classifier = np.load(f)
    
    # Load labels
    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns_ablated.shape[0])])
    else:
        with open(
            os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb"
        ) as f:
            labels = np.load(f)
            labels = labels[:, :, 0]
    
    # Compute baseline
    print("Computing baseline accuracy...")
    baseline = attns_ablated.sum(axis=(1, 2)) + mlps_ablated.sum(axis=1)
    baseline_acc = full_accuracy(
        torch.from_numpy(baseline @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Baseline:", baseline_acc)
    
    # Perform ablation
    print(f"\nPerforming ablation on {len(selected_heads)} heads...")
    
    # 7.1 Ablate specified heads
    ablated_count = 0
    for layer, head in selected_heads:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            attns_ablated[:, layer, head, :] = np.mean(
                attns_ablated[:, layer, head, :], axis=0, keepdims=True
            )
            ablated_count += 1
        else:
            print(f"⚠️  Invalid head: Layer {layer}, Head {head}")
    
    print(f"✓ Ablated {ablated_count} specified heads")
    
    # 7.2 Ablate early layers (optional)
    if args.ablate_early_layers:
        ablate_until_layer = num_layers - args.keep_last_layers
        early_ablated = 0
        for layer in range(ablate_until_layer):
            for head in range(num_heads):
                if (layer, head) not in selected_heads:  # Avoid duplicate
                    attns_ablated[:, layer, head, :] = np.mean(
                        attns_ablated[:, layer, head, :], axis=0, keepdims=True
                    )
                    early_ablated += 1
        print(f"✓ Ablated {early_ablated} early layer heads (keeping last {args.keep_last_layers} layers)")
    
    # 7.3 Ablate all MLPs (optional)
    if args.ablate_all_mlps:
        for layer in range(mlps_ablated.shape[1]):
            mlps_ablated[:, layer] = np.mean(
                mlps_ablated[:, layer], axis=0, keepdims=True
            )
        print(f"✓ Ablated all {mlps_ablated.shape[1]} MLP layers")
    
    # 8. Compute ablated accuracy
    print(f"\nComputing ablated accuracy...")
    ablated = attns_ablated.sum(axis=(1, 2)) + mlps_ablated.sum(axis=1)
    ablated_acc = full_accuracy(
        torch.from_numpy(ablated @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Ablated:", ablated_acc)
    
    # 9. Calculate and display differences
    print(f"\n{'='*80}")
    print("ACCURACY CHANGE")
    print(f"{'='*80}")
    print("\naccuracy explanation:")
    print("  (0, 0): Landbird on Land (normal combination)")
    print("  (0, 1): Landbird on Water (spurious combination)")
    print("  (1, 0): Waterbird on Land (spurious combination)")
    print("  (1, 1): Waterbird on Water (normal combination)")
    print("  full:   overall accuracy\n")
    
    for key in baseline_acc.keys():
        diff = ablated_acc[key] - baseline_acc[key]
        if key in ["(0, 1)", "(1, 0)"]:
            marker = " ⚠️ " if diff < 0 else " ✓ "
            print(f"  {key:10s}: {baseline_acc[key]:6.2f}% -> {ablated_acc[key]:6.2f}% (Δ {diff:+6.2f}%){marker}Spurious")
        else:
            print(f"  {key:10s}: {baseline_acc[key]:6.2f}% -> {ablated_acc[key]:6.2f}% (Δ {diff:+6.2f}%)")
    
    # 10. Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save selected heads
    heads_file = os.path.join(
        args.output_dir,
        f"manually_selected_heads_last{args.num_last_layers}_{args.model}.txt"
    )
    with open(heads_file, "w") as f:
        f.write(f"# Manually selected heads for ablation (last {args.num_last_layers} layers)\n")
        f.write(f"# Model: {args.model}\n")
        f.write("# Format: Layer,Head\n\n")
        for layer, head in selected_heads:
            f.write(f"{layer},{head}\n")
    print(f"✓ Selected heads saved to: {heads_file}")
    
    # Save ablation results
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_manual_ablation_results_{args.model}.txt"
    )
    with open(result_file, "w") as f:
        f.write(f"Ablation Results for {args.dataset} - {args.model}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Selection method: Manual interactive selection\n")
        f.write(f"Heads ablated: {len(selected_heads)}\n")
        f.write(f"Early layers ablated: {args.ablate_early_layers}\n")
        f.write(f"MLPs ablated: {args.ablate_all_mlps}\n")
        f.write("\nSelected heads:\n")
        for layer, head in selected_heads:
            f.write(f"  Layer {layer}, Head {head}: {head_text_spans[(layer, head)]}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Baseline Accuracy:\n")
        for key, value in baseline_acc.items():
            f.write(f"  {key}: {value:.2f}%\n")
        f.write("\nAblated Accuracy:\n")
        for key, value in ablated_acc.items():
            f.write(f"  {key}: {value:.2f}%\n")
        f.write("\nChange:\n")
        for key in baseline_acc.keys():
            diff = ablated_acc[key] - baseline_acc[key]
            f.write(f"  {key}: {diff:+.2f}%\n")
    
    print(f"✓ Ablation results saved to: {result_file}")
    
    print(f"\n{'='*80}")
    print("COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)