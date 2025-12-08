import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import random
import tqdm
from utils.misc import accuracy


def full_accuracy(preds, labels, locs_attributes):
    locs_labels = labels.detach().cpu().numpy()
    accs = {}
    for i in [0, 1]:
        for j in [0, 1]:
            locs = np.logical_and(locs_labels == i, locs_attributes == j)
            accs[f"({i}, {j})"] = accuracy(preds[locs], labels[locs])[0] * 100
    accs[f"full"] = accuracy(preds, labels)[0] * 100
    return accs


def load_heads_from_file(file_path, strategy=None):
    """
    load heads list from file
    
    Args:
        file_path: file path
        strategy: which strategy to use ('1', '2', '3', or None for all)
    
    Returns:
        list of tuples: [(layer, head), ...]
    """
    heads = []
    current_strategy = None
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # skip comments and empty lines
            if not line or line.startswith("#"):
                # check if it is a strategy marker
                if "Strategy 1" in line or "strategy 1" in line.lower():
                    current_strategy = "1"
                elif "Strategy 2" in line or "strategy 2" in line.lower():
                    current_strategy = "2"
                elif "Strategy 3" in line or "strategy 3" in line.lower():
                    current_strategy = "3"
                continue
            
            # parse layer,head
            if "," in line:
                try:
                    layer, head = map(int, line.split(","))
                    # if strategy is specified, only read the corresponding heads
                    if strategy is None or current_strategy == strategy:
                        heads.append((layer, head))
                except ValueError:
                    continue
    
    return heads


def get_args_parser():
    parser = argparse.ArgumentParser("Custom Strategy Ablation", add_help=False)
    
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-L-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--figures_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary_waterbirds",
        help="imagenet, waterbirds, waterbirds_binary or cub",
    )
    
    # Ablation parameters
    parser.add_argument(
        "--heads_file",
        type=str,
        default=None,
        help="Path to heads file. If not specified, will auto-generate from --n_clusters and --model. Example: output_dir/heads_to_ablate_projected_n5_ViT-L-14.txt",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (used to auto-generate heads_file if --heads_file not specified). Example: --n_clusters 5 will use heads_to_ablate_projected_n5_{model}.txt",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["1", "2", "3", None],
        help="Which strategy to use (1, 2, 3, or None for all)",
    )
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
    # 1. determine heads file path
    if args.heads_file is None:
        # auto-generate file name
        if args.n_clusters is not None:
            # use cluster number to generate file name (matching cluster_attention_heads.py's output format)
            heads_file = os.path.join(
                args.input_dir,
                f"heads_to_ablate_projected_n{args.n_clusters}_{args.model}.txt"
            )
            print(f"Auto-generating heads file from n_clusters={args.n_clusters}: {heads_file}")
        else:
            # use default file name (backward compatible)
            heads_file = os.path.join(
                args.input_dir,
                f"heads_to_ablate_projected_{args.model}.txt"
            )
            print(f"Using default heads file: {heads_file}")
    else:
        heads_file = args.heads_file
        print(f"Using specified heads file: {heads_file}")
    
    # 2. load heads list
    print(f"\nLoading heads from: {heads_file}")
    if not os.path.exists(heads_file):
        raise FileNotFoundError(
            f"Heads file not found: {heads_file}\n"
            f"Please specify --heads_file or use --n_clusters to auto-generate the path."
        )
    
    heads_to_ablate = load_heads_from_file(heads_file, strategy=args.strategy)
    
    if not heads_to_ablate:
        print(f"⚠️  No heads found for strategy {args.strategy}")
        if args.strategy:
            print(f"   Try running without --strategy to see all available heads")
        return
    
    print(f"✓ Loaded {len(heads_to_ablate)} heads to ablate")
    if len(heads_to_ablate) <= 10:
        print(f"   Heads: {heads_to_ablate}")
    else:
        print(f"   First 10 heads: {heads_to_ablate[:10]}")
        print(f"   ... and {len(heads_to_ablate) - 10} more")
    
    # 3. load data
    print(f"\nLoading data...")
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb"
    ) as f:
        mlps = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    
    num_layers, num_heads = attns.shape[1], attns.shape[2]
    print(f"✓ Data loaded: {attns.shape[0]} samples, {num_layers} layers, {num_heads} heads/layer")
    
    # 4. load labels
    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    else:
        with open(
            os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb"
        ) as f:
            labels = np.load(f)
            labels = labels[:, :, 0]
    
    # 5. compute baseline
    print(f"\n{'='*80}")
    print("Computing baseline accuracy...")
    baseline = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    baseline_acc = full_accuracy(
        torch.from_numpy(baseline @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Baseline:", baseline_acc)
    
    # 6. perform ablation
    print(f"\n{'='*80}")
    print("Performing ablation...")
    
    # create copies
    attns_ablated = attns.copy()
    mlps_ablated = mlps.copy()
    
    # 6.1 ablate specified heads
    ablated_count = 0
    for layer, head in heads_to_ablate:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            attns_ablated[:, layer, head, :] = np.mean(
                attns_ablated[:, layer, head, :], axis=0, keepdims=True
            )
            ablated_count += 1
        else:
            print(f"⚠️  Invalid head: Layer {layer}, Head {head} (valid range: 0-{num_layers-1}, 0-{num_heads-1})")
    
    print(f"✓ Ablated {ablated_count} specified heads")
    
    # 6.2 ablate early layers (optional)
    if args.ablate_early_layers:
        ablate_until_layer = num_layers - args.keep_last_layers
        early_ablated = 0
        for layer in range(ablate_until_layer):
            for head in range(num_heads):
                if (layer, head) not in heads_to_ablate:  # avoid duplicates
                    attns_ablated[:, layer, head, :] = np.mean(
                        attns_ablated[:, layer, head, :], axis=0, keepdims=True
                    )
                    early_ablated += 1
        print(f"✓ Ablated {early_ablated} early layer heads (keeping last {args.keep_last_layers} layers)")
    
    # 6.3 ablate all MLPs (optional)
    if args.ablate_all_mlps:
        for layer in range(mlps_ablated.shape[1]):
            mlps_ablated[:, layer] = np.mean(
                mlps_ablated[:, layer], axis=0, keepdims=True
            )
        print(f"✓ Ablated all {mlps_ablated.shape[1]} MLP layers")
    
    # 7. compute ablated accuracy
    print(f"\n{'='*80}")
    print("Computing ablated accuracy...")
    ablated = attns_ablated.sum(axis=(1, 2)) + mlps_ablated.sum(axis=1)
    ablated_acc = full_accuracy(
        torch.from_numpy(ablated @ classifier).float(),
        torch.from_numpy(labels[:, 0]),
        labels[:, 1],
    )
    print("Ablated:", ablated_acc)
    
    # 8. compute difference
    print(f"\n{'='*80}")
    print("Accuracy Change:")
    print(f"{'='*80}")
    for key in baseline_acc.keys():
        diff = ablated_acc[key] - baseline_acc[key]
        print(f"  {key:10s}: {baseline_acc[key]:6.2f}% -> {ablated_acc[key]:6.2f}% (Δ {diff:+6.2f}%)")
    
    # 9. save results (optional)
    if args.figures_dir:
        result_file = os.path.join(
            args.figures_dir,
            f"{args.dataset}_ablation_strategy_{args.strategy or 'all'}_{args.model}.txt"
        )
        with open(result_file, "w") as f:
            f.write(f"Ablation Results for {args.dataset} - {args.model}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Strategy: {args.strategy or 'All'}\n")
            f.write(f"Heads ablated: {len(heads_to_ablate)}\n")
            f.write(f"Early layers ablated: {args.ablate_early_layers}\n")
            f.write(f"MLPs ablated: {args.ablate_all_mlps}\n")
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
        
        print(f"\n✓ Results saved to: {result_file}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)