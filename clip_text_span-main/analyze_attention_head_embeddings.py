import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import Counter
import tqdm

# 1. load attention features
attn_path = "output_dir/binary_waterbirds_attn_ViT-L-14.npy"
with open(attn_path, "rb") as f:
    attns = np.load(f)  # [N, 24, 16, 1024]

# get model structure
num_samples, num_layers, num_heads, embedding_dim = attns.shape
print(f"Model structure: {num_layers} layers, {num_heads} heads per layer")

# 2. extract features from specific layer and head
layers = [22, 23]   # last 2 layer
print(f"Analyzing last 2 layers: {layers}")

# create dictionary to store head features and text spans
head_features_dict = {}
head_text_spans = {}

# 3. load text features (embeddings)
text_features_path = "output_dir/image_descriptions_general_ViT-L-14.npy"
with open(text_features_path, "rb") as f:
    text_features = np.load(f)  # [num_texts, 1024]

# 4. load text lines (original text strings)
text_file_path = "text_descriptions/image_descriptions_general.txt"
with open(text_file_path, "r") as f:
    text_lines = [line.replace("\n", "") for line in f.readlines()]

# 5. compute text span for each head
from compute_complete_text_set import replace_with_iterative_removal

print(f"\nComputing top 1 text span for all heads in last 2 layers...")
for layer in layers:
    for head in range(num_heads):
        # extract features of this head
        head_features = attns[:, layer, head]  # [N, 1024]
        head_features_dict[(layer, head)] = head_features
        
        # compute text span (only top 1)
        reconstruct, text_span = replace_with_iterative_removal(
            head_features,
            text_features,
            text_lines,
            1,  # iters: only top 1
            80,  # rank: SVD rank
            "cuda:0"
        )
        
        top1_text = text_span[0] if len(text_span) > 0 else "N/A"
        head_text_spans[(layer, head)] = top1_text

print(f"✓ Computation completed, {len(head_text_spans)} heads")

# 6. print text spans
print(f"\n{'='*80}")
print("Top 1 Text Span for each Head")
print(f"{'='*80}")
for layer in layers:
    print(f"\nLayer {layer}:")
    print("-"*70)
    for head in range(num_heads):
        print(f"Head {head:2d}: {head_text_spans[(layer, head)]}")

# 7. cluster heads features projected to text spans 
def cluster_by_projected_features(head_features_dict, head_text_spans,
                                  text_features, text_lines, n_clusters=5, rank=80):
    """project head features to text space and cluster"""
    head_list = list(head_features_dict.keys())
    projected_features = []
    
    print(f"\n{'='*80}")
    print(f"Using projected method to cluster")
    print(f"{'='*80}")
    print(f"Projecting and clustering {len(head_list)} heads...")
    
    for (layer, head) in tqdm.tqdm(head_list, desc="Projecting head features"):
        head_feat = head_features_dict[(layer, head)]  # [N, 1024]
        
        # use SVD projection (similar to replace_with_iterative_removal)
        u, s, vh = np.linalg.svd(head_feat, full_matrices=False)
        vh = vh[:rank]
        
        # project text features to head's span
        proj_text = (
            vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
        )
        
        # compute the representation of head features in the projected text space
        # use the similarity between head features and projected text as features
        head_avg = head_feat.mean(axis=0)
        similarities = head_avg @ proj_text.T
        projected_features.append(similarities)
    
    # K-Means clustering
    proj_matrix = np.array(projected_features)
    print(f"Projected feature matrix shape: {proj_matrix.shape}")
    
    print(f"\nExecuting K-Means clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(proj_matrix)
    
    return head_list, cluster_labels, proj_matrix

# 8. execute clustering
n_clusters = 5  # can adjust this number
head_list, cluster_labels, proj_matrix = cluster_by_projected_features(
    head_features_dict,
    head_text_spans,
    text_features,
    text_lines,
    n_clusters=n_clusters,
    rank=80
)

# 9. analyze clustering results
print(f"\n{'='*80}")
print("CLUSTERING RESULTS ANALYSIS")
print(f"{'='*80}\n")

cluster_results = {}
head_to_cluster = {}  # create mapping: from (layer, head) to cluster_id

for i, (layer, head) in enumerate(head_list):
    cluster_id = cluster_labels[i]
    head_to_cluster[(layer, head)] = cluster_id  # add mapping
    
    if cluster_id not in cluster_results:
        cluster_results[cluster_id] = {
            'heads': [],
            'texts': []
        }
    cluster_results[cluster_id]['heads'].append((layer, head))
    cluster_results[cluster_id]['texts'].append(head_text_spans[(layer, head)])

# print information of each cluster
for cluster_id in sorted(cluster_results.keys()):
    info = cluster_results[cluster_id]
    print(f"Cluster {cluster_id} ({len(info['heads'])} heads):")
    
    # find the most common text as representative
    text_counts = Counter(info['texts'])
    top_texts = [text for text, count in text_counts.most_common(5)]
    print(f"  Representative texts:")
    for text, count in text_counts.most_common(5):
        print(f"    - {text} (appeared {count} times)")
    
    print(f"  Heads: {info['heads'][:8]}")  # show first 8 heads
    if len(info['heads']) > 8:
        print(f"    ...etc {len(info['heads']) - 8} heads")
    print()

# 10. select heads to ablate based on clustering results
print(f"{'='*80}")
print("SELECTION STRATEGY SUGGESTIONS")
print(f"{'='*80}\n")

# strategy 1: select the largest cluster (possibly containing spurious correlation)
largest_cluster = max(cluster_results.keys(), 
                     key=lambda k: len(cluster_results[k]['heads']))
print(f"Strategy 1: select the largest cluster (Cluster {largest_cluster})")
print(f"  Including {len(cluster_results[largest_cluster]['heads'])} heads")
print(f"  These heads may learn similar features (possibly spurious correlation)")
print(f"  Suggest to ablate these heads: {cluster_results[largest_cluster]['heads']}\n")

# strategy 2: select cluster containing specific keywords (e.g. "background", "setting")
print("Strategy 2: select cluster containing specific keywords (e.g. 'background', 'setting')")
background_keywords = ['background', 'setting', 'scene', 'environment', 
                      'photo of', 'picture of', 'taken in', 'aerial view']
background_heads = []  # collect all background-related heads
background_clusters = []  # record which clusters contain background

for cluster_id, info in cluster_results.items():
    texts = info['texts']
    has_background = any(any(kw in text.lower() for kw in background_keywords) 
                        for text in texts)
    
    if has_background:
        background_heads.extend(info['heads'])  # collect heads
        background_clusters.append(cluster_id)  # record cluster
        print(f"  Cluster {cluster_id} containing background-related texts ({len(info['heads'])} heads)")
        print(f"  Suggest to ablate these heads: {info['heads']}\n")

if not background_heads:
    print("  No clusters containing background-related texts\n")

# strategy 3: select representative heads from each cluster (1-2 heads per cluster)
print("Strategy 3: select representative heads from each cluster (1-2 heads per cluster)")
selected_heads = []
for cluster_id, info in cluster_results.items():
    selected_heads.extend(info['heads'][:2])
print(f"  Selected heads: {selected_heads}\n")

# strategy 4: select heads individually (interactive selection)
print("Strategy 4: select heads individually (interactive selection)")
print("=" * 80)
print("You will be shown each head's text span and asked whether to ablate it.")
print("Commands:")
print("  'y' or 'yes' - Ablate this head")
print("  'n' or 'no'  - Keep this head")
print("  'q' or 'quit' - Quit and save current selection")
print("  'a' or 'abort' - Abort and clear all selections")
print("  's' or 'skip' - Skip remaining heads in current cluster")
print("=" * 80)

individually_selected_heads = []
skipped_heads = []
total_heads = len(head_list)
current_cluster = None
skip_current_cluster = False

print(f"\nReviewing {total_heads} heads...\n")

for idx, (layer, head) in enumerate(head_list, 1):
    text_span = head_text_spans[(layer, head)]
    cluster_id = head_to_cluster[(layer, head)]
    
    # check if we should skip the current cluster
    if skip_current_cluster and cluster_id == current_cluster:
        skipped_heads.append((layer, head))
        continue
    else:
        skip_current_cluster = False
        current_cluster = cluster_id
    
    print(f"\n{'='*80}")
    print(f"[{idx}/{total_heads}] Layer {layer}, Head {head}")
    print(f"{'='*80}")
    print(f"Text span: {text_span}")
    print(f"Cluster: {cluster_id} ({len(cluster_results[cluster_id]['heads'])} heads in this cluster)")
    
    # show other heads in the same cluster
    cluster_mates = [h for h in cluster_results[cluster_id]['heads'] if h != (layer, head)]
    if cluster_mates:
        print(f"Other heads in same cluster:")
        for mate_layer, mate_head in cluster_mates[:5]:
            mate_text = head_text_spans[(mate_layer, mate_head)]
            print(f"  - Layer {mate_layer}, Head {mate_head}: {mate_text}")
        if len(cluster_mates) > 5:
            print(f"  ... and {len(cluster_mates) - 5} more")
    
    # show current selection statistics
    print(f"\nCurrent selection: {len(individually_selected_heads)} heads selected, {len(skipped_heads)} skipped")
    
    # wait for user input
    while True:
        try:
            user_input = input(f"\n  Ablate this head? [y/n/q/a/s]: ").strip().lower()
            
            if user_input in ['y', 'yes']:
                individually_selected_heads.append((layer, head))
                print(f"  ✓ Added to ablation list (total: {len(individually_selected_heads)})")
                break
            elif user_input in ['n', 'no']:
                skipped_heads.append((layer, head))
                print(f"  - Skipped")
                break
            elif user_input in ['q', 'quit']:
                print(f"\n  Quitting selection.")
                print(f"  Selected {len(individually_selected_heads)} heads so far.")
                confirm = input(f"  Save and continue to next strategy? [y/n]: ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
                else:
                    # continue selection
                    continue
            elif user_input in ['a', 'abort']:
                print(f"\n  Aborting all selections.")
                confirm = input(f"  Clear all {len(individually_selected_heads)} selected heads? [y/n]: ").strip().lower()
                if confirm in ['y', 'yes']:
                    individually_selected_heads = []
                    print(f"  ✓ All selections cleared")
                    break
                else:
                    continue
            elif user_input in ['s', 'skip']:
                skip_current_cluster = True
                current_cluster = cluster_id
                skipped_heads.append((layer, head))
                remaining_in_cluster = len([h for h in cluster_results[cluster_id]['heads'] 
                                           if h != (layer, head)])
                print(f"  - Skipped. Will skip remaining {remaining_in_cluster} heads in cluster {cluster_id}")
                break
            else:
                print(f"  Invalid input. Please enter 'y', 'n', 'q', 'a', or 's'")
        except KeyboardInterrupt:
            print(f"\n\n  ⚠️  Interrupted by user (Ctrl+C)")
            print(f"  Selected {len(individually_selected_heads)} heads so far.")
            save_now = input(f"  Save current selection and exit? [y/n]: ").strip().lower()
            if save_now in ['y', 'yes']:
                break
            else:
                print(f"  Continuing...")
                continue
    
    # if user chooses to quit, break the loop
    if user_input in ['q', 'quit'] and confirm not in ['y', 'yes']:
        break

print(f"\n{'='*80}")
print(f"Strategy 4 Selection Summary")
print(f"{'='*80}")
print(f"  Selected for ablation: {len(individually_selected_heads)} heads")
print(f"  Skipped: {len(skipped_heads)} heads")
if individually_selected_heads:
    print(f"\n  Selected heads:")
    for layer, head in individually_selected_heads:
        print(f"    Layer {layer}, Head {head}: {head_text_spans[(layer, head)]}")
print()

# 11. save results
print(f"{'='*80}")
print("SAVE RESULTS")
print(f"{'='*80}\n")

# save clustering results
output_file = "output_dir/head_clustering_projected_results.txt"
with open(output_file, "w") as f:
    f.write("Projected clustering results for last 2 layers Attention Heads (method 4)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Clustering method: Projected head features to text space and cluster\n")
    f.write(f"Number of clusters: {n_clusters}\n")
    f.write(f"Total heads: {len(head_list)}\n\n")
    
    for cluster_id in sorted(cluster_results.keys()):
        info = cluster_results[cluster_id]
        f.write(f"Cluster {cluster_id} ({len(info['heads'])} heads):\n")
        f.write("-" * 80 + "\n")
        
        text_counts = Counter(info['texts'])
        f.write(f"Representative texts:\n")
        for text, count in text_counts.most_common(5):
            f.write(f"  - {text} (appeared {count} times)\n")
        f.write(f"\nHeads:\n")
        for layer, head in info['heads']:
            f.write(f"  Layer {layer}, Head {head}: {head_text_spans[(layer, head)]}\n")
        f.write("\n")

print(f"✓ Clustering results saved to: {output_file}")

# save ablation heads list - directly using results from strategy suggestion
ablation_file = "output_dir/heads_to_ablate_projected.txt"
with open(ablation_file, "w") as f:
    f.write("# Heads selected for ablation based on projected clustering (method 4)\n")
    f.write("# Format: Layer,Head\n\n")
    
    # strategy 1: using results from largest_cluster
    f.write(f"# Strategy 1: Largest cluster (Cluster {largest_cluster})\n")
    f.write(f"# Total heads: {len(cluster_results[largest_cluster]['heads'])}\n")
    for layer, head in cluster_results[largest_cluster]['heads']:
        f.write(f"{layer},{head}\n")
    f.write("\n")
    
    # strategy 2: directly using background_heads (collected in strategy suggestion)
    f.write("# Strategy 2: Background-related heads\n")
    if background_heads:
        f.write(f"# Found in clusters: {background_clusters}\n")
        f.write(f"# Total heads: {len(background_heads)}\n")
        for layer, head in background_heads:
            f.write(f"{layer},{head}\n")
    else:
        f.write("# No background-related heads found\n")
    f.write("\n")
    
    # strategy 3: directly using selected_heads (calculated in strategy suggestion)
    f.write("# Strategy 3: Representative heads from each cluster (2 per cluster)\n")
    f.write(f"# Total heads: {len(selected_heads)}\n")
    for layer, head in selected_heads:
        f.write(f"{layer},{head}\n")

    # strategy 4: individually selected heads
    f.write("# Strategy 4: Individually selected heads (interactive selection)\n")
    f.write(f"# Total heads: {len(individually_selected_heads)}\n")
    if individually_selected_heads:
        for layer, head in individually_selected_heads:
            f.write(f"{layer},{head}\n")
    else:
        f.write("# No heads selected\n")

print(f"✓ Ablation heads list saved to: {ablation_file}")