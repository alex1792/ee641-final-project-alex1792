import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import argparse
from pathlib import Path
import os
import gc

# set PyTorch CUDA memory allocator to avoid memory fragmentation (directly set in Google Colab)
# Note: this setting needs to be imported torch after, but before using CUDA
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if 'PYTORCH_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'


def force_release_cuda_memory(device, verbose=False):
    """Force release CUDA memory (suitable for Google Colab)
    
    This method tries multiple strategies to release GPU memory:
    1. Python garbage collection
    2. PyTorch cache cleanup
    3. CUDA synchronization
    4. Reset memory statistics
    
    Note: PyTorch's memory allocator will keep a memory pool, even if empty_cache() is called,
    the memory will not be released back to the operating system immediately. This is PyTorch's design, 
    to improve performance.
    
    Args:
        device: CUDA device
        verbose: whether to print detailed memory information
    """
    if not torch.cuda.is_available():
        return
    
    # record memory state before cleanup
    if verbose:
        allocated_before = torch.cuda.memory_allocated(device) / 1024**3
        reserved_before = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  Memory before cleanup: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
    
    # multiple garbage collection
    for _ in range(3):
        gc.collect()
    
    # clean up PyTorch CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    
    # try to release memory fragments (PyTorch 2.0+)
    try:
        torch.cuda.ipc_collect()
    except AttributeError:
        pass  # old version PyTorch may not have this method
    
    # clean up and synchronize again
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    
    # reset memory statistics (will not release memory, but helps monitor)
    torch.cuda.reset_peak_memory_stats(device)
    
    # record memory state after cleanup
    if verbose:
        allocated_after = torch.cuda.memory_allocated(device) / 1024**3
        reserved_after = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  Memory after cleanup: Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB")
        if allocated_after == allocated_before:
            print(f"  WARNING: Memory did not decrease! This is normal for PyTorch's memory allocator.")
            print(f"  PyTorch keeps a memory pool for performance. Actual free memory may be higher.")

from utils.factory import create_model_and_transforms, get_tokenizer
from utils.binary_waterbirds import BinaryWaterbirds
from utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from utils.cub_classes import waterbird_classes
from prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10
from compute_text_projection import zero_shot_classifier


def get_args_parser():
    parser = argparse.ArgumentParser("Train CLIP with Decorrelation Loss", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training (reduced for memory)")
    parser.add_argument("--eval_batch_size", default=None, type=int, help="Batch size for evaluation (default: same as batch_size, can be larger for faster eval)")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--lambda_decorr", default=0.1, type=float, help="Weight for decorrelation loss")
    parser.add_argument("--last_n_layers", default=2, type=int, help="Number of last layers to apply decorrelation loss")
    parser.add_argument("--train_last_n_layers", default=4, type=int, help="Only train last N layers (0 = train all, saves memory)")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training to save memory")
    
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-L-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k", type=str)
    
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/shared/group/ilsvrc", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="binary_waterbirds", help="binary_waterbirds, CIFAR10, CIFAR100"
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for training")
    parser.add_argument("--save_ckpt", action="store_true", help="Save model checkpoints")
    
    return parser


def cleanup_all_prs_hooks(model):
    """Clean up all PRSLogger related hooks (not dependent on specific prs objects)
    
    This function will clean up all hooks bound to PRSLogger instances, 
    regardless of which prs object.
    This is a more thorough cleanup method, used to prevent hooks accumulation.
    
    Args:
        model: CLIP model
    """
    def cleanup_hook_manager(hook_manager):
        """Recursive cleanup hook_manager and all forks"""
        # clean up main hook_dict
        for key in list(hook_manager.hook_dict.keys()):
            original_list = hook_manager.hook_dict[key]
            filtered_list = [
                func for func in original_list
                if not (hasattr(func, '__self__') and 
                       hasattr(func.__self__, '__class__') and
                       func.__self__.__class__.__name__ == 'PRSLogger')
            ]
            if len(filtered_list) < len(original_list):
                hook_manager.hook_dict[key] = filtered_list
                if len(filtered_list) == 0:
                    del hook_manager.hook_dict[key]
        
        # recursive cleanup all forks
        for fork_name, fork_manager in hook_manager.forks.items():
            cleanup_hook_manager(fork_manager)
    
    # clean up top level hook_manager (hooks are registered here)
    cleanup_hook_manager(model.hook_manager)
    
    # also clean up visual.hook (where hooks are actually executed)
    cleanup_hook_manager(model.visual.hook)


def cleanup_prs_hooks(model, prs):
    """Clean up specific PRS logger hooks
    
    Args:
        model: CLIP model
        prs: PRSLogger object (may be None)
    """
    if prs is None:
        return
    
    # clean up all PRSLogger hooks (more thorough method)
    cleanup_all_prs_hooks(model)


def compute_decorrelation_loss(attention_outputs, last_n_layers=2):
    """
    Compute decorrelation loss, making the last N layer's attention head outputs as dissimilar as possible
    
    Args:
        attention_outputs: [batch, layers, num_patches, num_heads, dim] (spatial mode)
        last_n_layers: compute decorrelation loss for last N layers
    
    Returns:
        decorr_loss: scalar tensor
    """
    # get CLS token attention outputs for last n layers
    # attention_outputs: [batch, layers, num_patches, num_heads, dim]
    num_layers = attention_outputs.shape[1]
    last_layers = attention_outputs[:, -last_n_layers:, ...]  # [batch, n_layers, num_patches, num_heads, dim]
    
    # get output of CLS token (index 0)
    cls_token_outputs = last_layers[:, :, 0, :, :]  # [batch, n_layers, num_heads, dim]
    
    # flatten: merge batch and layers
    # [batch * n_layers, num_heads, dim]
    flat_outputs = cls_token_outputs.reshape(-1, cls_token_outputs.shape[2], cls_token_outputs.shape[3])
    
    # compute similarity between heads (cosine similarity)
    normalized = F.normalize(flat_outputs, dim=-1)  # [batch*n_layers, num_heads, dim]
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))  # [batch*n_layers, num_heads, num_heads]
    
    # remove diagonal (similarity with itself = 1)
    batch_size, num_heads, _ = similarity_matrix.shape
    mask = torch.eye(num_heads, device=similarity_matrix.device).bool().unsqueeze(0)  # [1, num_heads, num_heads]
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)
    
    # penalize high similarity: compute sum of squared similarity between all head pairs
    # or use absolute mean
    decorr_loss = similarity_matrix.abs().mean()
    
    return decorr_loss


def train_epoch(model, dataloader, optimizer, classifier, device, lambda_decorr=0.1, last_n_layers=2, use_amp=False):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_classification_loss = 0
    total_decorr_loss = 0
    correct = 0
    total = 0
    
    # mixed precision training scaler (using new API)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # only register hook when decorrelation loss is needed (to save memory)
    prs = None
    if lambda_decorr > 0:
        # before registering new hooks, clean up all old PRSLogger hooks (safety measure)
        cleanup_all_prs_hooks(model)
        
        # now safely register new hooks
        prs = hook_prs_logger(model, device, spatial=True, training_mode=True, keep_last_n_layers=last_n_layers)
    
    for batch_idx, (images, labels) in enumerate(tqdm.tqdm(dataloader, desc="Training")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # reset prs state before forward pass (ensure hooks are reset before being called)
        if prs is not None:
            prs.reinit()
        
        # Forward pass with mixed precision (using new API)
        with torch.amp.autocast('cuda', enabled=use_amp):
            if prs is not None:
                representation = model.encode_image(images, attn_method="head", normalize=False)
                attentions, mlps = prs.finalize(representation)
            else:
                # if decorrelation is not needed, use direct method to save memory
                representation = model.encode_image(images, attn_method="direct", normalize=False)
                attentions = None
                mlps = None
            
            # compute classification loss
            logits = representation @ classifier
            classification_loss = F.cross_entropy(logits, labels)
            
            # compute decorrelation loss (if enabled)
            if lambda_decorr > 0 and attentions is not None:
                decorr_loss = compute_decorrelation_loss(attentions, last_n_layers=last_n_layers)
            else:
                decorr_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            loss = classification_loss + lambda_decorr * decorr_loss
        
        # Backward with mixed precision
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # statistics
        with torch.no_grad():
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_decorr_loss += decorr_loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        # clean up memory
        del logits, loss, decorr_loss, classification_loss, representation, pred
        if attentions is not None:
            del attentions
        if mlps is not None:
            del mlps
        
        # clean up data inside prs object (if exists)
        if prs is not None:
            if hasattr(prs, 'attentions'):
                if isinstance(prs.attentions, torch.Tensor):
                    del prs.attentions
                    prs.attentions = []
                elif isinstance(prs.attentions, list):
                    prs.attentions.clear()
            
            if hasattr(prs, 'mlps'):
                if isinstance(prs.mlps, torch.Tensor):
                    del prs.mlps
                    prs.mlps = []
                elif isinstance(prs.mlps, list):
                    prs.mlps.clear()
        
        # reduce memory cleanup frequency: clean up every 50 batches, give allocator a chance to reuse memory
        if batch_idx > 0 and batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # clean up scaler after training
    if scaler is not None:
        del scaler
    
    # clean up all data inside prs object, release GPU memory (if exists)
    if prs is not None:
        if hasattr(prs, 'attentions'):
            if isinstance(prs.attentions, torch.Tensor):
                del prs.attentions
            elif isinstance(prs.attentions, list):
                prs.attentions.clear()
            prs.attentions = []
        
        if hasattr(prs, 'mlps'):
            if isinstance(prs.mlps, torch.Tensor):
                del prs.mlps
            elif isinstance(prs.mlps, list):
                prs.mlps.clear()
            prs.mlps = []
        
        if hasattr(prs, 'post_ln_mean') and prs.post_ln_mean is not None:
            del prs.post_ln_mean
            prs.post_ln_mean = None
        
        if hasattr(prs, 'post_ln_std') and prs.post_ln_std is not None:
            del prs.post_ln_std
            prs.post_ln_std = None
        
        # reset prs state
        prs.current_layer = 0
        
        # clean up all PRSLogger hooks, prevent accumulation (using more thorough cleanup method)
        cleanup_all_prs_hooks(model)
    
    # clean up memory after training
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {
        'total_loss': total_loss / len(dataloader),
        'classification_loss': total_classification_loss / len(dataloader),
        'decorr_loss': total_decorr_loss / len(dataloader),
        'accuracy': correct / total * 100,
        'prs': prs  # return prs object (may be None), to remove its hooks when evaluating
    }


def eval_epoch(model, dataloader, classifier, device, train_prs=None):
    """Evaluate model - minimal version, only do minimum necessary operations
    
    Key strategies:
    1. clean up train_prs and hooks
    2. classifier always on CPU
    3. representation immediately move to CPU
    4. all calculations on CPU
    5. only clean up memory once before and after eval, not frequently in loop
    
    Args:
        model: model
        dataloader: validation data loader
        classifier: classifier weights (will be moved to CPU inside function)
        device: device
        train_prs: prs object during training (if provided, will remove its hooks first)
    """
    # 1. clean up prs object and hooks during training
    if train_prs is not None:
        cleanup_prs_hooks(model, train_prs)
        # delete train_prs object
        del train_prs
    
    # clean up memory before evaluation (multiple cleanup ensures memory is released)
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # ensure all operations are completed
    
    # print memory usage before evaluation (for debugging)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"Memory before evaluation: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # 2. set model to evaluation mode
    model.eval()
    
    # Note: no need to set requires_grad = False, because torch.inference_mode() has already disabled gradient computation
    # if requires_grad = False, it will cause gradient backpropagation to fail during next epoch training
    
    # 3. classifier stays on GPU (using GPU computing power, only move final result to CPU)
    # this can avoid performance bottleneck of CPU matrix multiplication
    classifier_gpu = classifier.to(device)
    
    correct = 0
    total = 0
    
    # 4. pure inference mode, no graph, no activation
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(tqdm.tqdm(dataloader, desc="Evaluating")):
            # use non_blocking asynchronous transfer, improve efficiency
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            reps = model.encode_image(images, attn_method="direct", normalize=False)
            
            # compute logits on GPU (using GPU computing power)
            logits = reps @ classifier_gpu
            preds = logits.argmax(dim=1)
            
            # only move final result to CPU (using item() to avoid creating new tensor)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # immediately clean up all intermediate results (including images and labels, to avoid memory accumulation)
            del reps, logits, preds, images, labels
            
            # clean up GPU cache after each batch, ensure memory is released immediately
            torch.cuda.empty_cache()
            
            # garbage collection after processing a certain number of batches, to prevent memory accumulation
            # use smaller interval (10) to clean up memory more frequently, avoid OOM
            if (batch_idx + 1) % 10 == 0:
                gc.collect()
                # synchronize every 50 batches, to avoid performance bottleneck of too frequent synchronization
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.synchronize()
    
    # clean up memory after evaluation (multiple cleanup ensures memory is released)
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # ensure all operations are completed
    
    # print memory usage after evaluation (for debugging)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"Memory after evaluation: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    return correct / total * 100


def main(args):
    # create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # load model
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.to(args.device)
    
    # set which layers to train
    # option 1: only train last few layers (recommended, to save memory)
    if args.train_last_n_layers > 0:
        num_layers = len(model.visual.transformer.resblocks)
        print(f"Freezing first {num_layers - args.train_last_n_layers} layers, training last {args.train_last_n_layers} layers")
        
        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # only train last N transformer blocks
        for i in range(num_layers - args.train_last_n_layers, num_layers):
            for param in model.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
        
        # train last projection layer
        for param in model.visual.ln_post.parameters():
            param.requires_grad = True
        
        # proj may be Parameter instead of Module, need to set directly
        if hasattr(model.visual, 'proj'):
            proj = model.visual.proj
            if isinstance(proj, torch.nn.Parameter):
                proj.requires_grad = True
            else:
                # if Module (e.g. Linear), iterate over its parameters
                for param in proj.parameters():
                    param.requires_grad = True
    else:
        # option 2: train all parameters
        for param in model.parameters():
            param.requires_grad = True
    
    # create optimizer (only optimize parameters with gradient)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # load classifier weights
    tokenizer = get_tokenizer(args.model)
    if args.dataset == "binary_waterbirds":
        classes = waterbird_classes
    elif args.dataset == "CIFAR10":
        from torchvision.datasets import CIFAR10
        classes = CIFAR10.classes
    elif args.dataset == "CIFAR100":
        from torchvision.datasets import CIFAR100
        classes = CIFAR100.classes
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    classifier_path = os.path.join(args.output_dir, f"{args.dataset}_classifier_{args.model}.npy")
    if os.path.exists(classifier_path):
        print(f"Loading classifier from {classifier_path}")
        classifier = torch.from_numpy(np.load(classifier_path)).to(args.device)
    else:
        print("Computing classifier weights...")
        classifier = zero_shot_classifier(
            model, tokenizer, classes, OPENAI_IMAGENET_TEMPLATES, args.device
        )
        np.save(classifier_path, classifier.detach().cpu().numpy())
        classifier = classifier.to(args.device)
    
    # load dataset
    if args.dataset == "binary_waterbirds":
        train_ds = BinaryWaterbirds(root=args.data_path, split="train", transform=preprocess)
        val_ds = BinaryWaterbirds(root=args.data_path, split="test", transform=preprocess)
    elif args.dataset == "CIFAR10":
        train_ds = CIFAR10(
            root=args.data_path, download=True, train=True, transform=preprocess
        )
        val_ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR100":
        train_ds = CIFAR100(
            root=args.data_path, download=True, train=True, transform=preprocess
        )
        val_ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    # when evaluating, use larger batch size to fully utilize GPU
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    # when evaluating, use more workers to accelerate data loading (no backward propagation during evaluation, less memory usage)
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"Training batch size: {args.batch_size}, Evaluation batch size: {eval_batch_size}")
    
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Lambda decorr: {args.lambda_decorr}")
    print(f"Last N layers: {args.last_n_layers}")
    print(f"Using mixed precision: {args.use_amp}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # check if training should be skipped (use pretrained model directly)
    # if lambda_decorr=0 and train_last_n_layers=0 and epochs=0, skip training
    skip_training = (args.lambda_decorr == 0 and args.train_last_n_layers == 0 and args.epochs == 0)
    
    if skip_training:
        print("\n" + "=" * 80)
        print("SKIPPING TRAINING - Using pretrained model directly")
        print("=" * 80)
        print("Reason: lambda_decorr=0, train_last_n_layers=0, and epochs=0")
        print("This means no decorrelation loss, training all layers, and 0 epochs.")
        print("The model will be evaluated with pretrained weights only.\n")
        
        # only evaluate (eval_epoch is defined at the top of the file)
        val_acc = eval_epoch(model, val_loader, classifier, args.device, train_prs=None)
        print(f"\nPretrained Model Val Accuracy: {val_acc:.2f}%")
        return
    
    # training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # before each epoch, clean up all possible old hooks (to prevent accumulation)
        # this is a safety measure, to ensure no residual hooks remain
        if args.lambda_decorr > 0:
            cleanup_all_prs_hooks(model)
        
        # train
        train_metrics = train_epoch(
            model, train_loader, optimizer, classifier, args.device,
            lambda_decorr=args.lambda_decorr,
            last_n_layers=args.last_n_layers,
            use_amp=args.use_amp
        )
        
        # extract metrics to print, then delete train_metrics
        train_prs = train_metrics.get('prs')
        train_loss = train_metrics['total_loss']
        cls_loss = train_metrics['classification_loss']
        decorr_l = train_metrics['decorr_loss']
        train_acc = train_metrics['accuracy']
        
        # delete train_metrics dictionary
        del train_metrics
        
        # force Python garbage collection (after training) - multiple cleanup ensures memory is released
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # try to release all possible GPU memory
        try:
            torch.cuda.ipc_collect()
        except:
            pass
        
        # print memory usage after training (for debugging)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(args.device) / 1024**3
            reserved = torch.cuda.memory_reserved(args.device) / 1024**3
            print(f"Memory after training cleanup: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        # evaluate (pass prs object during training, to remove its hooks)
        # Note: when using attn_method="direct", no hooks will be triggered
        val_acc = eval_epoch(model, val_loader, classifier, args.device, train_prs=train_prs)
        
        # evaluate again, clean up memory and hooks
        if train_prs is not None:
            cleanup_all_prs_hooks(model)  # ensure all hooks are cleaned up
            del train_prs
        # multiple cleanup ensures memory is released
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # try to release all possible GPU memory
        try:
            torch.cuda.ipc_collect()
        except:
            pass
        
        # reset optimizer internal state (if any), release some memory
        # Note: this will not affect optimizer parameters, only clean up internal cache
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True can release more memory
        
        # print training and validation results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  - Classification Loss: {cls_loss:.4f}")
        print(f"  - Decorrelation Loss: {decorr_l:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Val Accuracy: {val_acc:.2f}%")
        
        # save best model
        if args.save_ckpt and val_acc > best_val_acc:
            best_val_acc = val_acc
            # add decorr_loss related information to filename
            decorr_info = f"_decorr{args.lambda_decorr}_layers{args.last_n_layers}" if args.lambda_decorr > 0 else "_no_decorr"
            ckpt_path = os.path.join(
                args.output_dir, 
                f"best_model_{args.model}_{args.dataset}{decorr_info}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args,
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)