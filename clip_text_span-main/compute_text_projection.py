import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os.path
import argparse
import datetime
import json
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from utils.imagenet_classes import imagenet_classes
from utils.cub_classes import cub_classes, waterbird_classes


def get_args_parser():
    parser = argparse.ArgumentParser('Get classifier weights', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument('--dataset', default='imagenet', help='waterbirds or imagenet')
    parser.add_argument('--pretrained', default='laion2b_s32b_b79k', type=str)
    # Checkpoint loading option
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to checkpoint .pth file (if provided, will load from checkpoint instead of pretrained)')
    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    return parser



def zero_shot_classifier(model, tokenizer, classnames, templates, 
                         device, amp=True, use_format=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            texts = [template.format(c=classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def main(args):
    """Calculates the classifier projection weights."""
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = get_tokenizer(args.model)
    model.to(args.device)
    model.eval()
    
    # Load from checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        # PyTorch 2.6+ requires weights_only=False when loading checkpoints with non-weight objects
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        
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
        print(f"  Model loaded successfully from checkpoint!")
    else:
        print(f"Using pretrained model: {args.pretrained}")
    
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    classes = {
        'imagenet': imagenet_classes, 
        'waterbirds': cub_classes, 
        'binary_waterbirds': waterbird_classes, 
        'cub': cub_classes}[args.dataset]
    classifier = zero_shot_classifier(model, tokenizer, classes, OPENAI_IMAGENET_TEMPLATES, args.device)
    
    # Use checkpoint name in output filename if loading from checkpoint
    if args.checkpoint:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_filename = f'{args.dataset}_classifier_{checkpoint_name}.npy'
    else:
        output_filename = f'{args.dataset}_classifier_{args.model}.npy'
    
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, 'wb') as f:
        np.save(f, classifier.detach().cpu().numpy())
    print(f"\nSaved classifier to: {output_path}")
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)