import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path


class PRSLogger(object):
    def __init__(self, model, device, spatial: bool = True, training_mode: bool = False, keep_last_n_layers: int = None):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.spatial = spatial
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        self.training_mode = training_mode
        self.keep_last_n_layers = keep_last_n_layers  # In training mode, only save last N layers
        self.total_layers = len(model.visual.transformer.resblocks)  # Total number of layers


    # @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        assert len(ret.shape) == 5, "Verify that you use method=`head` and not method=`head_no_spatial`" # [b, n, m, h, d]
        assert self.spatial, "Verify that you use method=`head` and not method=`head_no_spatial`"
        
        # Safety check: if ret's shape is incorrect, this hook was incorrectly registered
        # In this case, we should directly return ret without any processing
        if len(ret.shape) != 5:
            print(f"Warning: compute_attentions_spatial called with wrong shape {ret.shape}, skipping")
            return ret
        
        # Safety check: ensure attentions attribute exists and is a list, initialize if not
        if not hasattr(self, 'attentions') or not isinstance(self.attentions, list):
            self.attentions = []
        
        # Boundary check: ensure current_layer does not exceed range
        if self.current_layer >= self.total_layers:
            # If out of range, reset and return (should not happen, but as a safety measure)
            # This usually happens when hooks are not properly cleaned up or reinit() is not called
            print(f"Warning: current_layer {self.current_layer} >= total_layers {self.total_layers}, resetting")
            self.current_layer = 0
            # Clear attentions list to avoid index errors
            if hasattr(self, 'attentions') and isinstance(self.attentions, list):
                self.attentions.clear()
            return ret
        
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        layer_idx = self.current_layer
        self.current_layer += 1
        
        if self.training_mode:
            # Training mode: only save last N layers, don't save other layers (save memory)
            if self.keep_last_n_layers is not None:
                # Only save last N layers
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    return_value = ret[:, 0]  # [b, n, h, d] - don't move to CPU, don't detach
                    attention_output = (
                        return_value
                        + bias_term[np.newaxis, np.newaxis, np.newaxis]
                        / (return_value.shape[1] * return_value.shape[2])
                    )
                    self.attentions.append(attention_output)
                else:
                    # Don't save earlier layers, save memory
                    self.attentions.append(None)  # Placeholder to maintain index consistency
            else:
                # Save all layers (original behavior)
                return_value = ret[:, 0]  # [b, n, h, d] - don't move to CPU, don't detach
                attention_output = (
                    return_value
                    + bias_term[np.newaxis, np.newaxis, np.newaxis]
                    / (return_value.shape[1] * return_value.shape[2])
                )
                self.attentions.append(attention_output)
        else:
            # Inference mode: also handle keep_last_n_layers to save memory
            if self.keep_last_n_layers is not None:
                # Only save last N layers
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
                    self.attentions.append(
                        return_value
                        + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
                        / (return_value.shape[1] * return_value.shape[2])
                    )  # [b, n, h, d]
                else:
                    # Don't save earlier layers, save memory
                    self.attentions.append(None)  # Placeholder
            else:
                # Save all layers (original behavior)
                return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
                self.attentions.append(
                    return_value
                    + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
                    / (return_value.shape[1] * return_value.shape[2])
                )  # [b, n, h, d]
        return ret

    # @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        # Safety check: if self.spatial is True, this function should not be called
        # Directly return ret without any processing
        if self.spatial:
            # This function should not be called because spatial attention is being used
            return ret
        
        # Safety check: if ret's shape is incorrect, also directly return
        if len(ret.shape) != 4:
            return ret
        
        assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial` and not method=`head`" # [b, n, h, d]
        assert not self.spatial, "Verify that you use method=`head_no_spatial` and not method=`head`"
        
        # Safety check: ensure attentions attribute exists and is a list, initialize if not
        if not hasattr(self, 'attentions') or not isinstance(self.attentions, list):
            self.attentions = []
        
        # Boundary check: ensure current_layer does not exceed range
        if self.current_layer >= self.total_layers:
            # If out of range, reset and return (should not happen, but as a safety measure)
            # This usually happens when hooks are not properly cleaned up or reinit() is not called
            print(f"Warning: current_layer {self.current_layer} >= total_layers {self.total_layers}, resetting")
            self.current_layer = 0
            # Clear attentions list to avoid index errors
            if hasattr(self, 'attentions') and isinstance(self.attentions, list):
                self.attentions.clear()
            return ret
        
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        layer_idx = self.current_layer
        self.current_layer += 1
        
        if self.training_mode:
            # Training mode: only save last N layers
            if self.keep_last_n_layers is not None:
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    return_value = ret[:, 0]  # [b, n, h, d]
                    attention_output = (
                        return_value
                        + bias_term[np.newaxis, np.newaxis]
                        / (return_value.shape[1])
                    )
                    self.attentions.append(attention_output)
                else:
                    self.attentions.append(None)
            else:
                return_value = ret[:, 0]  # [b, n, h, d]
                attention_output = (
                    return_value
                    + bias_term[np.newaxis, np.newaxis]
                    / (return_value.shape[1])
                )
                self.attentions.append(attention_output)
        else:
            # Inference mode: also handle keep_last_n_layers to save memory
            if self.keep_last_n_layers is not None:
                # Only save last N layers
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
                    self.attentions.append(
                        return_value
                        + bias_term[np.newaxis, np.newaxis].cpu()
                        / (return_value.shape[1])
                    )  # [b, h, d]
                else:
                    # Don't save earlier layers, save memory
                    self.attentions.append(None)  # Placeholder
            else:
                # Save all layers (original behavior)
                return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
                self.attentions.append(
                    return_value
                    + bias_term[np.newaxis, np.newaxis].cpu()
                    / (return_value.shape[1])
                )  # [b, h, d]
        return ret

    # @torch.no_grad()
    def compute_mlps(self, ret):
        # Safety check: ensure mlps attribute exists and is a list, initialize if not
        if not hasattr(self, 'mlps') or not isinstance(self.mlps, list):
            self.mlps = []
        
        # Get current layer index (mlps is called after resblock, so current_layer should equal resblock index)
        # Note: ln_pre_post also calls this function, so we need to distinguish
        # For simplicity, we use current_layer to judge (but need adjustment, since ln_pre_post is the 0th)
        layer_idx = self.current_layer - 1 if self.current_layer > 0 else -1  # -1 represents ln_pre_post
        
        if self.training_mode:
            if self.keep_last_n_layers is not None and layer_idx >= 0:
                # Only save last N layers' mlps (excluding ln_pre_post)
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    self.mlps.append(ret[:, 0])  # [b, d]
                else:
                    self.mlps.append(None)  # Placeholder
            else:
                self.mlps.append(ret[:, 0])  # [b, d]
        else:
            # Inference mode: also handle keep_last_n_layers to save memory
            if self.keep_last_n_layers is not None and layer_idx >= 0:
                # Only save last N layers' mlps (excluding ln_pre_post)
                if layer_idx >= self.total_layers - self.keep_last_n_layers:
                    self.mlps.append(ret[:, 0].detach().cpu())  # [b, d]
                else:
                    self.mlps.append(None)  # Placeholder
            else:
                # Save all mlps (including ln_pre_post)
                self.mlps.append(ret[:, 0].detach().cpu())  # [b, d]
        return ret

    # @torch.no_grad()
    def log_post_ln_mean(self, ret):
        if self.training_mode:
            self.post_ln_mean = ret
        else:
            self.post_ln_mean = ret.detach().cpu()  # [b, 1]
        return ret

    # @torch.no_grad()
    def log_post_ln_std(self, ret):
        if self.training_mode:
            self.post_ln_std = ret
        else:
            self.post_ln_std = ret.detach().cpu()  # [b, 1]
        return ret

    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        if self.training_mode:
            mean_centered = (
                self.mlps
                - self.post_ln_mean[:, :, np.newaxis] / len_intermediates
            )
            weighted_mean_centered = (
                self.model.visual.ln_post.weight * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis
            ]
            bias_term = (
                self.model.visual.ln_post.bias / len_intermediates
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj
        else:
            mean_centered = (
                self.mlps
                - self.post_ln_mean[:, :, np.newaxis].to(self.device) / len_intermediates
            )
            weighted_mean_centered = (
                self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis
            ].to(self.device)
            bias_term = (
                self.model.visual.ln_post.bias.detach().to(self.device) / len_intermediates
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj.detach().to(self.device)

    def _normalize_attentions_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2] * self.attentions.shape[3]
        )  # n * h
        # This is just the normalization layer:
        if self.training_mode:
            mean_centered = self.attentions - self.post_ln_mean[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ] / (len_intermediates * normalization_term)
            weighted_mean_centered = (
                self.model.visual.ln_post.weight * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ]
            bias_term = self.model.visual.ln_post.bias / (
                len_intermediates * normalization_term
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj
        else:
            mean_centered = self.attentions - self.post_ln_mean[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ].to(self.device) / (len_intermediates * normalization_term)
            weighted_mean_centered = (
                self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ].to(self.device)
            bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / (
                len_intermediates * normalization_term
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj.detach().to(self.device)

    def _normalize_attentions_non_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2]
        )  # h
        # This is just the normalization layer:
        if self.training_mode:
            mean_centered = self.attentions - self.post_ln_mean[
                :, :, np.newaxis, np.newaxis
            ] / (len_intermediates * normalization_term)
            weighted_mean_centered = (
                self.model.visual.ln_post.weight * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis, np.newaxis
            ]
            bias_term = self.model.visual.ln_post.bias / (
                len_intermediates * normalization_term
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj
        else:
            mean_centered = self.attentions - self.post_ln_mean[
                :, :, np.newaxis, np.newaxis
            ].to(self.device) / (len_intermediates * normalization_term)
            weighted_mean_centered = (
                self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
            )
            weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
                :, :, np.newaxis, np.newaxis
            ].to(self.device)
            bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / (
                len_intermediates * normalization_term
            )
            post_ln = weighted_mean_by_std + bias_term
            return post_ln @ self.model.visual.proj.detach().to(self.device)

    # @torch.no_grad()
    def finalize(self, representation):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        if self.training_mode:
            # Training mode: filter out None (unsaved layers)
            if self.keep_last_n_layers is not None:
                # Only keep last N layers (filter out None)
                self.attentions = [attn for attn in self.attentions if attn is not None]
                # mlps also need filtering, but keep ln_pre_post (the first one)
                filtered_mlps = [mlp for mlp in self.mlps if mlp is not None]
                # If filtered list is empty, there's a problem
                if len(filtered_mlps) == 0:
                    raise ValueError("After filtering None values, no MLP outputs remain. Check keep_last_n_layers setting.")
                self.mlps = filtered_mlps
            
            # Stack on GPU, preserve gradients
            if len(self.attentions) == 0:
                raise ValueError("No attention outputs collected. Check if hooks are properly registered.")
            if len(self.mlps) == 0:
                raise ValueError("No MLP outputs collected. Check if hooks are properly registered.")
            self.attentions = torch.stack(self.attentions, axis=1)  # [b, n_layers, n, h, d]
            self.mlps = torch.stack(self.mlps, axis=1)  # [b, l + 1, d]
        else:
            # Inference mode: also filter out None (if keep_last_n_layers is used)
            if self.keep_last_n_layers is not None:
                # Only keep last N layers (filter out None)
                self.attentions = [attn for attn in self.attentions if attn is not None]
                # mlps also need filtering
                filtered_mlps = [mlp for mlp in self.mlps if mlp is not None]
                self.mlps = filtered_mlps if len(filtered_mlps) > 0 else self.mlps
            
            # Stack on CPU (inference mode)
            if len(self.attentions) == 0:
                raise ValueError("No attention outputs collected. Check if hooks are properly registered.")
            if len(self.mlps) == 0:
                raise ValueError("No MLP outputs collected. Check if hooks are properly registered.")
            self.attentions = torch.stack(self.attentions, axis=1).to(
                self.device
            )  # [b, n_layers, n, h, d]
            self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        
        if self.spatial:
            projected_attentions = self._normalize_attentions_spatial()
        else:
            projected_attentions = self._normalize_attentions_non_spatial()
        projected_mlps = self._normalize_mlps()
        
        if self.training_mode:
            norm = representation.norm(dim=-1)  # Preserve gradients
        else:
            norm = representation.norm(dim=-1).detach()
        
        if self.spatial:
            return (
                projected_attentions
                / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                projected_mlps / norm[:, np.newaxis, np.newaxis],
            )
        return (
            projected_attentions
            / norm[:, np.newaxis, np.newaxis, np.newaxis],
            projected_mlps / norm[:, np.newaxis, np.newaxis],
        )
        
    def reinit(self):
        """Reset PRS logger state, prepare for new forward pass"""
        # Force reset current_layer, ensure starting from 0
        self.current_layer = 0
        # Ensure reset to list, even if previously converted to Tensor
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        if not self.training_mode:
            torch.cuda.empty_cache()


def hook_prs_logger(model, device, spatial: bool = True, training_mode: bool = False, keep_last_n_layers: int = None):
    """Hooks a projected residual stream logger to the model.
    
    Args:
        model: CLIP model
        device: device to use
        spatial: whether to use spatial attention (default: True)
        training_mode: if True, preserves gradients for training (default: False)
        keep_last_n_layers: if set, only keep last N layers in training mode to save memory (default: None)
    """
    prs = PRSLogger(model, device, spatial=spatial, training_mode=training_mode, keep_last_n_layers=keep_last_n_layers)
    if spatial:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_spatial
        )
    else:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_non_spatial
        )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.mlp.c_proj.post", prs.compute_mlps
    )
    model.hook_manager.register("visual.ln_pre_post", prs.compute_mlps)
    model.hook_manager.register("visual.ln_post.mean", prs.log_post_ln_mean)
    model.hook_manager.register("visual.ln_post.sqrt_var", prs.log_post_ln_std)
    return prs
