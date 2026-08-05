#!/usr/bin/env python3
"""
Train Hippocampal MoE with DG-based Gating + Analyze Model
This script implements a novel gating mechanism based on Dentate Gyrus (DG)
pattern similarity instead of a separate gating network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import torch.optim as optim
import sys
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score, pairwise_distances
import random
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Import from the optimal training script
from train_hippocampal_optimal_moe import *

# Import SparseActivation for our custom DG expert
from train_hippocampal_optimal_moe import SparseActivation


dg_dim=512

# ============================================================================
# CUSTOM TWO-CLASS RGB IMAGE DATASET
# ============================================================================

def create_custom_rgb_dataloaders(
    data_dir,
    batch_size=64,
    image_size=32,
    test_split=0.2,
    num_workers=4,
    seed=42,
    normalize_mean=(0.5, 0.5, 0.5),
    normalize_std=(0.5, 0.5, 0.5),
    disable_normalization=False,
):
    """
    Create one training task containing exactly two RGB image classes.

    Supported directory layouts
    ---------------------------
    Layout A: pre-split dataset

        data_dir/
            train/
                class_a/
                class_b/
            test/
                class_a/
                class_b/

    Layout B: one folder per class; this function makes a stratified split

        data_dir/
            class_a/
            class_b/

    All images are loaded as RGB by torchvision's ImageFolder loader,
    resized to image_size x image_size, converted to tensors, and optionally
    normalized. The returned loaders are wrapped in one-element lists because
    the rest of this script expects one loader per task/expert.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    if image_size < 8:
        raise ValueError("image_size must be at least 8 pixels.")
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1.")

    transform_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if not disable_normalization:
        transform_steps.append(
            transforms.Normalize(
                mean=tuple(normalize_mean),
                std=tuple(normalize_std),
            )
        )
    image_transform = transforms.Compose(transform_steps)

    train_root = data_dir / "train"
    test_root = data_dir / "test"

    if train_root.is_dir() and test_root.is_dir():
        train_dataset = datasets.ImageFolder(
            root=str(train_root),
            transform=image_transform,
        )
        test_dataset = datasets.ImageFolder(
            root=str(test_root),
            transform=image_transform,
        )

        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise ValueError(
                "The train and test folders must contain the same class-folder "
                "names. Found train mapping "
                f"{train_dataset.class_to_idx} and test mapping "
                f"{test_dataset.class_to_idx}."
            )
        class_names = train_dataset.classes

    elif not train_root.exists() and not test_root.exists():
        full_dataset = datasets.ImageFolder(
            root=str(data_dir),
            transform=image_transform,
        )
        class_names = full_dataset.classes

        # Stratified splitting ensures that both classes occur in both sets.
        generator = torch.Generator().manual_seed(seed)
        train_indices = []
        test_indices = []

        targets = torch.tensor(full_dataset.targets, dtype=torch.long)
        for class_id in range(len(class_names)):
            class_indices = torch.where(targets == class_id)[0]
            if len(class_indices) < 2:
                raise ValueError(
                    f"Class '{class_names[class_id]}' contains only "
                    f"{len(class_indices)} image(s). At least two are required "
                    "when an automatic train/test split is used."
                )

            order = torch.randperm(len(class_indices), generator=generator)
            class_indices = class_indices[order].tolist()

            num_test = max(1, int(round(len(class_indices) * test_split)))
            num_test = min(num_test, len(class_indices) - 1)

            test_indices.extend(class_indices[:num_test])
            train_indices.extend(class_indices[num_test:])

        # Shuffle the combined stratified index lists deterministically.
        train_order = torch.randperm(
            len(train_indices), generator=generator
        ).tolist()
        test_order = torch.randperm(
            len(test_indices), generator=generator
        ).tolist()
        train_indices = [train_indices[i] for i in train_order]
        test_indices = [test_indices[i] for i in test_order]

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

    else:
        raise ValueError(
            "Use either both data_dir/train and data_dir/test, or neither. "
            "A partially pre-split dataset is ambiguous."
        )

    if len(class_names) != 2:
        raise ValueError(
            "This configuration requires exactly two class folders, but found "
            f"{len(class_names)}: {class_names}"
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    class_name_map = {index: name for index, name in enumerate(class_names)}
    task_classes = [[0, 1]]

    logging.info("🖼️ Loaded custom two-class RGB dataset")
    logging.info(f"  Root: {data_dir}")
    logging.info(f"  Classes: {class_name_map}")
    logging.info(f"  Train samples: {len(train_dataset)}")
    logging.info(f"  Test samples: {len(test_dataset)}")
    logging.info(
        f"  Model input: RGB {image_size}×{image_size} "
        f"({3 * image_size * image_size} flattened values)"
    )

    return [train_loader], [test_loader], task_classes, class_name_map


# ============================================================================
# GRID CELL LAYER
# ============================================================================

class GridCellLayer(nn.Module):
    """
    Implements grid cell-like spatial encoding from entorhinal cortex
    Uses multiple spatial frequencies to create hexagonal grid patterns
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Different spatial frequencies (like biological grid cells)
        self.freq1 = nn.Conv2d(channels, channels//4, 1)
        self.freq2 = nn.Conv2d(channels, channels//4, 1)
        self.freq3 = nn.Conv2d(channels, channels//4, 1)
        self.freq4 = nn.Conv2d(channels, channels//4, 1)

        # Phase offsets (creates hexagonal patterns)
        self.register_buffer('phase1', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase2', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase3', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase4', torch.randn(1, channels//4, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Different spatial frequencies with phase offsets
        y1 = torch.sin(self.freq1(x) + self.phase1)
        y2 = torch.sin(self.freq2(x) + self.phase2)
        y3 = torch.sin(self.freq3(x) + self.phase3)
        y4 = torch.sin(self.freq4(x) + self.phase4)
        return torch.cat([y1, y2, y3, y4], dim=1)

# ============================================================================
# STANDARD FEATURE EXTRACTOR (COPIED FROM Y.PY)
# ============================================================================

class StandardFeatureExtractor(nn.Module):
    """
    Standard LeNet-style feature extractor with regular convolutions.
    """
    def __init__(self, input_channels, use_small_features=False):
        super().__init__()
        self.use_small_features = use_small_features
        
        if use_small_features:
            # Small version: 3→32→64→128 (like HippoLeNet small)
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                GridCellLayer(32),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        else:
            # Standard version: 3→64→128→256
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                GridCellLayer(64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# CUSTOM ENHANCED HIPPOCAMPAL EXPERT FOR DG-GATED MODEL
# ============================================================================

class CustomDentateGyrusExpert(nn.Module):
    """
    Dentate Gyrus projection and sparse pattern-separation layer.

    This explicitly implements the operation shown in the paper:

        z = ReLU(W_DG g + b_DG)
        p_sep = TopK(LayerNorm(z), k)

    `SparseActivation` is imported from train_hippocampal_optimal_moe.py and
    performs the TopK sparsification using `percent_on`.
    """
    def __init__(self, input_dim, hidden_dim, sparsity=0.05, expansion_factor=1):
        super().__init__()

        # W_DG g + b_DG: raw flattened input is projected directly to DG space.
        # There is no CNN/GridCell feature extractor before this layer.
        self.dg_projection = nn.Linear(input_dim, hidden_dim)
        self.dg_relu = nn.ReLU()
        self.dg_layer_norm = nn.LayerNorm(hidden_dim)

        # TopK(..., k), where k = ceil(sparsity * hidden_dim).
        # Keep this active: this is the sparse DG pattern-separation operation.
        self.sparse_activation = SparseActivation(percent_on=sparsity)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_features = hidden_dim
        self.sparsity = sparsity

    def forward(self, g):
        if g.ndim != 2:
            raise ValueError(
                f"DG expects a flattened tensor of shape [batch, {self.input_dim}], "
                f"but received {tuple(g.shape)}."
            )
        if g.size(1) != self.input_dim:
            raise ValueError(
                f"DG input dimension mismatch: expected {self.input_dim}, "
                f"received {g.size(1)}."
            )

        z = self.dg_relu(self.dg_projection(g))
        p_sep = self.sparse_activation(self.dg_layer_norm(z))
        return p_sep
    
    
class CustomCA3PatternCompletion(nn.Module):
    """
    Custom CA3 Pattern Completion for the DG-Gated model with reduced dim.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pattern_completion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # No expansion
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),  # Keep dim
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        # print(f"[CA3] Input shape: {x.shape}")
        out = self.pattern_completion(x)
        # print(f"[CA3] Output shape: {out.shape}")
        return out

class CustomEnhancedHippocampalExpert(nn.Module):
    """
    Custom Enhanced Hippocampal Expert for the DG-Gated model with expanded dims.
    """
    def __init__(self, input_dim, dg_dim, ca3_dim, target_sparsity=0.05, dropout_rate=0.1):
        super().__init__()
        self.dg = CustomDentateGyrusExpert(input_dim, dg_dim, target_sparsity)
        self.ca3 = CustomCA3PatternCompletion(dg_dim, ca3_dim)
        # CA1 integration: [DG + CA3 + raw features] -> dg_dim -> 256 -> 128
        self.ca1_integration = nn.Sequential(
            nn.Linear(dg_dim + ca3_dim + input_dim, dg_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(dg_dim),
            nn.Linear(dg_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(256),
            nn.Linear(256, 128),  # Changed from 256 to 128 to match output layers
            nn.Dropout(dropout_rate)
        )
    def forward(self, features):
        dg_output = self.dg(features)
        ca3_output = self.ca3(dg_output)
        combined = torch.cat([dg_output, ca3_output, features], dim=1)
        ca1_output = self.ca1_integration(combined)
        return dg_output, ca1_output

# ============================================================================
# NEW DG-GATED HIPPOCAMPAL MOE
# ============================================================================

class DGGatedHippocampalMoE(OptimalHippocampalMoE):
    """
    DG-Gated Hippocampal MoE with online per-class EMA prototype computation.
    """
    
    def __init__(self, num_experts, classes_per_task, input_channels, target_sparsity=0.05, memory_size=200, use_small_features=False, image_size=32):
        # Call parent with correct parameters
        super().__init__(num_experts, classes_per_task, input_channels, target_sparsity, memory_size)
        
        # Store target_sparsity as instance variable
        self.target_sparsity = target_sparsity
        
        # Initialize prototype tracking variables
        self.class_proto_ema = None
        self.class_proto_counts = None
        self.frozen_class_mask = None
        self.warmup_batches = 30
        self.ema_momentum = 0.9
        self.num_classes = num_experts * classes_per_task
        
        # Initialize EWC data storage
        self.ewc_data = []
        
        # Initialize replay buffer
        self.replay_buffer = [[] for _ in range(num_experts)]
        self.memory_size_per_task = memory_size
        
        # Gating parameters
        self.gating_strategy = 'soft_hard'
        self.gating_temperature = 1.0
        
        # Task classes will be set later
        self.task_classes = None
        
        logging.info(f"🔧 Initialized DG-Gated Hippocampal MoE with {num_experts} experts, {classes_per_task} classes per task")
        logging.info(f"🔧 Target DG sparsity: {target_sparsity}, Memory size per task: {memory_size}")
        logging.info(f"🔧 EMA parameters: warmup={self.warmup_batches}, momentum={self.ema_momentum}")
        
        # ------------------------------------------------------------------
        # BYPASS THE STANDARD CNN/GRID-CELL FEATURE EXTRACTOR
        # ------------------------------------------------------------------
        # Original feature-extraction path (intentionally disabled):
        # self.feature_extractor = StandardFeatureExtractor(
        #     input_channels,
        #     use_small_features=use_small_features,
        # )
        #
        # The parent class may already have created a feature extractor, so
        # replace it explicitly with Identity rather than only commenting out
        # the assignment. All forward paths call prepare_dg_input(), which
        # validates and flattens the raw image tensor directly.
        self.feature_extractor = nn.Identity()

        self.expected_input_channels = input_channels
        self.expected_input_height = image_size
        self.expected_input_width = image_size
        feature_extractor_output_dim = (
            self.expected_input_channels
            * self.expected_input_height
            * self.expected_input_width
        )

        logging.info(
            "🔧 StandardFeatureExtractor disabled; passing raw inputs "
            f"directly to DG as {feature_extractor_output_dim}-D vectors "
            f"({input_channels}×{image_size}×{image_size})."
        )

        # Kept only for compatibility with code that checks this attribute.
        self.feature_projector = None
        
        # Use CustomEnhancedHippocampalExpert with controlled parameters
        self.hippocampal_experts = nn.ModuleList([
            CustomEnhancedHippocampalExpert(
                input_dim=feature_extractor_output_dim,
                dg_dim=dg_dim,
                ca3_dim=256,
                target_sparsity=target_sparsity,
                dropout_rate=0.1
            ) for _ in range(num_experts)
        ])
        
        # Use parent class output layers (CA1 output dimension is 128)
        self.output_layers = nn.ModuleList([
            nn.Linear(128, classes_per_task) for _ in range(num_experts)
        ])
        
        # Initialize online EMA tracking for prototypes
        self.class_proto_ema = None  # Will be initialized when first expert starts
        self.class_proto_counts = None  # Will be initialized when first expert starts
        self.frozen_class_mask = None  # Will be initialized when first expert starts
        self.ema_momentum = 0.9  # EMA momentum for prototype updates
        self.warmup_batches = 30  # Number of batches for warm-up (cumulative mean)
        
        # Track which experts have finished training
        self.trained_experts = 0
        
        # Add memory buffer for replay
        self.memory_size = memory_size
        self.replay_buffer = [[] for _ in range(self.num_experts)]
        self.memory_size_per_task = self.memory_size
        
        # Initialize DG prototypes buffer (will be updated by EMA)
        self.register_buffer('dg_prototypes', torch.zeros(num_experts, dg_dim))
        self.prototypes_computed = False
        
        # Remove gating network since we use DG-based gating
        self.gating_network = None
        
        # Add feature_to_ca1 projection for feature distillation (from parent class)
        self.feature_to_ca1 = nn.Linear(feature_extractor_output_dim, 128)  # Flattened raw RGB input -> 128 for legacy CA1 distillation
        
        # Log parameter counts for verification
        total_params = sum(p.numel() for p in self.parameters())
        expert_params = sum(p.numel() for expert in self.hippocampal_experts for p in expert.parameters())
        logging.info(f"🔧 Total model parameters: {total_params:,}")
        logging.info(f"🔧 Expert parameters: {expert_params:,} ({expert_params/num_experts:,} per expert)")
    
    def prepare_dg_input(self, x):
        """
        Validate and flatten raw inputs before passing them directly to DG.

        Expected input: [batch, input_channels, image_size, image_size]
        Returned input: [batch, input_channels * 32 * 32]

        No convolution, pooling, grid-cell encoding, resizing, or learned
        feature extraction is performed here.
        """
        if x.ndim != 4:
            raise ValueError(
                "Raw DG input must have shape [batch, channels, height, width], "
                f"but received {tuple(x.shape)}."
            )

        expected_shape = (
            self.expected_input_channels,
            self.expected_input_height,
            self.expected_input_width,
        )
        received_shape = tuple(x.shape[1:])

        if received_shape != expected_shape:
            raise ValueError(
                "Raw DG input shape mismatch: expected each sample to have "
                f"shape {expected_shape}, but received {received_shape}. "
                "Change expected_input_height/width and the expert input_dim "
                "together if using a different image resolution."
            )

        return torch.flatten(x, start_dim=1)

    def compute_fisher_importance(self, dataloader, device, num_samples=500):
        """
        Compute Fisher Information Matrix for EWC.
        This is the one-time calculation between tasks.
        IMPROVED: Now captures ALL trainable parameters, not just those with gradients.
        """
        self.eval()
        fisher_info = {}
        
        # Initialize fisher info for ALL trainable parameters
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                total_params += param.numel()
        
        logging.info(f"🔧 EWC: Initializing Fisher info for {len(fisher_info)} parameter groups ({total_params:,} total parameters)")
        
        # Sample data for Fisher computation
        sample_count = 0
        for inputs, labels in dataloader:
            if sample_count >= num_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass - use a simplified version that doesn't require DG prototypes
            self.zero_grad()
            
            # Prepare raw flattened DG inputs
            features = self.prepare_dg_input(inputs)
            
            # Get outputs from all experts (without gating)
            all_outputs = []
            for expert_id in range(self.num_experts):
                dg_output, ca1_output = self.hippocampal_experts[expert_id](features)
                expert_output = self.output_layers[expert_id](ca1_output)
                all_outputs.append(expert_output)
            
            # Concatenate all expert outputs
            outputs = torch.cat(all_outputs, dim=1)
            
            # Compute loss (assuming classification)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate Fisher information for ALL parameters
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data ** 2
                    else:
                        # For parameters without gradients, add small epsilon to avoid zeros
                        # This ensures all parameters are protected by EWC
                        fisher_info[name] += torch.ones_like(param.data) * 1e-8
            
            sample_count += inputs.size(0)
        
        # Average over samples
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        # Log statistics about Fisher information
        non_zero_params = sum(1 for fisher in fisher_info.values() if fisher.sum() > 1e-8)
        total_fisher_params = sum(fisher.numel() for fisher in fisher_info.values())
        
        logging.info(f"🔧 EWC: Computed Fisher info for {non_zero_params}/{len(fisher_info)} parameter groups")
        logging.info(f"🔧 EWC: Total Fisher parameters: {total_fisher_params:,}")
        
        self.train()
        return fisher_info

    def compute_fisher_importance_enhanced(self, dataloader, device, num_samples=500, num_forward_passes=3):
        """
        Enhanced Fisher Information Matrix computation for EWC.
        Uses multiple forward passes with different inputs to get more robust estimates.
        """
        self.eval()
        fisher_info = {}
        
        # Initialize fisher info for ALL trainable parameters
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                total_params += param.numel()
        
        logging.info(f"🔧 EWC Enhanced: Initializing Fisher info for {len(fisher_info)} parameter groups ({total_params:,} total parameters)")
        
        # Sample data for Fisher computation
        sample_count = 0
        for inputs, labels in dataloader:
            if sample_count >= num_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Multiple forward passes with different inputs for robustness
            batch_fisher = {name: torch.zeros_like(param.data) for name, param in self.named_parameters() if param.requires_grad}
            
            for pass_idx in range(num_forward_passes):
                self.zero_grad()
                
                # Prepare raw flattened DG inputs
                features = self.prepare_dg_input(inputs)
                
                # Get outputs from all experts (without gating)
                all_outputs = []
                for expert_id in range(self.num_experts):
                    dg_output, ca1_output = self.hippocampal_experts[expert_id](features)
                    expert_output = self.output_layers[expert_id](ca1_output)
                    all_outputs.append(expert_output)
                
                # Concatenate all expert outputs
                outputs = torch.cat(all_outputs, dim=1)
                
                # Compute loss (assuming classification)
                loss = F.cross_entropy(outputs, labels)
                
                # Backward pass to get gradients
                loss.backward()
                
                # Accumulate Fisher information for this forward pass
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            batch_fisher[name] += param.grad.data ** 2
                        else:
                            # For parameters without gradients, add small epsilon
                            batch_fisher[name] += torch.ones_like(param.data) * 1e-8
            
            # Average over forward passes and accumulate
            for name in batch_fisher:
                batch_fisher[name] /= num_forward_passes
                fisher_info[name] += batch_fisher[name]
            
            sample_count += inputs.size(0)
        
        # Average over samples
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        # Log statistics about Fisher information
        non_zero_params = sum(1 for fisher in fisher_info.values() if fisher.sum() > 1e-8)
        total_fisher_params = sum(fisher.numel() for fisher in fisher_info.values())
        
        logging.info(f"🔧 EWC Enhanced: Computed Fisher info for {non_zero_params}/{len(fisher_info)} parameter groups")
        logging.info(f"🔧 EWC Enhanced: Total Fisher parameters: {total_fisher_params:,}")
        
        self.train()
        return fisher_info

    def analyze_fisher_quality(self, fisher_matrix):
        """
        Analyze the quality of Fisher information matrix.
        Provides diagnostics to ensure EWC is capturing meaningful parameter importance.
        """
        if not fisher_matrix:
            return {}
        
        analysis = {}
        
        # Count parameters by type
        total_params = 0
        non_zero_params = 0
        param_types = {}
        
        for name, fisher_values in fisher_matrix.items():
            param_count = fisher_values.numel()
            total_params += param_count
            
            # Count non-zero Fisher values
            non_zero_count = (fisher_values > 1e-8).sum().item()
            non_zero_params += non_zero_count
            
            # Categorize by parameter type
            if 'feature_extractor' in name:
                param_type = 'feature_extractor'
            elif 'hippocampal_experts' in name:
                param_type = 'hippocampal_experts'
            elif 'output_layers' in name:
                param_type = 'output_layers'
            elif 'gate' in name:
                param_type = 'gating'
            else:
                param_type = 'other'
            
            if param_type not in param_types:
                param_types[param_type] = {'total': 0, 'non_zero': 0}
            param_types[param_type]['total'] += param_count
            param_types[param_type]['non_zero'] += non_zero_count
        
        # Calculate statistics
        analysis['total_parameters'] = total_params
        analysis['non_zero_parameters'] = non_zero_params
        analysis['coverage_ratio'] = non_zero_params / total_params if total_params > 0 else 0
        analysis['param_types'] = param_types
        
        # Fisher value statistics
        all_fisher_values = torch.cat([f.flatten() for f in fisher_matrix.values()])
        analysis['fisher_stats'] = {
            'mean': float(all_fisher_values.mean()),
            'std': float(all_fisher_values.std()),
            'min': float(all_fisher_values.min()),
            'max': float(all_fisher_values.max()),
            'median': float(all_fisher_values.median())
        }
        
        # Log analysis
        logging.info(f"🔧 EWC Fisher Analysis:")
        logging.info(f"  - Total parameters: {total_params:,}")
        logging.info(f"  - Non-zero Fisher values: {non_zero_params:,}")
        logging.info(f"  - Coverage ratio: {analysis['coverage_ratio']:.3f}")
        logging.info(f"  - Fisher value range: [{analysis['fisher_stats']['min']:.2e}, {analysis['fisher_stats']['max']:.2e}]")
        logging.info(f"  - Fisher mean/std: {analysis['fisher_stats']['mean']:.2e}/{analysis['fisher_stats']['std']:.2e}")
        
        for param_type, stats in param_types.items():
            coverage = stats['non_zero'] / stats['total'] if stats['total'] > 0 else 0
            logging.info(f"  - {param_type}: {stats['non_zero']:,}/{stats['total']:,} ({coverage:.3f})")
        
        return analysis

    def calculate_ewc_loss(self, ewc_lambda=1000.0):
        """
        Calculate EWC loss to prevent forgetting of previous tasks.
        IMPROVED: Better logging and more robust calculation.
        """
        if not self.ewc_data:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        ewc_loss = 0.0
        total_ewc_terms = 0
        
        for task_idx, task_data in enumerate(self.ewc_data):
            fisher_matrix = task_data['fisher']
            star_params = task_data['star_params']
            
            task_ewc_loss = 0.0
            task_terms = 0
            
            for name, param in self.named_parameters():
                if name in fisher_matrix and name in star_params:
                    if param.requires_grad:
                        # Compute squared difference from optimal parameters
                        param_diff = param - star_params[name]
                        fisher_weighted_diff = fisher_matrix[name] * (param_diff ** 2)
                        task_ewc_loss += fisher_weighted_diff.sum()
                        task_terms += fisher_weighted_diff.numel()
            
            ewc_loss += task_ewc_loss
            total_ewc_terms += task_terms
            
            # Log per-task EWC contribution
            if task_ewc_loss > 0:
                logging.debug(f"🔧 EWC Task {task_idx}: Loss={task_ewc_loss:.6f}, Terms={task_terms:,}")
        
        # Apply lambda scaling
        final_ewc_loss = ewc_lambda * ewc_loss
        
        # Log EWC statistics
        if ewc_loss > 0:
            logging.debug(f"🔧 EWC Total: Loss={ewc_loss:.6f}, Scaled={final_ewc_loss:.6f}, Total Terms={total_ewc_terms:,}")
        
        return final_ewc_loss
        
    def initialize_prototype_tracking(self, dg_dim, num_classes, device):
        """Initialize EMA tracking variables for online prototype computation."""
        self.class_proto_ema = torch.zeros(num_classes, dg_dim, device=device)
        self.class_proto_counts = torch.zeros(num_classes, device=device)
        self.frozen_class_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        logging.info(f"🔧 Initialized online prototype tracking: {num_classes} classes, {dg_dim} DG dims")
    
    def update_class_prototype_ema(self, dg_outputs, labels, expert_id):
        """
        Update class prototypes using EMA for the current expert's classes.
        
        Args:
            dg_outputs: DG outputs from current expert [batch_size, dg_dim]
            labels: Global class labels [batch_size]
            expert_id: Current expert being trained
        """
        if self.class_proto_ema is None:
            # Initialize tracking on first call
            dg_dim = dg_outputs.size(1)
            num_classes = self.num_classes
            self.initialize_prototype_tracking(dg_dim, num_classes, dg_outputs.device)
        
        # Check if task_classes is initialized
        if not hasattr(self, 'task_classes') or self.task_classes is None or len(self.task_classes) <= expert_id:
            return  # Skip if task_classes not ready
        
        # Check if tracking variables are initialized
        if (self.class_proto_ema is None or self.class_proto_counts is None or 
            self.frozen_class_mask is None):
            return
        
        with torch.no_grad():
            # Get classes for current expert
            expert_classes = self.task_classes[expert_id]
            
            for class_id in expert_classes:
                if self.frozen_class_mask[class_id]:
                    continue  # Skip frozen classes
                
                # Find samples belonging to this class
                class_mask = (labels == class_id)
                if class_mask.sum() > 0:
                    class_dg_outputs = dg_outputs[class_mask]
                    class_mean = class_dg_outputs.mean(dim=0)
                    
                    if self.class_proto_counts[class_id] < self.warmup_batches:
                        # Cumulative mean during warm-up
                        total_prev = self.class_proto_counts[class_id].item()
                        self.class_proto_ema[class_id] = (self.class_proto_ema[class_id] * total_prev + class_mean) / (total_prev + 1)
                    else:
                        # EMA update after warm-up
                        self.class_proto_ema[class_id] = (1 - self.ema_momentum) * self.class_proto_ema[class_id] + self.ema_momentum * class_mean
                    
                    self.class_proto_counts[class_id] += 1
    
    def freeze_expert_prototypes(self, expert_id):
        """Freeze prototypes for all classes of the given expert."""
        if (self.frozen_class_mask is not None and 
            hasattr(self, 'task_classes') and 
            self.task_classes is not None and 
            len(self.task_classes) > expert_id):
            expert_classes = self.task_classes[expert_id]
            for class_id in expert_classes:
                self.frozen_class_mask[class_id] = True
            logging.info(f"🔒 Frozen prototypes for expert {expert_id} classes: {expert_classes}")
    
    def compute_expert_prototypes_from_classes(self, expert_id):
        """Compute expert prototype as mean of its class prototypes."""
        if (self.class_proto_ema is None or 
            not hasattr(self, 'task_classes') or 
            self.task_classes is None or 
            len(self.task_classes) <= expert_id):
            return None
        
        expert_classes = self.task_classes[expert_id]
        class_prototypes = self.class_proto_ema[expert_classes]
        expert_prototype = class_prototypes.mean(dim=0)
        return expert_prototype
    
    def get_active_prototypes_for_diagnostics(self, expert_id):
        """
        Get active prototypes for diagnostics, including current expert's prototype
        even if not yet frozen.
        """
        if not hasattr(self, 'dg_prototypes') or self.dg_prototypes is None:
            return None, 0
        
        # Determine active experts (trained + current)
        active_E = max(self.trained_experts, expert_id + 1)
        
        # Create temporary prototypes tensor
        temp_protos = self.dg_prototypes.clone()
        
        # Synthesize current expert prototype from EMA if not frozen yet
        if expert_id >= self.trained_experts:
            cur_proto = self.compute_expert_prototypes_from_classes(expert_id)
            if cur_proto is not None:
                temp_protos[expert_id] = cur_proto
        
        # Return only active prototypes
        active_protos = temp_protos[:active_E]
        return active_protos, active_E
    
    def update_dg_prototypes_from_ema(self):
        """Update dg_prototypes tensor from current EMA class prototypes."""
        if self.class_proto_ema is None:
            return
        
        for expert_id in range(self.trained_experts):
            expert_prototype = self.compute_expert_prototypes_from_classes(expert_id)
            if expert_prototype is not None:
                with torch.no_grad():
                    # Use proper tensor indexing
                    if hasattr(self, 'dg_prototypes') and self.dg_prototypes is not None:
                        self.dg_prototypes[expert_id] = expert_prototype
        
        # Mark prototypes as computed if we have at least one expert
        if self.trained_experts > 0:
            self.prototypes_computed = True
            logging.info(f"🔄 Updated DG prototypes for {self.trained_experts} trained experts")
    
    def log_prototype_stats(self):
        """Log statistics about current prototype quality."""
        if self.class_proto_ema is None or self.trained_experts == 0:
            return
        
        # Calculate prototype separation
        with torch.no_grad():
            if hasattr(self, 'dg_prototypes') and self.dg_prototypes is not None:
                trained_prototypes = self.dg_prototypes[:self.trained_experts]
                if trained_prototypes.size(0) > 1:
                    prototypes_norm = F.normalize(trained_prototypes, p=2, dim=1)
                    sim_matrix = prototypes_norm @ prototypes_norm.T
                
                # Off-diagonal similarities (should be low for good separation)
                mask = ~torch.eye(trained_prototypes.size(0), dtype=torch.bool, device=trained_prototypes.device)
                off_diag_sims = sim_matrix[mask]
                
                mean_separation = off_diag_sims.mean().item()
                min_separation = off_diag_sims.min().item()
                
                logging.info(f"📊 Prototype separation - Mean: {mean_separation:.3f}, Min: {min_separation:.3f}")
                
                # Log class prototype counts
                if self.class_proto_counts is not None:
                    active_classes = (self.class_proto_counts > 0).sum().item()
                    total_classes = self.class_proto_counts.size(0)
                else:
                    active_classes = 0
                    total_classes = 0
                logging.info(f"📊 Class prototypes - Active: {active_classes}/{total_classes}")

    def set_task_classes(self, task_classes):
        """Set the task classes for the model."""
        self.task_classes = task_classes

    def add_to_replay_buffer(self, inputs, labels, task_id):
        """Adds samples to the replay buffer for a given task using random replacement."""
        for i in range(inputs.size(0)):
            if len(self.replay_buffer[task_id]) < self.memory_size_per_task:
                self.replay_buffer[task_id].append((inputs[i], labels[i]))
            else:
                # Randomly replace an existing sample
                idx = np.random.randint(0, self.memory_size_per_task)
                self.replay_buffer[task_id][idx] = (inputs[i], labels[i])

    def sample_from_replay_buffer(self, task_id, batch_size):
        """Samples a batch from the replay buffer of a given task."""
        if not self.replay_buffer[task_id] or batch_size == 0:
            return None, None
        
        buffer = self.replay_buffer[task_id]
        # Ensure batch_size is not larger than the number of available samples
        actual_batch_size = min(batch_size, len(buffer))
        
        sample_indices = np.random.choice(len(buffer), size=actual_batch_size, replace=len(buffer) < actual_batch_size)
        
        samples = [buffer[i] for i in sample_indices]
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        
        return inputs, labels

    def set_gating_strategy(self, strategy):
        """Sets the gating strategy for inference."""
        if strategy not in ['hard', 'soft', 'top2', 'soft_hard']:
            raise ValueError("Gating strategy must be one of 'hard', 'soft', 'top2', or 'soft_hard'")
        self.gating_strategy = strategy
        logging.info(f"🚪 Set gating strategy to: {self.gating_strategy}")

    def set_gating_temperature(self, temperature):
        """Sets the temperature for softmax gating."""
        self.gating_temperature = temperature
        logging.info(f"🌡️ Set gating temperature to: {self.gating_temperature}")

    def forward_all_tasks(self, x):
        """
        Forward pass for Class-IL evaluation using DG-gating.
        This simply calls the standard forward method without a task_id.
        """
        final_outputs, _, _ = self.forward(x)
        return final_outputs
        
    def compute_dg_prototypes(self, train_loaders, device):
        """
        Computes the prototype DG pattern for each expert by averaging the DG
        output over all training samples for that expert's task.
        """
        logging.info("🧠 Computing all DG Prototypes post-Phase 1...")
        self.eval()
        with torch.no_grad():
            for task_id, train_loader in enumerate(tqdm(train_loaders, desc="Computing Prototypes")):
                all_dg_outputs = []
                for inputs, _ in train_loader:
                    inputs = inputs.to(device)
                    features = self.prepare_dg_input(inputs)
                    dg_output, _ = self.hippocampal_experts[task_id](features)
                    all_dg_outputs.append(dg_output)
                
                # Average all DG outputs for this task
                with torch.no_grad():
                    self.dg_prototypes[task_id] = torch.cat(all_dg_outputs, dim=0).mean(dim=0)
        
        self.prototypes_computed = True
        logging.info("✅ All expert prototypes computed and stored.")

    def forward(self, x, task_id=None):
        """
        Forward pass with DG-based gating.
        If task_id is provided, it uses oracle routing. Otherwise, it uses
        DG pattern similarity to find the best expert.
        Returns a dictionary in the third position for analysis data.
        """
        if not self.prototypes_computed and task_id is None:
            # During inference, prototypes must have been computed.
            if not self.training:
                raise RuntimeError("DG prototypes have not been computed. Call compute_dg_prototypes() first.")

        features = self.prepare_dg_input(x)
        gate_logits = None # Default for oracle routing
        analysis_data = {}

        if self.training and task_id is not None:
            # Oracle routing for training experts
            # Process through the single chosen expert
            dg_output, ca1_output = self.hippocampal_experts[task_id](features)
            expert_output = self.output_layers[task_id](ca1_output)
            
            final_outputs = torch.zeros(x.size(0), self.num_classes, device=x.device)
            start_idx = task_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            final_outputs[:, start_idx:end_idx] = expert_output
            
            analysis_data['dg_output'] = dg_output
            return final_outputs, gate_logits, analysis_data

        # --- Gating for Inference ---
        all_dg_outputs = []
        for i in range(self.num_experts):
            dg_output, _ = self.hippocampal_experts[i](features)
            all_dg_outputs.append(dg_output)
        
        all_dg_outputs = torch.stack(all_dg_outputs, dim=1)
        all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
        
        # Only use prototypes for trained experts
        trained_prototypes = self.dg_prototypes[:self.trained_experts]
        if trained_prototypes.size(0) == 0:
            # No prototypes available, use uniform routing
            gate_logits = torch.zeros(x.size(0), self.num_experts, device=x.device)
        else:
            prototypes_norm = F.normalize(trained_prototypes, p=2, dim=1).to(x.device)
            # Calculate DG pattern similarity for gating (only for trained experts)
            gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm[:, :self.trained_experts], prototypes_norm)
            # Pad with zeros for untrained experts
            if self.trained_experts < self.num_experts:
                padding = torch.zeros(x.size(0), self.num_experts - self.trained_experts, device=x.device)
                gate_logits = torch.cat([gate_logits, padding], dim=1)
        
        # --- Apply Gating Strategy ---
        final_outputs = torch.zeros(x.size(0), self.num_classes, device=x.device)

        if self.gating_strategy == 'hard':
            # Winner-take-all: choose the expert with the highest similarity
            chosen_experts = torch.argmax(gate_logits, dim=1)
            for i in range(x.size(0)):
                expert_id = int(chosen_experts[i].item())
                _, ca1_output = self.hippocampal_experts[expert_id](features[i].unsqueeze(0))
                expert_output = self.output_layers[expert_id](ca1_output)
                start_idx = expert_id * self.classes_per_task
                end_idx = start_idx + self.classes_per_task
                final_outputs[i, start_idx:end_idx] = expert_output

        elif self.gating_strategy == 'soft':
            # Soft gating: weighted average of all expert outputs
            gating_weights = F.softmax(gate_logits / self.gating_temperature, dim=1)
            for expert_id in range(self.num_experts):
                weight = gating_weights[:, expert_id].unsqueeze(1)
                _, ca1_output = self.hippocampal_experts[expert_id](features)
                expert_output = self.output_layers[expert_id](ca1_output)
                start_idx = expert_id * self.classes_per_task
                end_idx = start_idx + self.classes_per_task
                final_outputs[:, start_idx:end_idx] += weight * expert_output

        elif self.gating_strategy == 'top2':
            # Top-2 gating: weighted average of the top two expert outputs
            top2_logits, top2_indices = torch.topk(gate_logits, 2, dim=1)
            top2_weights = F.softmax(top2_logits / self.gating_temperature, dim=1)
            
            for i in range(x.size(0)):
                for j in range(2):
                    expert_id = int(top2_indices[i, j].item())
                    weight = top2_weights[i, j]
                    _, ca1_output = self.hippocampal_experts[expert_id](features[i].unsqueeze(0))
                    expert_output = self.output_layers[expert_id](ca1_output)
                    start_idx = expert_id * self.classes_per_task
                    end_idx = start_idx + self.classes_per_task
                    final_outputs[i, start_idx:end_idx] += weight * expert_output.squeeze(0)

        elif self.gating_strategy == 'soft_hard':
            # Soft-hard gating: always use soft gating, hard gating only in final evaluation
            # This will be handled in the evaluation function
            gating_weights = F.softmax(gate_logits / self.gating_temperature, dim=1)
            for expert_id in range(self.num_experts):
                weight = gating_weights[:, expert_id].unsqueeze(1)
                _, ca1_output = self.hippocampal_experts[expert_id](features)
                expert_output = self.output_layers[expert_id](ca1_output)
                start_idx = expert_id * self.classes_per_task
                end_idx = start_idx + self.classes_per_task
                final_outputs[:, start_idx:end_idx] += weight * expert_output

        analysis_data['chosen_experts'] = torch.argmax(gate_logits, dim=1) if gate_logits is not None else None
            
        return final_outputs, gate_logits, analysis_data

def analyze_dg_gated_model(model, test_loaders, device, save_dir):
    """Analyze the DG-Gated model."""
    logging.info("\n" + "🔬" * 60)
    logging.info("🔬 ANALYZING THE DG-GATED MODEL")
    logging.info("🔬" * 60)
    
    model.eval()
    analysis_dir = os.path.join(save_dir, 'dg_gated_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Collect data
    all_gate_logits = []
    all_dg_outputs = []
    all_ca1_outputs = []
    all_task_labels = []
    routing_matrix = np.zeros((model.num_experts, model.num_experts))
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, labels in tqdm(test_loader, desc=f"Analyzing DG-Gated Task {task_id}"):
                inputs = inputs.to(device)
                
                # Get DG-based gating decisions
                _, gate_logits, analysis_data = model(inputs)
                predicted_experts = analysis_data['chosen_experts']
                
                for pred_expert in predicted_experts:
                    routing_matrix[task_id, pred_expert.item()] += 1
                
                # Get representations
                features_flat = model.prepare_dg_input(inputs)
                dg_output, ca1_output = model.hippocampal_experts[task_id](features_flat)
                
                all_gate_logits.append(gate_logits)
                all_dg_outputs.append(dg_output)
                all_ca1_outputs.append(ca1_output)
                all_task_labels.extend([task_id] * inputs.size(0))
    
    all_gate_logits = torch.cat(all_gate_logits, dim=0)
    all_dg_outputs = torch.cat(all_dg_outputs, dim=0)
    all_ca1_outputs = torch.cat(all_ca1_outputs, dim=0)
    all_task_labels = np.array(all_task_labels)
    
    routing_matrix = routing_matrix / (routing_matrix.sum(axis=1, keepdims=True) + 1e-8)
    expert_utilization = np.bincount(all_gate_logits.detach().cpu().numpy().argmax(axis=1), minlength=model.num_experts) / len(all_gate_logits)

    create_dg_gated_visualizations(
        all_gate_logits.detach().cpu().numpy(), all_dg_outputs.detach().cpu().numpy(), all_ca1_outputs.detach().cpu().numpy(),
        all_task_labels, routing_matrix, expert_utilization, model.dg_prototypes.detach().cpu().numpy(), analysis_dir
    )
    
    return {
        'routing_matrix': routing_matrix,
        'expert_utilization': expert_utilization,
        'routing_accuracy': np.diag(routing_matrix).mean()
    }

def create_dg_gated_visualizations(gate_logits, dg_outputs, ca1_outputs, task_labels, 
                                 routing_matrix, expert_utilization, dg_prototypes, save_dir):
    """Create visualizations for the DG-Gated model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧠 DG-Gated Hippocampal MoE Analysis', fontsize=16, fontweight='bold')
    
    # 1. Routing Matrix
    sns.heatmap(routing_matrix, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,0], square=True)
    axes[0,0].set_title('🚪 DG-Gated Task→Expert Routing')
    axes[0,0].set_xlabel('Predicted Expert ID')
    axes[0,0].set_ylabel('True Task ID')
    
    # 2. DG Prototype Similarity
    proto_sim = np.dot(dg_prototypes, dg_prototypes.T) / (np.linalg.norm(dg_prototypes, axis=1, keepdims=True) * np.linalg.norm(dg_prototypes, axis=1, keepdims=True).T)
    sns.heatmap(proto_sim, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=axes[0,1], square=True)
    axes[0,1].set_title('🧠 DG Prototype Similarity')
    axes[0,1].set_xlabel('Prototype ID')
    axes[0,1].set_ylabel('Prototype ID')

    # 3. CA1 Representations t-SNE
    n_samples = min(2000, len(ca1_outputs))
    indices = np.random.choice(len(ca1_outputs), n_samples, replace=False)
    pca = PCA(n_components=50)
    ca1_pca = pca.fit_transform(ca1_outputs[indices])
    tsne = TSNE(n_components=2, random_state=42)
    ca1_tsne = tsne.fit_transform(ca1_pca)
    axes[0,2].scatter(ca1_tsne[:, 0], ca1_tsne[:, 1], c=task_labels[indices], cmap='tab10', alpha=0.6, s=10)
    axes[0,2].set_title('🗺️ CA1 Features (t-SNE)')
    axes[0,2].set_xlabel('t-SNE 1')
    axes[0,2].set_ylabel('t-SNE 2')

    # 4. DG Representations t-SNE
    dg_pca = pca.fit_transform(dg_outputs[indices])
    dg_tsne = tsne.fit_transform(dg_pca)
    axes[1,0].scatter(dg_tsne[:, 0], dg_tsne[:, 1], c=task_labels[indices], cmap='tab10', alpha=0.6, s=10)
    axes[1,0].set_title('🧬 DG Features (t-SNE)')
    axes[1,0].set_xlabel('t-SNE 1')
    axes[1,0].set_ylabel('t-SNE 2')

    # 5. Gating Confidence (Similarity Scores)
    max_confidences = gate_logits.max(axis=1)
    axes[1,1].hist(max_confidences, bins=30, alpha=0.7, color='purple')
    axes[1,1].set_title('🎲 Gating Confidence (Max Similarity)')
    axes[1,1].set_xlabel('Max Cosine Similarity')
    axes[1,1].set_ylabel('Density')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Key Insights Text
    routing_accuracy = np.diag(routing_matrix).mean()
    axes[1,2].text(0.1, 0.9, f'📊 DG-Gated Model Analysis', fontsize=14, fontweight='bold', transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.8, f'Routing Accuracy: {routing_accuracy:.1%}', transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.7, f'Prototype Distinctness: {1.0 - np.mean(proto_sim[np.triu_indices(proto_sim.shape[0], k=1)]):.3f}', transform=axes[1,2].transAxes)
    status_color = 'green' if routing_accuracy > 0.6 else 'red'
    axes[1,2].text(0.1, 0.5, '✅ GOOD ROUTING' if routing_accuracy > 0.6 else '❌ POOR ROUTING', 
                  color=status_color, fontweight='bold', transform=axes[1,2].transAxes)
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'DG_Gated_Model_Analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📊 DG-Gated model analysis saved!")

def analyze_dg_deep_dive(model, test_loaders, device, save_dir):
    """Analyze the DG layer separation and sparsity in more detail."""
    logging.info("\n" + "🔍" * 60)
    logging.info("🔍 DEEP-DIVE ANALYSIS OF DG REPRESENTATIONS")
    logging.info("🔍" * 60)
    
    model.eval()
    
    # --- Setup Forward Hooks to Capture SparseActivation Outputs ---
    sparse_activation_outputs = [[] for _ in range(model.num_experts)]
    
    def get_sparse_activation_hook(expert_id):
        def hook(module, input, output):
            sparse_activation_outputs[expert_id].append(output.detach())
        return hook
    
    # Register hooks on SparseActivation layers
    hooks = []
    for expert_id in range(model.num_experts):
        # Find the SparseActivation layer in the DG expert
        for name, module in model.hippocampal_experts[expert_id].dg.named_modules():
            if isinstance(module, SparseActivation):
                hook = module.register_forward_hook(get_sparse_activation_hook(expert_id))
                hooks.append(hook)
                break
    
    # --- Data Collection ---
    # Store DG outputs for each task's data, processed by the correct expert
    dg_outputs_per_task = [[] for _ in range(model.num_experts)]
    # Store similarity scores of each task's data against all prototypes
    similarity_profiles = [[] for _ in range(model.num_experts)]

    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in tqdm(test_loader, desc=f"Deep Analyzing Task {task_id}"):
                inputs = inputs.to(device)
                features = model.prepare_dg_input(inputs)
                
                # Get DG output from the correct expert for this task's data
                dg_output, _ = model.hippocampal_experts[task_id](features)
                dg_outputs_per_task[task_id].append(dg_output)

                # Get similarity of this data's DG pattern to ALL expert prototypes
                all_dg_for_input = torch.stack([model.hippocampal_experts[i](features)[0] for i in range(model.num_experts)], dim=1)
                all_dg_norm = F.normalize(all_dg_for_input, p=2, dim=2)
                proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                sims = torch.einsum('bne,ne->bn', all_dg_norm, proto_norm)
                similarity_profiles[task_id].append(sims)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    dg_outputs_per_task = [torch.cat(outputs, dim=0) for outputs in dg_outputs_per_task]
    sparse_activation_outputs = [torch.cat(outputs, dim=0) for outputs in sparse_activation_outputs]
    similarity_profiles = [torch.cat(profs, dim=0) for profs in similarity_profiles]

    # --- Metric Calculation ---
    # 1. TRUE Sparsity per expert (measured at SparseActivation layer)
    expert_sparsity = [(d > 0).float().mean().item() for d in sparse_activation_outputs]
    
    # 2. Final DG output sparsity (for comparison)
    final_dg_sparsity = [(d > 0).float().mean().item() for d in dg_outputs_per_task]

    # Log the sparsity comparison
    logging.info("\n" + "🔍" * 60)
    logging.info("🔍 SPARSITY ANALYSIS RESULTS")
    logging.info("🔍" * 60)
    for i in range(model.num_experts):
        logging.info(f"Expert {i}: SparseActivation={expert_sparsity[i]:.1%}, Final DG={final_dg_sparsity[i]:.1%}")
    
    avg_sparse_activation = np.mean(expert_sparsity)
    avg_final_dg = np.mean(final_dg_sparsity)
    logging.info(f"Average SparseActivation sparsity: {avg_sparse_activation:.1%}")
    logging.info(f"Average Final DG sparsity: {avg_final_dg:.1%}")
    logging.info("🔍" * 60)

    # 2. Pattern Separation (Sparse Pattern Overlap Analysis)
    separation_data = []
    n_samples = 1000  # Number of samples to analyze

    for i in range(model.num_experts):
        # Get sparse activation patterns (binary masks)
        sparse_i = (dg_outputs_per_task[i] > 0).float()
        
        # Intra-task overlap (Jaccard similarity of active neurons)
        if len(sparse_i) > 1:
            indices1 = torch.randint(0, len(sparse_i), (n_samples,))
            indices2 = torch.randint(0, len(sparse_i), (n_samples,))
            # Ensure we don't compare a pattern to itself
            mask = indices1 != indices2
            indices1, indices2 = indices1[mask], indices2[mask]
            
            # Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
            pattern1 = sparse_i[indices1]
            pattern2 = sparse_i[indices2]
            intersection = (pattern1 * pattern2).sum(dim=1)
            union = (pattern1 + pattern2).clamp(0, 1).sum(dim=1)
            jaccard_sims = intersection / (union + 1e-8)  # Avoid division by zero
            for s in jaccard_sims.detach().cpu().numpy():
                separation_data.append({'expert': f'Expert {i}', 'type': 'Intra-Task', 'similarity': s})

        # Inter-task overlap between expert i and others
        for j in range(i + 1, model.num_experts):
            sparse_j = (dg_outputs_per_task[j] > 0).float()
            
            # Sample patterns from both experts
            sample_i = sparse_i[torch.randperm(len(sparse_i))[:n_samples//2]]
            sample_j = sparse_j[torch.randperm(len(sparse_j))[:n_samples//2]]
            
            # Compute Jaccard similarity between patterns from different experts
            intersection = (sample_i.unsqueeze(1) * sample_j.unsqueeze(0)).sum(dim=2)
            union = (sample_i.unsqueeze(1) + sample_j.unsqueeze(0)).clamp(0, 1).sum(dim=2)
            jaccard_sims = intersection / (union + 1e-8)
            
            for s in jaccard_sims.flatten().detach().cpu().numpy():
                separation_data.append({'expert': f'Expert {i}', 'type': 'Inter-Task', 'similarity': s})

    # --- Visualization ---
    create_dg_deep_dive_visualizations(
        expert_sparsity,
        final_dg_sparsity,
        separation_data,
        similarity_profiles,
        save_dir
    )

def create_dg_deep_dive_visualizations(expert_sparsity, final_dg_sparsity, separation_data, similarity_profiles, save_dir):
    """Create advanced visualizations for DG analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🔍 Dentate Gyrus (DG) Deep-Dive Analysis', fontsize=16, fontweight='bold')
    
    # 1. TRUE Sparsity per expert (measured at SparseActivation layer)
    num_experts = len(expert_sparsity)
    axes[0,0].bar(range(num_experts), expert_sparsity, color='skyblue')
    axes[0,0].set_title('🧬 TRUE DG Sparsity (SparseActivation Layer)')
    axes[0,0].set_xlabel('Expert ID')
    axes[0,0].set_ylabel('Sparsity (Fraction Active)')
    axes[0,0].axhline(y=0.05, color='red', linestyle='--', label='Biological Target (5%)')
    axes[0,0].axhline(y=0.10, color='orange', linestyle='--', label='Target (10%)')
    axes[0,0].set_xticks(range(num_experts))
    axes[0,0].legend()
    for i, v in enumerate(expert_sparsity):
        axes[0,0].text(i, v + 0.005, f"{v:.1%}", ha='center', va='bottom')

    # 2. Final DG Output Sparsity (for comparison)
    axes[0,1].bar(range(num_experts), final_dg_sparsity, color='lightgreen')
    axes[0,1].set_title('🧬 Final DG Output Sparsity (After Processing)')
    axes[0,1].set_xlabel('Expert ID')
    axes[0,1].set_ylabel('Sparsity (Fraction Active)')
    axes[0,1].axhline(y=0.05, color='red', linestyle='--', label='Biological Target (5%)')
    axes[0,1].set_xticks(range(num_experts))
    axes[0,1].legend()
    for i, v in enumerate(final_dg_sparsity):
        axes[0,1].text(i, v + 0.005, f"{v:.1%}", ha='center', va='bottom')

    # 3. Pattern Separation (Intra- vs. Inter-Task Similarity)
    import pandas as pd
    df_sep = pd.DataFrame(separation_data)
    sns.violinplot(data=df_sep, x='expert', y='similarity', hue='type', split=True, inner='quartile', ax=axes[1,0])
    axes[1,0].set_title('🧩 Sparse Pattern Overlap Analysis (Jaccard)')
    axes[1,0].set_xlabel('Expert')
    axes[1,0].set_ylabel('Jaccard Similarity (Overlap)')
    axes[1,0].legend(title='Similarity Type')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Gating Decision Profile
    df_profiles = []
    for task_id, profs in enumerate(similarity_profiles):
        for expert_id in range(num_experts):
            for val in profs[:, expert_id]:
                df_profiles.append({
                    'true_task': f'Task {task_id}',
                    'prototype_id': f'Proto {expert_id}',
                    'similarity': val.item()
                })
    df_profiles = pd.DataFrame(df_profiles)
    sns.boxplot(data=df_profiles, x='true_task', y='similarity', hue='prototype_id', ax=axes[1,1])
    axes[1,1].set_title('🚦 Gating Decision Profile')
    axes[1,1].set_xlabel('True Input Task')
    axes[1,1].set_ylabel('Similarity to Prototype')
    axes[1,1].legend(title='Prototype', bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'DG_Deep_Dive_Analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"📊 DG deep-dive analysis saved!")

def calculate_global_contrastive_loss(dg_outputs_all_experts, task_ids, prototypes, margin):
    """
    Enhanced global contrastive loss with expert-to-expert comparison.
    PULLS the DG output from the CORRECT expert towards the correct prototype.
    PUSHES the DG outputs from ALL INCORRECT experts away from their own prototypes.
    ADDITIONALLY compares each expert's DG output to ALL other experts' outputs.
    
    Args:
    - dg_outputs_all_experts (list or torch.Tensor): A list of DG outputs from each expert,
      or a tensor of shape (batch_size, num_experts, dg_dim).
    - task_ids (torch.Tensor): The true task ID for each sample in the batch.
    - prototypes (torch.Tensor): The prototype vectors for all experts, shape (num_experts, dg_dim).
    - margin (float): The margin for the contrastive loss.
    """
    if not isinstance(dg_outputs_all_experts, torch.Tensor):
        dg_outputs_all_experts = torch.stack(dg_outputs_all_experts, dim=1)

    # Normalize for cosine similarity
    dg_outputs_norm = F.normalize(dg_outputs_all_experts, p=2, dim=2)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)

    # === PROTOTYPE-BASED CONTRASTIVE LOSS ===
    # Einsum for batched dot product: (Batch, Experts, Dims) x (Experts, Dims) -> (Batch, Experts)
    # This gives similarity of each expert's output to its *own* prototype
    similarities = torch.einsum('bed,ed->be', dg_outputs_norm, prototypes_norm)

    # For each sample in the batch, find the similarity to the *correct* prototype
    correct_expert_sims = similarities.gather(1, task_ids.unsqueeze(1)).squeeze()

    # PULL Loss: encourages the correct expert's output to be similar to its prototype
    pull_loss = (1 - correct_expert_sims).mean()

    # PUSH Loss: encourages incorrect experts' outputs to be dissimilar to their prototypes
    # Create a mask to zero out the correct expert's similarity for each sample
    mask = torch.ones_like(similarities)
    mask.scatter_(1, task_ids.unsqueeze(1), 0)
    
    # Use the mask to calculate push loss only on incorrect experts
    hinge_loss = F.relu(similarities - margin) * mask
    
    # Avoid division by zero if a batch has only one task
    num_other_experts = similarities.size(1) - 1
    if num_other_experts > 0:
        push_loss = hinge_loss.sum() / mask.sum()
    else:
        push_loss = torch.tensor(0.0, device=similarities.device)

    prototype_loss = pull_loss + push_loss

    # === EXPERT-TO-EXPERT CONTRASTIVE LOSS ===
    # Compare each expert's DG output to ALL other experts' outputs for the same sample
    batch_size, num_experts, dg_dim = dg_outputs_norm.shape
    expert_to_expert_loss = 0.0
    
    for i in range(batch_size):
        task_id = task_ids[i].item()
        current_expert_dg = dg_outputs_norm[i, task_id]  # Correct expert's DG output
        
        # Compare to all other experts' DG outputs for this sample
        for expert_id in range(num_experts):
            if expert_id != task_id:
                other_expert_dg = dg_outputs_norm[i, expert_id]
                # Calculate cosine similarity between current and other expert
                sim = F.cosine_similarity(current_expert_dg, other_expert_dg, dim=0)
                # Push them apart with stronger margin
                expert_to_expert_loss += F.relu(sim + margin * 1.5)  # Stronger margin for expert-to-expert
    
    expert_to_expert_loss = expert_to_expert_loss / (batch_size * (num_experts - 1)) if num_experts > 1 else 0.0

    # Combine both losses with higher weight on expert-to-expert
    total_loss = prototype_loss + 3.0 * expert_to_expert_loss
    
    return total_loss

def create_balanced_loader(task_loaders, batches_per_epoch=200):
    """Creates a generator that yields balanced batches from all task loaders."""
    num_tasks = len(task_loaders)
    task_iters = [iter(loader) for loader in task_loaders]
    
    for _ in range(batches_per_epoch):
        inputs_batch, labels_batch, tasks_batch = [], [], []
        
        # Sample a mini-batch from each task
        for task_id in range(num_tasks):
            try:
                inputs, labels = next(task_iters[task_id])
            except StopIteration:
                task_iters[task_id] = iter(task_loaders[task_id])
                inputs, labels = next(task_iters[task_id])
            
            inputs_batch.append(inputs)
            labels_batch.append(labels)
            tasks_batch.extend([task_id] * len(inputs))
        
        yield torch.cat(inputs_batch), torch.cat(labels_batch), torch.tensor(tasks_batch)

def phase2_contrastive_tuning(model, train_loaders, device, args, log_dir):
    """
    Phase 2: Jointly fine-tune all DG expert layers using a contrastive loss
    to enforce pattern separation. Raw inputs are passed directly to DG; no feature extractor is used.
    Uses ONLY replay buffer data to prevent forgetting.
    Adds global decorrelation loss on DG prototypes.
    """
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: JOINT CONTRASTIVE FINE-TUNING OF DG EXPERTS (REPLAY ONLY)")
    logger.info("="*80)

    # --- Setup ---
    # Freeze all layers except for the DG layers within the hippocampal experts
    for name, p in model.named_parameters():
        if 'hippocampal_experts' in name and 'dg' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.contrastive_lr)
    
    logger.info(f"  🥶 CA3 and output layers frozen; no feature extractor is used.")
    logger.info(f"  🔥 Surgically tuning {sum(p.numel() for p in trainable_params):,} parameters in DG layers only.")
    logger.info(f"  📚 Using ONLY replay buffer data to prevent forgetting.")
    
    # --- Training Loop ---
    for epoch in range(args.contrastive_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        # Create replay-based balanced loader for each epoch
        replay_loader = create_replay_balanced_loader(model, args.batch_size)
        progress_bar = tqdm(replay_loader, total=200, desc=f"Contrastive Epoch {epoch+1}/{args.contrastive_epochs}")

        for inputs, labels, task_ids in progress_bar:
            inputs, labels, task_ids = inputs.to(device), labels.to(device), task_ids.to(device)
            
            optimizer.zero_grad()
            
            # Prepare raw flattened inputs for DG
            features = model.prepare_dg_input(inputs)
            
            # Get DG outputs from all experts
            all_dg_outputs = [expert(features)[0] for expert in model.hippocampal_experts]
            
            # Calculate the global contrastive loss
            loss = calculate_global_contrastive_loss(
                all_dg_outputs, task_ids, model.dg_prototypes, margin=args.contrastive_margin
            )
            
            # Calculate contrastive diagnostics for logging
            all_dg_stack = torch.stack(all_dg_outputs, dim=1)
            pull_sim, pull_dist, pull_viol, push_active = calculate_contrastive_diagnostics(
                all_dg_stack, task_ids, model.dg_prototypes, margin=args.contrastive_margin
            )
            
            # --- Global decorrelation loss on prototypes ---
            prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1)
            sim_matrix = prototypes_norm @ prototypes_norm.T
            num_experts = sim_matrix.size(0)
            off_diag = sim_matrix - torch.eye(num_experts, device=sim_matrix.device)
            decorrelation_loss = (off_diag ** 2).sum() / (num_experts * (num_experts - 1))
            loss = loss + 0.3 * decorrelation_loss  # Reduced weight to prevent over-separation
            
            # --- Expert balancing loss ---
            # Get gating logits for the current batch
            all_dg_norm = F.normalize(torch.stack(all_dg_outputs, dim=1), p=2, dim=2)
            gate_logits = torch.einsum('bne,ne->bn', all_dg_norm, prototypes_norm)
            balancing_loss = calculate_expert_balancing_loss(gate_logits)
            loss = loss + 0.2 * balancing_loss  # Increased weight for expert balancing
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar with contrastive diagnostics
            postfix = {
                'loss': f'{total_loss/total_batches:.4f}',
                'pull_sim': f'{pull_sim:.4f}',
                'pull_dist': f'{pull_dist:.4f}',
                'pull_viol': f'{pull_viol:.4f}',
                'push_active': f'{push_active:.4f}'
            }
            progress_bar.set_postfix(postfix)
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logger.info(f"  Epoch {epoch+1} | Average Contrastive Loss: {avg_loss:.4f}")
        logger.info(f"[Epoch {epoch+1}] Pull sim: {pull_sim:.4f}, Pull dist: {pull_dist:.4f}, Pull violations: {pull_viol:.4f}, Push active: {push_active:.4f}")
        
        # Additional diagnostic logs for Phase 2
        # Expert utilization stats
        util_stats = calculate_expert_utilization_stats(gate_logits)
        logger.info(f"[Epoch {epoch+1}] Expert Util - Mean: {util_stats['mean']:.3f}, Std: {util_stats['std']:.3f}, Min: {util_stats['min']:.3f}, Max: {util_stats['max']:.3f}")
        
        # Gating confidence stats
        gating_stats = calculate_gating_confidence_stats(gate_logits)
        logger.info(f"[Epoch {epoch+1}] Gating - Confidence: {gating_stats['mean_confidence']:.3f}±{gating_stats['std_confidence']:.3f}, Entropy: {gating_stats['mean_entropy']:.3f}±{gating_stats['std_entropy']:.3f}")
        
        # Prototype similarity stats
        proto_stats = calculate_prototype_similarity_stats(model.dg_prototypes)
        logger.info(f"[Epoch {epoch+1}] Prototypes - Off-diag sim: {proto_stats['mean_off_diag']:.3f}±{proto_stats['std_off_diag']:.3f}, Range: [{proto_stats['min_off_diag']:.3f}, {proto_stats['max_off_diag']:.3f}]")
        
        # Memory stats
        memory_stats = calculate_memory_stats(model)
        logger.info(f"[Epoch {epoch+1}] Memory - Total replay: {memory_stats['total_replay_samples']}, Distribution: {memory_stats['replay_distribution']}")
        
        # Gradient norms (if available)
        grad_norm, param_count = calculate_gradient_norms(model)
        if param_count > 0:
            logger.info(f"[Epoch {epoch+1}] Grad norm: {grad_norm:.4f} (over {param_count} params)")
        # === DIAGNOSTICS: log expert utilization and active loss fraction ===
        if epoch == 0 or (epoch+1) % 1 == 0:  # every epoch
            log_expert_utilization(gate_logits, epoch+1, log_dir)
            log_active_loss_fraction(torch.stack(all_dg_outputs, dim=1), task_ids, model.dg_prototypes,log_dir, args.contrastive_margin, epoch+1)
        # === DIAGNOSTICS: plot contrastive similarity histogram ===
        plot_contrastive_similarity_histogram(torch.stack(all_dg_outputs, dim=1), task_ids, model.dg_prototypes, epoch+1, log_dir)
    # === DIAGNOSTICS: plot prototype distance matrix after Phase 2 ===
    plot_prototype_distance_matrix(model.dg_prototypes, log_dir)
    # === DIAGNOSTICS: t-SNE of DG representations after Phase 2 ===
    # Use a batch from the first test loader for t-SNE
    test_loader = train_loaders[0]
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    features = model.prepare_dg_input(inputs)
    all_dg = []
    routed_experts = []
    for i in range(model.num_experts):
        dg_output, _ = model.hippocampal_experts[i](features)
        all_dg.append(dg_output.detach().cpu().numpy())
    all_dg = np.concatenate(all_dg, axis=0)
    routed_experts = np.tile(np.arange(model.num_experts), len(inputs))
    plot_tsne_dg(all_dg, np.repeat(labels.detach().cpu().numpy(), model.num_experts), routed_experts, log_dir, tag="after_phase2")

def create_replay_balanced_loader(model, batch_size, batches_per_epoch=200):
    """
    Create a balanced loader using only replay buffer data.
    This ensures Phase 2 training doesn't cause forgetting.
    Now samples class-balanced batches from the replay buffer.
    """
    # Collect all replay buffer data as (input, label, task_id)
    all_replay_data = []
    for task_id in range(model.num_experts):
        if model.replay_buffer[task_id]:
            for inputs, labels in model.replay_buffer[task_id]:
                all_replay_data.append((inputs, labels, task_id))
    if not all_replay_data:
        logging.warning("⚠️ No replay buffer data available! Using empty loader.")
        return []
    logging.info(f"📚 Using {len(all_replay_data)} replay samples for Phase 2 training.")
    # Organize by class for balanced sampling
    class_to_samples = defaultdict(list)
    for inp, lbl, tid in all_replay_data:
        class_to_samples[lbl.item()].append((inp, lbl, tid))
    all_classes = list(class_to_samples.keys())
    batches = []
    for _ in range(batches_per_epoch):
        # Sample batch_size//num_classes from each class
        per_class = max(1, batch_size // max(1, len(all_classes)))
        batch_data = []
        for c in all_classes:
            samples = class_to_samples[c]
            if len(samples) >= per_class:
                batch_data.extend(random.sample(samples, per_class))
            else:
                batch_data.extend(random.choices(samples, k=per_class))
        # If not enough, pad with random
        if len(batch_data) < batch_size:
            extra = random.sample(all_replay_data, batch_size - len(batch_data))
            batch_data.extend(extra)
        batch_data = batch_data[:batch_size]
        inputs = torch.stack([data[0] for data in batch_data])
        labels = torch.stack([data[1] for data in batch_data])
        task_ids = torch.tensor([data[2] for data in batch_data])
        batches.append((inputs, labels, task_ids))
    return batches

def calculate_class_balanced_weights(class_accuracies, epsilon=0.1, smoothing=0.1):
    """
    Calculate class-balanced weights based on per-class accuracy.
    
    Args:
        class_accuracies: Dict mapping class_id to accuracy (0-1)
        epsilon: Small constant to prevent division by zero
        smoothing: Smoothing factor to prevent extreme weights
    
    Returns:
        Dict mapping class_id to weight
    """
    weights = {}
    
    # Compute inverse accuracy weights with smoothing
    for class_id, acc in class_accuracies.items():
        # Add smoothing to prevent extreme weights
        smoothed_acc = acc * (1 - smoothing) + smoothing * 0.5  # Smooth towards 0.5
        # Inverse accuracy weight with epsilon to prevent division by zero
        weight = 1.0 / (smoothed_acc + epsilon)
        weights[class_id] = weight
    
    # Normalize weights so their mean is 1.0
    mean_weight = sum(weights.values()) / len(weights)
    normalized_weights = {class_id: weight / mean_weight for class_id, weight in weights.items()}
    
    return normalized_weights

def calculate_prototype_regularization_loss(model, inputs, labels, expert_id, device):
    """
    Calculate prototype regularization loss to keep DG prototypes close to their assigned class centers.
    
    Args:
        model: The DG-Gated model
        inputs: Input batch
        labels: Global class labels
        expert_id: Current expert being trained
        device: Device to compute on
    
    Returns:
        Prototype regularization loss
    """
    # Get features and DG outputs for current batch
    features = model.prepare_dg_input(inputs)
    dg_output, _ = model.hippocampal_experts[expert_id](features)
    
    # Get the classes assigned to this expert
    task_classes = model.task_classes[expert_id]
    
    # Compute class centers for this expert's classes
    class_centers = {}
    for class_id in task_classes:
        # Find samples belonging to this class
        class_mask = (labels == class_id)
        if class_mask.sum() > 0:
            # Compute mean DG output for this class
            class_dg_outputs = dg_output[class_mask]
            class_centers[class_id] = class_dg_outputs.mean(dim=0)
    
    # If no class centers found, return zero loss
    if not class_centers:
        return torch.tensor(0.0, device=device)
    
    # Compute prototype regularization loss
    prototype_loss = 0.0
    for class_id, class_center in class_centers.items():
        # Get the current prototype for this expert (if computed)
        if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0 and model.prototypes_computed:
            current_prototype = model.dg_prototypes[expert_id]
            # L2 distance between class center and prototype
            prototype_loss += F.mse_loss(class_center, current_prototype)
        else:
            # During training, use a milder regularization to prevent collapse
            # Instead of forcing all outputs to be identical, just encourage some consistency
            class_mask = (labels == class_id)
            if class_mask.sum() > 1:  # Need at least 2 samples
                class_dg_outputs = dg_output[class_mask]
                # Use a small penalty for excessive variance, but don't force collapse
                class_variance = torch.var(class_dg_outputs, dim=0).mean()
                # Only penalize if variance is very high (encourage some consistency without forcing collapse)
                if class_variance > 0.5:  # Threshold to prevent collapse
                    prototype_loss += 0.1 * class_variance
    
    return prototype_loss

def calculate_routing_confidence_penalty(model, inputs, device, expert_id=None):
    """
    Calculate routing confidence penalty to encourage more decisive gating.
    
    Args:
        model: The DG-Gated model
        inputs: Input batch
        device: Device to compute on
        expert_id: Current expert being trained (for training mode)
    
    Returns:
        Routing confidence penalty loss
    """
    # Get features and compute DG outputs from all experts
    features = model.prepare_dg_input(inputs)
    
    all_dg_outputs = []
    for i in range(model.num_experts):
        dg_output, _ = model.hippocampal_experts[i](features)
        all_dg_outputs.append(dg_output)
    
    all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [batch, num_experts, dg_dim]
    all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
    
    # Compute similarities to prototypes if available
    if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0:
        prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
        # Calculate similarity scores
        similarities = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
        
        # Apply temperature scaling
        gate_logits = similarities / model.gating_temperature
        
        # Convert to probabilities
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # Calculate entropy penalty (encourage low entropy = more decisive routing)
        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
        confidence_penalty = entropy.mean()
        
        return confidence_penalty
    else:
        # During training without prototypes, use DG output variance as penalty
        # This encourages each expert to have distinct DG representations
        if expert_id is not None:
            # Get current expert's DG output
            current_dg = all_dg_outputs[:, expert_id]  # [batch, dg_dim]
            
            # Calculate variance across batch (encourage consistent DG patterns)
            dg_variance = torch.var(current_dg, dim=0).mean()
            
            # Also encourage separation from other experts
            other_experts = [i for i in range(model.num_experts) if i != expert_id]
            if other_experts:
                other_dg = all_dg_outputs[:, other_experts]  # [batch, num_other_experts, dg_dim]
                
                # Calculate cosine similarity between current and other experts
                current_dg_norm = F.normalize(current_dg, p=2, dim=1)
                other_dg_norm = F.normalize(other_dg.view(-1, other_dg.size(-1)), p=2, dim=1)
                
                # Reshape for batch computation
                current_dg_expanded = current_dg_norm.unsqueeze(1).expand(-1, len(other_experts), -1)
                current_dg_expanded = current_dg_expanded.reshape(-1, current_dg.size(-1))
                
                similarities = F.cosine_similarity(current_dg_expanded, other_dg_norm, dim=1)
                separation_penalty = similarities.mean()
                
                # Combine variance and separation penalties
                confidence_penalty = dg_variance + separation_penalty
            else:
                confidence_penalty = dg_variance
                
            return confidence_penalty
        else:
            # If no expert_id provided, return zero penalty
            return torch.tensor(0.0, device=device)

def phase1_train_experts_sequentially(model, train_loaders, test_loaders, device, args,log_dir):
    """
    MODIFIED Phase 1: Train experts independently with distillation from previous experts.
    The goal is to learn good initial weights for each expert while preserving knowledge.
    """
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: TRAINING HIPPOCAMPAL EXPERTS WITH DISTILLATION")
    logger.info("="*80)

    # Use a new hyperparameter for the contrastive loss coefficient
    contrastive_coef = getattr(args, 'contrastive_loss_coef', 1.0)
    contrastive_margin = getattr(args, 'contrastive_margin', 0.5)

    # Handle optional label smoothing (default to 0.0 if not specified)
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    expert_results = []
    
    # Store previous experts' outputs for distillation
    previous_experts_outputs = []
    
    # Initialize class-balanced loss tracking
    class_weights = None
    if args.use_class_balanced_loss:
        logger.info("🔧 Using class-balanced loss weighting")
        # Initialize with equal weights for all classes
        all_classes = list(range(args.num_experts * args.classes_per_task))
        class_weights = {class_id: 1.0 for class_id in all_classes}
    
    # Initialize prototype regularization
    if args.use_prototype_regularization:
        logger.info("🔧 Using prototype regularization to anchor DG prototypes")
    
    # Initialize routing confidence penalty
    if args.use_routing_confidence_penalty:
        logger.info("🔧 Using routing confidence penalty to encourage decisive gating")
    
    # Initialize EWC
    if args.ewc_lambda > 0:
        logger.info(f"🔧 Using EWC with lambda={args.ewc_lambda} to prevent forgetting")

    for expert_id in range(1):
        train_loader = train_loaders[expert_id]
        test_loader = test_loaders[expert_id]
        # --- Freeze/Unfreeze Parameters ---
        # Unfreeze current expert and shared layers, freeze others
        for name, p in model.named_parameters():
            is_current_expert = f"hippocampal_experts.{expert_id}" in name or f"output_layers.{expert_id}" in name
            is_shared_component = "ca1_integration" in name
            p.requires_grad = is_current_expert or is_shared_component
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = args.expert_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.expert_epochs)
        max_steps = args.expert_epochs * len(train_loader)
        step_count = 0
        
        # Early stopping setup
        patience = args.early_stopping_patience
        best_acc = 0.0
        patience_counter = 0

        logger.info(f"\nTraining Expert {expert_id}...")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        if expert_id > 0:
            logger.info(f"  Using distillation from {expert_id} previous expert(s)")

        best_acc = 0.0

        # Create a mapping from global CIFAR labels to local task labels (0, 1, ...)
        task_class_list = model.task_classes[expert_id]
        global_to_local_map = {global_class: local_idx for local_idx, global_class in enumerate(task_class_list)}
        
        for epoch in range(args.expert_epochs):
            model.train()
            total_loss = 0
            total_distillation_loss = 0
            correct = 0
            total = 0
            total_ce_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}/{args.expert_epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Convert global labels to local labels for this task
                local_labels = torch.tensor([global_to_local_map[l.item()] for l in labels], dtype=torch.long, device=device)
                
                # Add current batch to replay buffer before creating loss graph
                model.add_to_replay_buffer(inputs.detach(), labels.detach(), expert_id)

                optimizer.zero_grad()
                
                # --- ONLINE EMA PROTOTYPE UPDATE ---
                # Update class prototypes using EMA for current expert's classes
                features = model.prepare_dg_input(inputs)
                dg_output, _ = model.hippocampal_experts[expert_id](features)
                model.update_class_prototype_ema(dg_output.detach(), labels, expert_id)
                
                # --- Loss for the current task ---
                outputs, _, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                # Apply class-balanced loss if enabled
                if args.use_class_balanced_loss and class_weights is not None:
                    # Get global class IDs for this batch
                    global_class_ids = labels.detach().cpu().numpy()
                    # Get weights for this batch
                    batch_weights = torch.tensor([class_weights[class_id] for class_id in global_class_ids], 
                                               dtype=torch.float32, device=device)
                    # Apply weighted cross-entropy
                    log_probs = F.log_softmax(task_outputs, dim=1)
                    targets = torch.zeros_like(log_probs).scatter_(1, local_labels.unsqueeze(1), 1.0)
                    if label_smoothing > 0:
                        targets = targets * (1 - label_smoothing) + label_smoothing / args.classes_per_task
                    classification_loss = -(targets * log_probs).sum(dim=1)
                    classification_loss = (classification_loss * batch_weights).mean()
                else:
                    classification_loss = criterion(task_outputs, local_labels)

                # Initialize total loss
                loss = classification_loss
                
                # --- PUSH-PULL CONTRASTIVE LOSS (Phase 1) ---
                push_pull_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply after first expert
                    # Debug: Check if we're entering push-pull section for Expert 2
                    if expert_id == 2 and progress_bar.n == len(train_loader) - 1:
                        logger.info(f"DEBUG: Entering push-pull section for Expert {expert_id}, Epoch {epoch+1}, Batch {progress_bar.n+1}/{len(train_loader)}")
                    # Get features and DG outputs from all experts
                    features = model.prepare_dg_input(inputs)
                    all_dg_outputs = []
                    for i in range(model.num_experts):
                        dg_output, _ = model.hippocampal_experts[i](features)
                        all_dg_outputs.append(dg_output)
                    
                    # Create task IDs for this batch (all samples belong to current expert)
                    batch_task_ids = torch.full((inputs.size(0),), expert_id, device=device)
                    
                    # Use a simple prototype-based contrastive loss for Phase 1
                    # This encourages the current expert to be different from previous experts
                    current_expert_dg = all_dg_outputs[expert_id]
                    current_expert_dg_norm = F.normalize(current_expert_dg, p=2, dim=1)
                    
                    # Compare to previous experts' DG outputs
                    for prev_expert_id in range(expert_id):
                        prev_expert_dg = all_dg_outputs[prev_expert_id]
                        prev_expert_dg_norm = F.normalize(prev_expert_dg, p=2, dim=1)
                        
                        # Calculate cosine similarity between current and previous expert DG outputs
                        sim = F.cosine_similarity(current_expert_dg_norm, prev_expert_dg_norm, dim=1)
                        # Push them apart (minimize similarity) - sim is already a tensor of shape [batch_size]
                        push_pull_loss += F.relu(sim + 0.05).mean()  # Take mean over batch
                    
                    push_pull_loss = push_pull_loss / expert_id  # Average over previous experts
                    loss += 0.2 * push_pull_loss  # Reduced push-pull weight for better task learning
                    
                    # Calculate contrastive diagnostics for logging (only at end of epoch)
                    if progress_bar.n == len(train_loader) - 1:  # Log every epoch, only at end of epoch
                        logger.info(f"DEBUG: Generating post-epoch logs for Expert {expert_id}, Epoch {epoch+1}")
                        
                        # Get active prototypes for diagnostics
                        active_protos, active_E = model.get_active_prototypes_for_diagnostics(expert_id)
                        
                        if active_protos is not None and active_E > 0:
                            # Check if we have enough prototype counts for meaningful diagnostics
                            expert_classes = model.task_classes[expert_id]
                            min_counts = 5  # Minimum counts per class for stable prototypes
                            has_enough_counts = False
                            if model.class_proto_counts is not None:
                                class_counts = model.class_proto_counts[expert_classes]
                                has_enough_counts = (class_counts >= min_counts).any()
                            
                            if has_enough_counts:
                                # Collect DG outputs for active experts only
                                active_dg_list = []
                                for e in range(active_E):
                                    dg_e, _ = model.hippocampal_experts[e](features)
                                    active_dg_list.append(dg_e)
                                active_dg_stack = torch.stack(active_dg_list, dim=1)  # [B, active_E, dg_dim]
                                
                                # Use active prototypes and experts
                                pull_sim, pull_dist, pull_viol, push_active = calculate_contrastive_diagnostics(
                                    active_dg_stack, batch_task_ids, active_protos, margin=0.5, active_experts=active_E
                                )
                                logger.info(f"[Epoch {epoch+1}] Pull sim: {pull_sim:.4f}, Pull dist: {pull_dist:.4f}, Pull violations: {pull_viol:.4f}, Push active: {push_active:.4f}")
                            else:
                                logger.info(f"[Epoch {epoch+1}] Prototypes still warming up (min {min_counts} counts needed)")
                        else:
                            logger.info(f"[Epoch {epoch+1}] No active prototypes available for diagnostics")
                        
                        # Additional diagnostic logs for Phase 1
                        # DG sparsity stats
                        dg_sparsity_stats = calculate_dg_sparsity_stats(model, inputs, device)
                        current_expert_stats = dg_sparsity_stats[expert_id]
                        logger.info(f"[Epoch {epoch+1}] DG Sparsity - Target: {current_expert_stats['target_sparsity']:.3f}, Actual: {current_expert_stats['actual_sparsity']:.3f}, Error: {current_expert_stats['sparsity_error']:.3f}")
                        
                        # Memory stats
                        memory_stats = calculate_memory_stats(model)
                        logger.info(f"[Epoch {epoch+1}] Memory - Total replay: {memory_stats['total_replay_samples']}, Avg per task: {memory_stats['mean_replay_per_task']:.1f}")
                        
                        # Learning rate
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(f"[Epoch {epoch+1}] Learning rate: {current_lr:.2e}")
                
                # --- Prototype Regularization Loss ---
                prototype_reg_loss = torch.tensor(0.0, device=device)
                if args.use_prototype_regularization:
                    prototype_reg_loss = calculate_prototype_regularization_loss(
                        model, inputs, labels, expert_id, device
                    )
                    loss += args.prototype_reg_coef * prototype_reg_loss
                
                # --- Routing Confidence Penalty ---
                routing_confidence_loss = torch.tensor(0.0, device=device)
                if args.use_routing_confidence_penalty:
                    routing_confidence_loss = calculate_routing_confidence_penalty(
                        model, inputs, device, expert_id
                    )
                    loss += args.routing_confidence_coef * routing_confidence_loss
                
                total_ce_loss += classification_loss.item()
                
                # --- Distillation Loss from Previous Experts ---
                distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.distillation_coef > 0:
                    # Prepare raw flattened DG inputs from inputs
                    features = model.prepare_dg_input(inputs)
                    
                    # Get DG and CA1 outputs from all previous experts
                    previous_dg_outputs = []
                    previous_ca1_outputs = []
                    for prev_expert_id in range(expert_id):
                        with torch.no_grad():
                            prev_dg, prev_ca1 = model.hippocampal_experts[prev_expert_id](features)
                            previous_dg_outputs.append(prev_dg)
                            previous_ca1_outputs.append(prev_ca1)
                    
                    # Get current expert's DG and CA1 outputs
                    current_dg, current_ca1 = model.hippocampal_experts[expert_id](features)
                    
                    # Calculate feature-based distillation loss
                    distillation_loss = calculate_feature_distillation_loss(
                        current_dg, 
                        previous_dg_outputs, 
                        current_ca1, 
                        previous_ca1_outputs
                    )
                    loss += args.distillation_coef * distillation_loss
                
                # --- Raw-input-to-CA1 Distillation Loss (legacy option) ---
                feature_distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.feature_distillation_coef > 0:
                    # Prepare current raw flattened DG inputs
                    current_features = model.prepare_dg_input(inputs)
                    
                    # Get CA1 outputs from all previous experts as target features
                    previous_ca1_outputs = []
                    for prev_expert_id in range(expert_id):
                        with torch.no_grad():
                            _, prev_ca1 = model.hippocampal_experts[prev_expert_id](current_features)
                            previous_ca1_outputs.append(prev_ca1)
                    
                    if len(previous_ca1_outputs) > 0:
                        # Project current raw input vector to CA1 dimension for comparison
                        projected_features = model.feature_to_ca1(current_features)
                        # Normalize projected features
                        projected_features_norm = F.normalize(projected_features, p=2, dim=1)
                        
                        # Calculate average previous expert CA1 features
                        avg_previous_ca1 = torch.mean(torch.stack(previous_ca1_outputs), dim=0)
                        avg_previous_ca1_norm = F.normalize(avg_previous_ca1, p=2, dim=1)
                        
                        # Cosine similarity loss (maximize similarity)
                        similarity = F.cosine_similarity(projected_features_norm, avg_previous_ca1_norm, dim=1).mean()
                        feature_distillation_loss = 1.0 - similarity
                        
                        loss += args.feature_distillation_coef * feature_distillation_loss
                
                # --- EWC Loss to prevent forgetting ---
                ewc_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply for tasks after the first one
                    ewc_loss = model.calculate_ewc_loss(args.ewc_lambda)
                    loss += ewc_loss
                
                # --- Replay Loss from previous tasks ---
                if expert_id > 0 and args.replay_loss_coef > 0:
                    replay_loss = torch.tensor(0.0, device=device)
                    for prev_task_id in range(expert_id):
                        replay_inputs, replay_global_labels = model.sample_from_replay_buffer(prev_task_id, args.batch_size)
                        
                        if replay_inputs is not None:
                            replay_inputs = replay_inputs.to(device)
                            replay_global_labels = replay_global_labels.to(device)
                            
                            # We need a global to local map for the replayed task
                            prev_task_class_list = model.task_classes[prev_task_id]
                            prev_global_to_local_map = {global_class: local_idx for local_idx, global_class in enumerate(prev_task_class_list)}
                            replay_local_labels = torch.tensor([prev_global_to_local_map.get(l.item()) for l in replay_global_labels], dtype=torch.long, device=device)
                            
                            # Forward pass for the replayed task
                            replay_outputs, _, _ = model(replay_inputs, task_id=prev_task_id)
                            
                            # Get the specific logits for the replayed task
                            replay_start_idx = prev_task_id * model.classes_per_task
                            replay_end_idx = replay_start_idx + model.classes_per_task
                            replay_task_outputs = replay_outputs[:, replay_start_idx:replay_end_idx]
                            
                            # Apply class-balanced loss to replay if enabled
                            if args.use_class_balanced_loss and class_weights is not None:
                                # Get global class IDs for replay batch
                                replay_global_class_ids = replay_global_labels.detach().cpu().numpy()
                                # Get weights for replay batch
                                replay_batch_weights = torch.tensor([class_weights[class_id] for class_id in replay_global_class_ids], 
                                                                   dtype=torch.float32, device=device)
                                # Apply weighted cross-entropy to replay
                                replay_log_probs = F.log_softmax(replay_task_outputs, dim=1)
                                replay_targets = torch.zeros_like(replay_log_probs).scatter_(1, replay_local_labels.unsqueeze(1), 1.0)
                                if label_smoothing > 0:
                                    replay_targets = replay_targets * (1 - label_smoothing) + label_smoothing / args.classes_per_task
                                replay_class_loss = -(replay_targets * replay_log_probs).sum(dim=1)
                                replay_class_loss = (replay_class_loss * replay_batch_weights).mean()
                                replay_loss += replay_class_loss
                            else:
                                replay_loss += criterion(replay_task_outputs, replay_local_labels)

                    if expert_id > 0:
                        replay_loss /= expert_id
                        loss += args.replay_loss_coef * replay_loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_distillation_loss += distillation_loss.item()
                _, predicted = task_outputs.max(1)
                total += local_labels.size(0)
                correct += predicted.eq(local_labels).sum().item()
                # Update progress bar with distillation info and learning rate
                postfix = {
                    'loss': f'{total_loss/(progress_bar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'ce': f"{total_ce_loss/(progress_bar.n+1):.4f}"
                }
                if expert_id > 0 and args.distillation_coef > 0:
                    postfix['distill'] = f'{total_distillation_loss/(progress_bar.n+1):.4f}'
                if expert_id > 0 and args.feature_distillation_coef > 0:
                    postfix['feat_distill'] = f'{feature_distillation_loss.item():.4f}'
                if args.use_class_balanced_loss and class_weights is not None:
                    # Show average class weight for this expert's classes
                    expert_class_weights = [class_weights[global_class] for global_class in task_class_list]
                    avg_weight = sum(expert_class_weights) / len(expert_class_weights)
                    postfix['avg_w'] = f'{avg_weight:.2f}'
                if args.use_prototype_regularization:
                    # Handle both tensor and float types
                    if isinstance(prototype_reg_loss, torch.Tensor):
                        postfix['proto_reg'] = f'{prototype_reg_loss.item():.4f}'
                    else:
                        postfix['proto_reg'] = f'{prototype_reg_loss:.4f}'
                if args.use_routing_confidence_penalty:
                    # Handle both tensor and float types
                    if hasattr(routing_confidence_loss, 'item'):
                        postfix['routing_conf'] = f'{routing_confidence_loss.item():.4f}'
                    else:
                        postfix['routing_conf'] = f'{routing_confidence_loss:.4f}'
                if expert_id > 0:
                    postfix['push_pull'] = f'{push_pull_loss.item():.4f}'
                    postfix['ewc'] = f'{ewc_loss.item():.4f}'
                progress_bar.set_postfix(postfix)
                
                # Note: Post-epoch logging is handled in the push-pull loss section above
            
            # --- Evaluation ---
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            per_class_correct = {global_class: 0 for global_class in task_class_list}
            per_class_total = {global_class: 0 for global_class in task_class_list}
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Convert global labels to local labels for this task
                    local_labels = torch.tensor([global_to_local_map[l.item()] for l in labels], dtype=torch.long, device=device)

                    outputs, _, _ = model(inputs, task_id=expert_id)
                    task_outputs = outputs[:, start_idx:end_idx]
                    loss = criterion(task_outputs, local_labels)
                    test_loss += loss.item()
                    _, predicted = task_outputs.max(1)
                    test_total += local_labels.size(0)
                    test_correct += predicted.eq(local_labels).sum().item()
                    
                    # Track per-class accuracy for class-balanced loss
                    if args.use_class_balanced_loss:
                        for i, (pred, true_local, true_global) in enumerate(zip(predicted, local_labels, labels)):
                            if pred == true_local:
                                per_class_correct[true_global.item()] += 1
                            per_class_total[true_global.item()] += 1
            
            acc = 100. * test_correct / test_total
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update class weights based on per-class accuracy if using class-balanced loss
            if args.use_class_balanced_loss and class_weights is not None:
                # Calculate per-class accuracies for this expert's classes
                class_accuracies = {}
                for global_class in task_class_list:
                    if per_class_total[global_class] > 0:
                        class_acc = per_class_correct[global_class] / per_class_total[global_class]
                        class_accuracies[global_class] = class_acc
                
                # Update weights for this expert's classes
                if len(class_accuracies) > 0:
                    new_weights = calculate_class_balanced_weights(
                        class_accuracies, 
                        epsilon=args.class_balance_epsilon,
                        smoothing=args.class_balance_smoothing
                    )
                    for global_class, weight in new_weights.items():
                        class_weights[global_class] = weight
                    
                    # Log weight updates for this expert's classes
                    weight_info = ", ".join([f"C{global_class}:{weight:.2f}" for global_class, weight in new_weights.items()])
                    logger.info(f"  Epoch {epoch+1} | Updated class weights: {weight_info}")
            
            # Call log_active_loss_fraction every epoch for all experts
            if expert_id > 0:
                # Get active prototypes for diagnostics
                active_protos, active_E = model.get_active_prototypes_for_diagnostics(expert_id)
                
                if active_protos is not None and active_E > 0:
                    # Check if we have enough prototype counts for meaningful diagnostics
                    expert_classes = model.task_classes[expert_id]
                    min_counts = 5  # Minimum counts per class for stable prototypes
                    has_enough_counts = False
                    if model.class_proto_counts is not None:
                        class_counts = model.class_proto_counts[expert_classes]
                        has_enough_counts = (class_counts >= min_counts).any()
                    
                    if has_enough_counts:
                        # Get a sample batch to compute DG outputs
                        sample_inputs, _ = next(iter(train_loader))
                        sample_inputs = sample_inputs.to(device)
                        features = model.prepare_dg_input(sample_inputs)
                        
                        # Get DG outputs for active experts only
                        active_dg_list = []
                        for e in range(active_E):
                            dg_output, _ = model.hippocampal_experts[e](features)
                            active_dg_list.append(dg_output)
                        
                        # Stack DG outputs
                        active_dg_stack = torch.stack(active_dg_list, dim=1)  # [batch, active_E, dg_dim]
                        
                        # Create task IDs for this batch (all samples belong to current expert)
                        batch_task_ids = torch.full((sample_inputs.size(0),), expert_id, device=device)
                        
                        # Call log_active_loss_fraction with active prototypes
                        log_active_loss_fraction(active_dg_stack, batch_task_ids, active_protos, log_dir, margin=0.5, epoch=epoch+1,)
        
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # Log distillation info
        if expert_id > 0 and args.distillation_coef > 0:
            logger.info(f"  Epoch {epoch+1} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%) | Distill Loss: {total_distillation_loss/len(train_loader):.4f}")
        else:
            logger.info(f"  Epoch {epoch+1} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

        # === EWC Step 1: Compute and Store Importance (AFTER training is done) ===
        # Use enhanced Fisher computation for better robustness
        fisher_matrix = model.compute_fisher_importance_enhanced(train_loader, device, num_samples=500, num_forward_passes=3)
        star_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Analyze Fisher quality
        fisher_analysis = model.analyze_fisher_quality(fisher_matrix)
        
        model.ewc_data.append({'fisher': fisher_matrix, 'star_params': star_params})
        logger.info(f"🔧 EWC: Computed Fisher importance for expert {expert_id} ({len(fisher_matrix)} parameter groups, {fisher_analysis.get('total_parameters', 0):,} total parameters)")
        # =======================================================================
        
        # --- FREEZE EXPERT PROTOTYPES AFTER TRAINING ---
        model.freeze_expert_prototypes(expert_id)
        model.trained_experts += 1
        
        # Update DG prototypes from EMA
        model.update_dg_prototypes_from_ema()
        
        expert_results.append({'expert_id': expert_id, 'final_accuracy': best_acc})
        
    return expert_results


def evaluate_dg_gated_model_standardized(model, test_loaders, task_classes, device):
    """
    MODIFIED Final evaluation for the DG-Gated model.
    Handles the model's specific outputs and Class-IL inference.
    """
    logging.info("\n" + "="*80)
    logging.info("FINAL DG-GATED PERFORMANCE EVALUATION")
    logging.info("="*80)
    
    model.eval()
    
    # Task-IL evaluation (oracle provides task_id)
    task_il_correct = 0
    task_il_total = 0
    expert_accuracies = []
    all_true_taskil = []
    all_pred_taskil = []
    all_expert_taskil = []
    all_loss_taskil = []
    all_conf_taskil = []
    with torch.no_grad():
        for expert_id, test_loader in enumerate(test_loaders):
            expert_correct = 0
            expert_total = 0
            task_class_list = task_classes[expert_id]
            global_to_local_map = {global_class: i for i, global_class in enumerate(task_class_list)}
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                local_labels = torch.tensor([global_to_local_map[l.item()] for l in labels], dtype=torch.long, device=device)
                outputs, _, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                loss = F.cross_entropy(task_outputs, local_labels, reduction='none')
                probs = F.softmax(task_outputs, dim=1)
                conf = probs.max(dim=1)[0]
                _, predicted = torch.max(task_outputs, 1)
                all_true_taskil.extend(labels.detach().cpu().numpy())
                all_pred_taskil.extend((predicted + start_idx).detach().cpu().numpy())
                all_expert_taskil.extend([expert_id]*len(labels))
                all_loss_taskil.extend(loss.detach().cpu().numpy())
                all_conf_taskil.extend(conf.detach().cpu().numpy())
                expert_correct += (predicted == local_labels).sum().item()
                expert_total += local_labels.size(0)
            expert_acc = (expert_correct / expert_total) * 100
            expert_accuracies.append(expert_acc)
            task_il_correct += expert_correct
            task_il_total += expert_total
            logging.info(f"Task-IL Expert {expert_id} (classes {task_class_list}): {expert_acc:.2f}%")
    task_il_accuracy = (task_il_correct / task_il_total) * 100
    # Class-IL evaluation (model must infer task via DG-gating)
    class_il_correct = 0
    class_il_total = 0
    all_test_data = [item for loader in test_loaders for item in loader]
    np.random.shuffle(all_test_data)
    all_true_classil = []
    all_pred_classil = []
    all_expert_classil = []
    all_loss_classil = []
    all_conf_classil = []
    with torch.no_grad():
        for inputs, labels in tqdm(all_test_data, desc="Class-IL Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Use hard gating for final evaluation if strategy is 'soft_hard'
            if model.gating_strategy == 'soft_hard':
                # Temporarily switch to hard gating for evaluation
                original_strategy = model.gating_strategy
                model.gating_strategy = 'hard'
                outputs, _, analysis_data = model(inputs)
                model.gating_strategy = original_strategy  # Restore original strategy
            else:
                outputs, _, analysis_data = model(inputs)
            
            loss = F.cross_entropy(outputs, labels, reduction='none')
            probs = F.softmax(outputs, dim=1)
            conf = probs.max(dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            chosen_experts = analysis_data['chosen_experts'].detach().cpu().numpy() if analysis_data.get('chosen_experts') is not None else np.full(len(labels), -1)
            all_true_classil.extend(labels.detach().cpu().numpy())
            all_pred_classil.extend(predicted.detach().cpu().numpy())
            all_expert_classil.extend(chosen_experts)
            all_loss_classil.extend(loss.detach().cpu().numpy())
            all_conf_classil.extend(conf.detach().cpu().numpy())
            class_il_correct += (predicted == labels).sum().item()
            class_il_total += labels.size(0)
    class_il_accuracy = (class_il_correct / class_il_total) * 100
    # Per-class confusion matrices
    cm_taskil = confusion_matrix(all_true_taskil, all_pred_taskil)
    cm_classil = confusion_matrix(all_true_classil, all_pred_classil)
    # Per-class average loss and confidence
    def per_class_stats(true, loss, conf, n_classes):
        stats = {}
        true = np.array(true)
        loss = np.array(loss)
        conf = np.array(conf)
        for c in range(n_classes):
            idx = (true == c)
            stats[c] = {
                'avg_loss': float(loss[idx].mean()) if idx.sum() > 0 else None,
                'avg_conf': float(conf[idx].mean()) if idx.sum() > 0 else None,
                'count': int(idx.sum())
            }
        return stats
    n_classes = model.num_classes
    stats_taskil = per_class_stats(all_true_taskil, all_loss_taskil, all_conf_taskil, n_classes)
    stats_classil = per_class_stats(all_true_classil, all_loss_classil, all_conf_classil, n_classes)
    # Routing decisions for Class-IL
    routing_counts = {}
    for c in range(n_classes):
        idx = (np.array(all_true_classil) == c)
        experts = np.array(all_expert_classil)[idx]
        unique, counts = np.unique(experts, return_counts=True)
        routing_counts[c] = dict(zip(unique.tolist(), counts.tolist()))
    # Log all this info
    logging.info("=== PER-CLASS CONFUSION MATRIX (TASK-IL) ===\n" + str(cm_taskil))
    logging.info("=== PER-CLASS CONFUSION MATRIX (CLASS-IL) ===\n" + str(cm_classil))
    logging.info("=== PER-CLASS STATS (TASK-IL) ===\n" + str(stats_taskil))
    logging.info("=== PER-CLASS STATS (CLASS-IL) ===\n" + str(stats_classil))
    logging.info("=== ROUTING COUNTS (CLASS-IL) ===\n" + str(routing_counts))
    
    logging.info(f"\nFINAL RESULTS:")
    logging.info(f"  - Task-IL Accuracy (Oracle): {task_il_accuracy:.2f}%")
    logging.info(f"  - Class-IL Accuracy (DG-Gated): {class_il_accuracy:.2f}%")
    
    return {'task_il_accuracy': task_il_accuracy, 'class_il_accuracy': class_il_accuracy}

def calculate_sparsity_loss(dg_outputs):
    """Calculate sparsity loss to encourage sparse activations."""
    return torch.mean(torch.abs(dg_outputs))

def calculate_expert_balancing_loss(gate_logits, target_utilization=None):
    """
    Calculate expert balancing loss to encourage equal utilization across experts.
    
    Args:
        gate_logits: Logits from gating mechanism [batch_size, num_experts]
        target_utilization: Target utilization per expert (default: uniform)
    
    Returns:
        Balancing loss that penalizes uneven expert utilization
    """
    batch_size, num_experts = gate_logits.shape
    
    # Convert logits to probabilities
    gate_probs = F.softmax(gate_logits, dim=1)  # [batch_size, num_experts]
    
    # Calculate current utilization (average probability per expert)
    current_utilization = gate_probs.mean(dim=0)  # [num_experts]
    
    # Target utilization (uniform by default)
    if target_utilization is None:
        target_utilization = torch.ones(num_experts, device=gate_logits.device) / num_experts
    
    # KL divergence between current and target utilization
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    current_utilization = current_utilization + epsilon
    target_utilization = target_utilization + epsilon
    
    # Normalize to probability distributions
    current_utilization = current_utilization / current_utilization.sum()
    target_utilization = target_utilization / target_utilization.sum()
    
    # KL divergence: KL(target || current)
    kl_loss = F.kl_div(
        torch.log(current_utilization), 
        target_utilization, 
        reduction='batchmean'
    )
    
    # Alternative: L2 loss on utilization differences
    l2_loss = F.mse_loss(current_utilization, target_utilization)
    
    # Combine both losses
    balancing_loss = kl_loss + l2_loss
    
    return balancing_loss

def calculate_distillation_loss(current_expert_outputs, previous_experts_outputs, temperature=2.0):
    """
    Calculate distillation loss between current expert and previous experts.
    
    Args:
        current_expert_outputs: Logits from the current expert (student)
        previous_experts_outputs: List of logits from previous experts (teachers)
        temperature: Temperature for softmax scaling
    
    Returns:
        distillation_loss: KL divergence loss
    """
    if not previous_experts_outputs:
        return torch.tensor(0.0, device=current_expert_outputs.device)
    
    # Apply temperature scaling to get softer probability distributions
    current_probs = F.softmax(current_expert_outputs / temperature, dim=1)
    
    # Average the teacher probabilities from all previous experts
    teacher_probs = torch.zeros_like(current_probs)
    for teacher_outputs in previous_experts_outputs:
        teacher_probs += F.softmax(teacher_outputs / temperature, dim=1)
    teacher_probs /= len(previous_experts_outputs)
    
    # Calculate KL divergence loss
    distillation_loss = F.kl_div(
        current_probs.log(), 
        teacher_probs, 
        reduction='batchmean'
    )
    
    return distillation_loss

def calculate_feature_distillation_loss(current_dg_output, previous_dg_outputs, current_ca1_output, previous_ca1_outputs):
    """
    Calculate feature-based distillation loss between current expert and previous experts.
    ONLY distills CA1 features to allow DG to remain task-specific.
    
    Args:
        current_dg_output: DG output from the current expert (student) - NOT USED
        previous_dg_outputs: List of DG outputs from previous experts (teachers) - NOT USED
        current_ca1_output: CA1 output from the current expert (student)
        previous_ca1_outputs: List of CA1 outputs from previous experts (teachers)
    
    Returns:
        distillation_loss: CA1-only distillation loss
    """
    if not previous_ca1_outputs:
        return torch.tensor(0.0, device=current_ca1_output.device)
    
    # Normalize CA1 features for better comparison
    current_ca1_norm = F.normalize(current_ca1_output, p=2, dim=1)
    
    # Calculate average teacher CA1 features
    avg_teacher_ca1 = torch.mean(torch.stack([F.normalize(ca1, p=2, dim=1) for ca1 in previous_ca1_outputs]), dim=0)
    
    # Calculate cosine similarity loss (maximize similarity)
    ca1_similarity = F.cosine_similarity(current_ca1_norm, avg_teacher_ca1, dim=1).mean()
    
    # Convert to loss (1 - similarity, so we minimize the loss)
    ca1_loss = 1.0 - ca1_similarity
    
    return ca1_loss

def analyze_class_il_breakdown(model, test_loaders, device, save_dir):
    """Generate a Class-IL breakdown analysis and visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import defaultdict
    import os

    model.eval()
    all_true = []
    all_pred = []
    all_expert = []
    all_task = []

    # Collect predictions and routing info
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use hard gating for analysis if strategy is 'soft_hard'
                if model.gating_strategy == 'soft_hard':
                    # Temporarily switch to hard gating for analysis
                    original_strategy = model.gating_strategy
                    model.gating_strategy = 'hard'
                    outputs, _, analysis_data = model(inputs)
                    model.gating_strategy = original_strategy  # Restore original strategy
                else:
                    outputs, _, analysis_data = model(inputs)
                
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                # Defensive: handle None for chosen_experts
                if analysis_data.get('chosen_experts') is not None:
                    experts = analysis_data['chosen_experts'].detach().cpu().numpy()
                else:
                    # Fallback: assign -1 if not available
                    experts = np.full(len(labels), -1)
                all_true.extend(labels.detach().cpu().numpy())
                all_pred.extend(preds)
                all_expert.extend(experts)
                all_task.extend([task_id] * len(labels))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_expert = np.array(all_expert).astype(int).copy()
    all_task = np.array(all_task)

    num_classes = model.num_classes
    num_tasks = model.num_experts
    num_experts = model.num_experts

    # Per-class accuracy
    class_acc = []
    for c in range(num_classes):
        idx = (all_true == c)
        acc = (all_pred[idx] == c).mean() if idx.sum() > 0 else 0.0
        class_acc.append(acc * 100)

    # Per-task accuracy
    task_acc = []
    for t in range(num_tasks):
        idx = (all_task == t)
        acc = (all_pred[idx] == all_true[idx]).mean() if idx.sum() > 0 else 0.0
        task_acc.append(acc * 100)

    # Per-expert accuracy
    expert_acc = []
    for e in range(num_experts):
        idx = (all_expert == e)
        acc = (all_pred[idx] == all_true[idx]).mean() if idx.sum() > 0 else 0.0
        expert_acc.append(acc * 100)

    # Class→Expert routing matrix
    routing_matrix = np.zeros((num_classes, num_experts))
    for c in range(num_classes):
        idx = (all_true == c)
        if idx.sum() > 0:
            for e in range(num_experts):
                routing_matrix[c, e] = (all_expert[idx] == e).sum() / idx.sum()

    # Expert usage (ignore -1 if present)
    expert_usage = np.bincount(all_expert[all_expert >= 0], minlength=num_experts) / np.sum(all_expert >= 0)

    # --- Visualization ---
    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)

    # 1. Per-Class Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in class_acc]
    ax1.bar(range(num_classes), class_acc, color=colors)
    for i, acc in enumerate(class_acc):
        ax1.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax1.set_title('Per-Class Class-IL Accuracy')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(float(np.mean(class_acc)), color='orange', linestyle='--')
    ax1.axhline(float(np.min(class_acc)), color='red', linestyle='--')

    # 2. Per-Task Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in task_acc]
    ax2.bar(range(num_tasks), task_acc, color=colors)
    for i, acc in enumerate(task_acc):
        ax2.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax2.set_title('Per-Task Class-IL Accuracy')
    ax2.set_xlabel('Task ID')
    ax2.set_ylabel('Accuracy (%)')
    ax2.axhline(float(np.mean(task_acc)), color='orange', linestyle='--')
    ax2.axhline(float(np.min(task_acc)), color='red', linestyle='--')

    # 3. Per-Expert Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in expert_acc]
    ax3.bar(range(num_experts), expert_acc, color=colors)
    for i, acc in enumerate(expert_acc):
        ax3.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax3.set_title('Per-Expert Class-IL Accuracy')
    ax3.set_xlabel('Expert ID')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axhline(float(np.mean(expert_acc)), color='orange', linestyle='--')
    ax3.axhline(float(np.min(expert_acc)), color='red', linestyle='--')

    # 4. Class→Expert Routing Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(routing_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax4, square=True)
    ax4.set_title('Class→Expert Routing Matrix (Class-IL)')
    ax4.set_xlabel('Expert ID')
    ax4.set_ylabel('Class ID')

    # 5. Expert Usage Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.pie(expert_usage, labels=[f"Expert {i}" for i in range(num_experts)], autopct='%1.1f%%', startangle=90)
    ax5.set_title('Expert Usage Distribution (Class-IL)')

    # 6. Summary Text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = (
        f"Class-IL Breakdown Summary\n\n"
        f"Average Class Accuracy: {np.mean(class_acc):.1f}%\n"
        f"Average Task Accuracy: {np.mean(task_acc):.1f}%\n"
        f"Average Expert Accuracy: {np.mean(expert_acc):.1f}%\n\n"
        f"Worst Classes:\n"
    )
    worst_classes = np.argsort(class_acc)[:3]
    for c in worst_classes:
        summary += f"Class {c}: {class_acc[c]:.1f}%\n"
    ax6.text(0, 1, summary, va='top', ha='left', fontsize=12, family='monospace')

    plt.suptitle('Class-IL Performance Breakdown Analysis', fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, 'Class_IL_Breakdown_Analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Class-IL Breakdown] Analysis saved to {os.path.join(save_dir, 'Class_IL_Breakdown_Analysis.png')}")

def analyze_task_il_breakdown(model, test_loaders, device, save_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    model.eval()
    all_true = []
    all_pred = []
    all_expert = []
    all_task = []
    with torch.no_grad():
        for expert_id, test_loader in enumerate(test_loaders):
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = model(inputs, task_id=expert_id)
                preds = outputs.argmax(dim=1).detach().cpu().numpy() + expert_id * model.classes_per_task
                all_true.extend(labels.detach().cpu().numpy())
                all_pred.extend(preds)
                all_expert.extend([expert_id]*len(labels))
                all_task.extend([expert_id]*len(labels))
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_expert = np.array(all_expert).astype(int).copy()
    all_task = np.array(all_task)
    num_classes = model.num_classes
    num_tasks = model.num_experts
    num_experts = model.num_experts
    # Per-class accuracy
    class_acc = []
    for c in range(num_classes):
        idx = (all_true == c)
        acc = (all_pred[idx] == c).mean() if idx.sum() > 0 else 0.0
        class_acc.append(acc * 100)
    # Per-task accuracy
    task_acc = []
    for t in range(num_tasks):
        idx = (all_task == t)
        acc = (all_pred[idx] == all_true[idx]).mean() if idx.sum() > 0 else 0.0
        task_acc.append(acc * 100)
    # Per-expert accuracy
    expert_acc = []
    for e in range(num_experts):
        idx = (all_expert == e)
        acc = (all_pred[idx] == all_true[idx]).mean() if idx.sum() > 0 else 0.0
        expert_acc.append(acc * 100)
    # Class→Expert routing matrix (oracle routing)
    routing_matrix = np.zeros((num_classes, num_experts))
    for c in range(num_classes):
        idx = (all_true == c)
        if idx.sum() > 0:
            for e in range(num_experts):
                routing_matrix[c, e] = (all_expert[idx] == e).sum() / idx.sum()
    # Expert usage
    expert_usage = np.bincount(all_expert[all_expert >= 0], minlength=num_experts) / np.sum(all_expert >= 0)
    # --- Visualization ---
    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    # 1. Per-Class Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in class_acc]
    ax1.bar(range(num_classes), class_acc, color=colors)
    for i, acc in enumerate(class_acc):
        ax1.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax1.set_title('Per-Class Task-IL Accuracy')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(float(np.mean(class_acc)), color='orange', linestyle='--')
    ax1.axhline(float(np.min(class_acc)), color='red', linestyle='--')
    # 2. Per-Task Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in task_acc]
    ax2.bar(range(num_tasks), task_acc, color=colors)
    for i, acc in enumerate(task_acc):
        ax2.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax2.set_title('Per-Task Task-IL Accuracy')
    ax2.set_xlabel('Task ID')
    ax2.set_ylabel('Accuracy (%)')
    ax2.axhline(float(np.mean(task_acc)), color='orange', linestyle='--')
    ax2.axhline(float(np.min(task_acc)), color='red', linestyle='--')
    # 3. Per-Expert Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if acc > 50 else 'orange' if acc > 20 else 'red' for acc in expert_acc]
    ax3.bar(range(num_experts), expert_acc, color=colors)
    for i, acc in enumerate(expert_acc):
        ax3.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)
    ax3.set_title('Per-Expert Task-IL Accuracy')
    ax3.set_xlabel('Expert ID')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axhline(float(np.mean(expert_acc)), color='orange', linestyle='--')
    ax3.axhline(float(np.min(expert_acc)), color='red', linestyle='--')
    # 4. Class→Expert Routing Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(routing_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax4, square=True)
    ax4.set_title('Class→Expert Routing Matrix (Task-IL)')
    ax4.set_xlabel('Expert ID')
    ax4.set_ylabel('Class ID')
    # 5. Expert Usage Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.pie(expert_usage, labels=[f"Expert {i}" for i in range(num_experts)], autopct='%1.1f%%', startangle=90)
    ax5.set_title('Expert Usage Distribution (Task-IL)')
    # 6. Summary Text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = (
        f"Task-IL Breakdown Summary\n\n"
        f"Average Class Accuracy: {np.mean(class_acc):.1f}%\n"
        f"Average Task Accuracy: {np.mean(task_acc):.1f}%\n"
        f"Average Expert Accuracy: {np.mean(expert_acc):.1f}%\n\n"
        f"Worst Classes:\n"
    )
    worst_classes = np.argsort(class_acc)[:3]
    for c in worst_classes:
        summary += f"Class {c}: {class_acc[c]:.1f}%\n"
    ax6.text(0, 1, summary, va='top', ha='left', fontsize=12, family='monospace')
    plt.suptitle('Task-IL Performance Breakdown Analysis', fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, 'Task_IL_Breakdown_Analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Task-IL Breakdown] Analysis saved to {os.path.join(save_dir, 'Task_IL_Breakdown_Analysis.png')}")

def log_expert_utilization(gate_logits, epoch, log_dir):
    """
    Logs and plots the fraction of samples routed to each expert for the current epoch.
    gate_logits: [num_samples, num_experts] or routing decisions [num_samples]
    epoch: int, current epoch number
    log_dir: directory to save the plot/log
    """
    # If gate_logits are logits, convert to routing decisions
    if len(gate_logits.shape) == 2:
        routed_experts = np.argmax(gate_logits.detach().cpu().numpy(), axis=1)
    else:
        routed_experts = gate_logits

    num_experts = np.max(routed_experts) + 1
    counts = np.bincount(routed_experts, minlength=num_experts)
    utilization = counts / counts.sum()

    # Save as CSV
    csv_path = os.path.join(log_dir, f"expert_utilization_epoch{epoch}.csv")
    np.savetxt(csv_path, utilization, delimiter=",", header="expert_utilization", comments='')

    # Plot
    plt.figure()
    plt.bar(range(num_experts), utilization)
    plt.xlabel("Expert")
    plt.ylabel("Utilization Fraction")
    plt.title(f"Expert Utilization - Epoch {epoch}")
    plt.savefig(os.path.join(log_dir, f"expert_utilization_epoch{epoch}.png"))
    plt.close()

def log_active_loss_fraction(all_dg_outputs, task_ids, dg_prototypes,log_dir, margin, epoch):
    """
    Logs the pull and push diagnostics for the contrastive loss.
    Pull: mean similarity, mean distance, fraction below margin.
    Push: fraction of negatives inside margin.
    """
    import numpy as np
    import os
    import torch
    import torch.nn.functional as F
    # from evaluation import evaluate_baseline_model_standardized, evaluate_dg_gated_model_standardized
    with torch.no_grad():
        dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
        prototypes_norm = F.normalize(dg_prototypes, p=2, dim=1)
        similarities = torch.einsum('bed,ed->be', dg_outputs_norm, prototypes_norm)
        batch_indices = torch.arange(task_ids.size(0), device=task_ids.device)
        correct_expert_sims = similarities[batch_indices, task_ids]

        # Pull diagnostics
        pull_sim_mean = correct_expert_sims.mean().item()
        pull_distance_mean = (1 - correct_expert_sims).mean().item()
        pull_margin = 0.8
        pull_violations_frac = (correct_expert_sims < pull_margin).float().mean().item()

        # Push diagnostics (as before)
        mask = torch.ones_like(similarities)
        mask[batch_indices, task_ids] = 0
        push_active = ((similarities - margin) > 0) & (mask > 0)
        push_active_count = push_active.sum().item()
        total_push = mask.sum().item()
        push_active_frac = push_active_count / total_push if total_push > 0 else 0.0

        # Log to file
        log_path = os.path.join(log_dir, f"active_loss_fraction_epoch{epoch}.txt")
        with open(log_path, 'w') as f:
            f.write(f"Pull mean similarity: {pull_sim_mean:.4f}\n")
            f.write(f"Pull mean distance: {pull_distance_mean:.4f}\n")
            f.write(f"Pull violations (<{pull_margin}): {pull_violations_frac:.4f}\n")
            f.write(f"Push active fraction: {push_active_frac:.4f}\n")
        print(f"[Epoch {epoch}] Pull sim: {pull_sim_mean:.4f}, Pull dist: {pull_distance_mean:.4f}, Pull violations: {pull_violations_frac:.4f}, Push active: {push_active_frac:.4f}")

def plot_contrastive_similarity_histogram(*args, **kwargs):
    pass

def plot_prototype_distance_matrix(*args, **kwargs):
    pass

def plot_tsne_dg(*args, **kwargs):
    pass

def calculate_contrastive_diagnostics(dg_outputs_all_experts, task_ids, prototypes, margin, active_experts=None):
    """
    Calculate contrastive learning diagnostics for monitoring training.
    
    Args:
        dg_outputs_all_experts: [batch_size, num_experts, dg_dim] DG outputs from all experts
        task_ids: [batch_size] Task/expert IDs for each sample
        prototypes: [num_prototypes, dg_dim] Expert prototypes
        margin: Margin for contrastive loss
        active_experts: Number of active experts (if None, use all)
    
    Returns:
        pull_sim_mean: Average similarity between samples and their correct prototypes
        pull_distance_mean: Average distance between samples and their correct prototypes  
        pull_violations: Fraction of samples with similarity below margin
        push_active: Fraction of negative pairs with similarity above margin
    """
    with torch.no_grad():
        # Limit to active experts if specified
        if active_experts is not None:
            dg_outputs_all_experts = dg_outputs_all_experts[:, :active_experts]
            prototypes = prototypes[:active_experts]
            # Clamp task_ids to valid range
            task_ids = torch.clamp(task_ids, max=active_experts-1)
        
        dg_outputs_norm = F.normalize(dg_outputs_all_experts, p=2, dim=2)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        similarities = torch.einsum('bed,ed->be', dg_outputs_norm, prototypes_norm)
        
        batch_indices = torch.arange(task_ids.size(0), device=task_ids.device)
        correct_expert_sims = similarities[batch_indices, task_ids]
        
        # Pull diagnostics
        pull_sim_mean = correct_expert_sims.mean().item()
        pull_distance_mean = (1 - correct_expert_sims).mean().item()
        pull_violations = (correct_expert_sims < margin).float().mean().item()
        
        # Push diagnostics
        # Create mask for negative pairs (same sample, different experts)
        batch_size, num_experts = similarities.shape
        push_active_count = 0
        total_negative_pairs = 0
        
        for i in range(batch_size):
            correct_expert = task_ids[i]
            for j in range(num_experts):
                if j != correct_expert:
                    if similarities[i, j] > margin:
                        push_active_count += 1
                    total_negative_pairs += 1
        
        push_active = push_active_count / max(1, total_negative_pairs)
        
        return pull_sim_mean, pull_distance_mean, pull_violations, push_active

def calculate_expert_utilization_stats(gate_logits):
    """
    Calculate expert utilization statistics for monitoring routing balance.
    """
    with torch.no_grad():
        # Convert to probabilities
        probs = F.softmax(gate_logits, dim=1)
        # Calculate utilization per expert
        utilization = probs.mean(dim=0)  # [num_experts]
        
        # Statistics
        mean_util = utilization.mean().item()
        std_util = utilization.std().item()
        min_util = utilization.min().item()
        max_util = utilization.max().item()
        variance_util = utilization.var().item()
        
        return {
            'mean': mean_util,
            'std': std_util, 
            'min': min_util,
            'max': max_util,
            'variance': variance_util,
            'utilization': utilization.detach().cpu().numpy()
        }

def calculate_dg_sparsity_stats(model, inputs, device):
    """
    Calculate actual vs target DG sparsity statistics.
    """
    with torch.no_grad():
        features = model.prepare_dg_input(inputs)
        all_sparsity_stats = []
        
        for expert_id in range(model.num_experts):
            dg_output, _ = model.hippocampal_experts[expert_id](features)
            # Count non-zero activations
            active_units = (dg_output > 0).float().mean(dim=0)  # [dg_dim]
            actual_sparsity = active_units.mean().item()
            sparsity_std = active_units.std().item()
            
            all_sparsity_stats.append({
                'expert_id': expert_id,
                'actual_sparsity': actual_sparsity,
                'target_sparsity': getattr(model, 'target_sparsity', 0.15),
                'sparsity_std': sparsity_std,
                'sparsity_error': abs(actual_sparsity - getattr(model, 'target_sparsity', 0.15))
            })
        
        return all_sparsity_stats

def calculate_gating_confidence_stats(gate_logits):
    """
    Calculate gating confidence and entropy statistics.
    """
    with torch.no_grad():
        probs = F.softmax(gate_logits, dim=1)
        
        # Confidence: max probability per sample
        confidence = probs.max(dim=1)[0]  # [batch_size]
        mean_confidence = confidence.mean().item()
        std_confidence = confidence.std().item()
        
        # Entropy: measure of uncertainty
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [batch_size]
        mean_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()
        
        return {
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy
        }

def calculate_prototype_similarity_stats(prototypes):
    """
    Calculate prototype cosine similarity matrix statistics.
    """
    with torch.no_grad():
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        sim_matrix = prototypes_norm @ prototypes_norm.T
        
        # Off-diagonal similarities (should be low for good separation)
        num_experts = sim_matrix.size(0)
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
        off_diag_sims = sim_matrix[mask]
        
        mean_off_diag = off_diag_sims.mean().item()
        std_off_diag = off_diag_sims.std().item()
        max_off_diag = off_diag_sims.max().item()
        min_off_diag = off_diag_sims.min().item()
        
        return {
            'mean_off_diag': mean_off_diag,
            'std_off_diag': std_off_diag,
            'max_off_diag': max_off_diag,
            'min_off_diag': min_off_diag
        }
        
        
def compute_dg_prototypes(self, train_loaders, device):
    """
    Computes the prototype DG pattern for each expert by averaging the DG
    output over all training samples for that expert's task.
    """
    logging.info("🧠 Computing all DG Prototypes post-Phase 1...")
    self.eval()
    with torch.no_grad():
        for task_id, train_loader in enumerate(tqdm(train_loaders, desc="Computing Prototypes")):
            all_dg_outputs = []
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                features = self.prepare_dg_input(inputs)
                dg_output, _ = self.hippocampal_experts[task_id](features)
                all_dg_outputs.append(dg_output)
            
            # Average all DG outputs for this task
            self.dg_prototypes[task_id] = torch.cat(all_dg_outputs, dim=0).mean(dim=0)
    
    self.prototypes_computed = True
    logging.info("✅ All expert prototypes computed and stored.")

def forward(self, x, task_id=None):
    """
    Forward pass with DG-based gating.
    If task_id is provided, it uses oracle routing. Otherwise, it uses
    DG pattern similarity to find the best expert.
    Returns a dictionary in the third position for analysis data.
    """
    if not self.prototypes_computed and task_id is None:
        # During inference, prototypes must have been computed.
        if not self.training:
            raise RuntimeError("DG prototypes have not been computed. Call compute_dg_prototypes() first.")

    features = self.prepare_dg_input(x)
    gate_logits = None # Default for oracle routing
    analysis_data = {}

    if self.training and task_id is not None:
        # Oracle routing for training experts
        # Process through the single chosen expert
        dg_output, ca1_output = self.hippocampal_experts[task_id](features)
        expert_output = self.output_layers[task_id](ca1_output)
        
        final_outputs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        start_idx = task_id * self.classes_per_task
        end_idx = start_idx + self.classes_per_task
        final_outputs[:, start_idx:end_idx] = expert_output
        
        analysis_data['dg_output'] = dg_output
        return final_outputs, gate_logits, analysis_data

    # --- Gating for Inference ---
    all_dg_outputs = []
    for i in range(self.num_experts):
        dg_output, _ = self.hippocampal_experts[i](features)
        all_dg_outputs.append(dg_output)
    
    all_dg_outputs = torch.stack(all_dg_outputs, dim=1)
    all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
    prototypes_norm = F.normalize(self.dg_prototypes, p=2, dim=1).to(x.device)
    
    # Calculate DG pattern similarity for gating
    gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
    
    # --- Apply Gating Strategy ---
    final_outputs = torch.zeros(x.size(0), self.num_classes, device=x.device)

    if self.gating_strategy == 'hard':
        # Winner-take-all: choose the expert with the highest similarity
        chosen_experts = torch.argmax(gate_logits, dim=1)
        for i in range(x.size(0)):
            expert_id = chosen_experts[i].item()
            _, ca1_output = self.hippocampal_experts[expert_id](features[i].unsqueeze(0))
            expert_output = self.output_layers[expert_id](ca1_output)
            start_idx = expert_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            final_outputs[i, start_idx:end_idx] = expert_output

    elif self.gating_strategy == 'soft':
        # Soft gating: weighted average of all expert outputs
        gating_weights = F.softmax(gate_logits / self.gating_temperature, dim=1)
        for expert_id in range(self.num_experts):
            weight = gating_weights[:, expert_id].unsqueeze(1)
            _, ca1_output = self.hippocampal_experts[expert_id](features)
            expert_output = self.output_layers[expert_id](ca1_output)
            start_idx = expert_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            final_outputs[:, start_idx:end_idx] += weight * expert_output

    elif self.gating_strategy == 'top2':
        # Top-2 gating: weighted average of the top two expert outputs
        top2_logits, top2_indices = torch.topk(gate_logits, 2, dim=1)
        top2_weights = F.softmax(top2_logits / self.gating_temperature, dim=1)
        
        for i in range(x.size(0)):
            for j in range(2):
                expert_id = top2_indices[i, j].item()
                weight = top2_weights[i, j]
                _, ca1_output = self.hippocampal_experts[expert_id](features[i].unsqueeze(0))
                expert_output = self.output_layers[expert_id](ca1_output)
                start_idx = expert_id * self.classes_per_task
                end_idx = start_idx + self.classes_per_task
                final_outputs[i, start_idx:end_idx] += weight * expert_output.squeeze(0)

    elif self.gating_strategy == 'soft_hard':
        # Soft-hard gating: always use soft gating, hard gating only in final evaluation
        # This will be handled in the evaluation function
        gating_weights = F.softmax(gate_logits / self.gating_temperature, dim=1)
        for expert_id in range(self.num_experts):
            weight = gating_weights[:, expert_id].unsqueeze(1)
            _, ca1_output = self.hippocampal_experts[expert_id](features)
            expert_output = self.output_layers[expert_id](ca1_output)
            start_idx = expert_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            final_outputs[:, start_idx:end_idx] += weight * expert_output

    analysis_data['chosen_experts'] = torch.argmax(gate_logits, dim=1) if gate_logits is not None else None
        
    return final_outputs, gate_logits, analysis_data



def calculate_memory_stats(model):
    """
    Calculate memory usage and replay buffer statistics.
    """
    total_replay_samples = sum(len(buffer) for buffer in model.replay_buffer)
    replay_distribution = [len(buffer) for buffer in model.replay_buffer]
    
    return {
        'total_replay_samples': total_replay_samples,
        'replay_distribution': replay_distribution,
        'mean_replay_per_task': total_replay_samples / model.num_experts if model.num_experts > 0 else 0
    }

def calculate_gradient_norms(model):
    """
    Calculate gradient norms for debugging training stability.
    """
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, param_count

# ============================================================================
# DG PATTERN SEPARATION DIAGNOSTICS
# ============================================================================

def analyze_dg_pattern_separation(model, test_loaders, device, save_dir=None):
    """
    DG Pattern Separation Diagnostic
    
    This diagnostic quantifies how well the DG embeddings separate by expert (task) clusters.
    It computes:
    1. Silhouette Score – measures how similar an embedding is to its own cluster vs. other clusters (range -1 to 1).
    2. Davies–Bouldin Index – average "similarity" between each cluster and its most similar one (lower is better).
    3. Mean Inter‑Cluster vs. Intra‑Cluster Distances – average distance within each expert's DG embeddings vs. between experts.
    """
    logging.info("\n" + "🔬" * 60)
    logging.info("🔬 DG PATTERN SEPARATION DIAGNOSTIC")
    logging.info("🔬" * 60)
    
    model.eval()
    
    # Collect DG outputs and true expert labels
    all_dg_outputs = []
    all_true_experts = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in tqdm(test_loader, desc=f"Collecting DG outputs for Task {task_id}"):
                inputs = inputs.to(device)
                features = model.prepare_dg_input(inputs)
                
                # Get DG outputs from all experts
                batch_dg_outputs = []
                for expert_id in range(model.num_experts):
                    dg_output, _ = model.hippocampal_experts[expert_id](features)
                    batch_dg_outputs.append(dg_output)
                
                # Stack outputs from all experts: [batch_size, num_experts, dg_dim]
                batch_dg_outputs = torch.stack(batch_dg_outputs, dim=1)
                
                all_dg_outputs.append(batch_dg_outputs)
                all_true_experts.append(torch.full((inputs.size(0),), task_id))
    
    # Concatenate all batches
    dg_outputs = torch.cat(all_dg_outputs, dim=0)  # [N, num_experts, dg_dim]
    true_expert = torch.cat(all_true_experts, dim=0)  # [N]
    
    N, num_experts, dg_dim = dg_outputs.shape
    
    logging.info(f"📊 Collected {N} samples across {num_experts} experts")
    logging.info(f"📊 DG embedding dimension: {dg_dim}")
    
    # 1. Flatten and assign each sample to the highest-scoring expert
    normed = torch.nn.functional.normalize(dg_outputs.view(N, -1), p=2, dim=1)
    embeddings = normed.numpy()
    labels = true_expert.numpy()
    
    # Validate data before clustering
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        logging.warning("⚠️ Invalid values detected in embeddings (NaN or Inf)")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Check if we have enough samples per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples_per_cluster = counts.min() if len(counts) > 0 else 0
    if min_samples_per_cluster < 2:
        logging.warning(f"⚠️ Insufficient samples per cluster (min: {min_samples_per_cluster}, need at least 2)")
        return {
            'silhouette_score': float('nan'),
            'davies_bouldin_index': float('nan'),
            'mean_intra_cluster_distance': float('nan'),
            'mean_inter_cluster_distance': float('nan'),
            'separation_ratio': float('nan'),
            'num_samples': N,
            'num_experts': num_experts,
            'dg_dim': dg_dim,
            'error': 'insufficient_samples_per_cluster'
        }
    
    # 2. Compute clustering quality scores
    try:
        sil_score = silhouette_score(embeddings, labels)
        logging.info(f"✅ Silhouette Score: {sil_score:.4f}")
    except Exception as e:
        sil_score = float('nan')
        logging.warning(f"⚠️ Failed to compute Silhouette Score: {e}")
    
    try:
        db_index = davies_bouldin_score(embeddings, labels)
        logging.info(f"✅ Davies–Bouldin Index: {db_index:.4f}")
    except Exception as e:
        db_index = float('nan')
        logging.warning(f"⚠️ Failed to compute Davies–Bouldin Index: {e}")
    
        # 3. Compute intra/inter distances manually with better numerical stability
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        logging.warning("⚠️ Invalid values in embeddings, skipping distance calculations")
        mean_intra = mean_inter = separation_ratio = float('nan')
    else:
        def pairwise_dists(x):
            # Use sklearn's pairwise_distances for better numerical stability
            return pairwise_distances(x, metric='euclidean')
        
        try:
            dists = pairwise_dists(embeddings)
            
            intra = []
            inter = []
            for i in range(num_experts):
                mask_i = (labels == i)
                if np.sum(mask_i) > 1:  # Need at least 2 samples per cluster
                    # Get intra-cluster distances (excluding diagonal)
                    intra_mask = dists[mask_i][:, mask_i]
                    intra_indices = np.triu_indices(intra_mask.shape[0], k=1)
                    intra += list(intra_mask[intra_indices])
                    
                    # Get inter-cluster distances
                    inter_mask = dists[mask_i][:, ~mask_i]
                    inter += list(inter_mask.ravel())
            
            mean_intra = np.mean(intra) if intra else float('nan')
            mean_inter = np.mean(inter) if inter else float('nan')
            
            logging.info(f"✅ Mean Intra-Cluster Dist: {mean_intra:.4f}")
            logging.info(f"✅ Mean Inter-Cluster Dist: {mean_inter:.4f}")
            
            if not np.isnan(mean_intra) and not np.isnan(mean_inter):
                separation_ratio = mean_inter / mean_intra
                logging.info(f"✅ Separation Ratio (Inter/Intra): {separation_ratio:.4f}")
                
                if separation_ratio > 2.0:
                    logging.info("🎉 EXCELLENT: Strong DG pattern separation!")
                elif separation_ratio > 1.5:
                    logging.info("✅ GOOD: Moderate DG pattern separation")
                else:
                    logging.info("⚠️ WEAK: Poor DG pattern separation")
            else:
                separation_ratio = float('nan')
                logging.warning("⚠️ Could not compute separation ratio")
                
        except Exception as e:
            mean_intra = mean_inter = separation_ratio = float('nan')
            logging.warning(f"⚠️ Failed to compute distance metrics: {e}")
    
    # 4. Return comprehensive results
    results = {
        'silhouette_score': sil_score,
        'davies_bouldin_index': db_index,
        'mean_intra_cluster_distance': mean_intra,
        'mean_inter_cluster_distance': mean_inter,
        'separation_ratio': separation_ratio,
        'num_samples': N,
        'num_experts': num_experts,
        'dg_dim': dg_dim
    }
    
    logging.info("🔬" * 60)
    logging.info("🔬 DG PATTERN SEPARATION DIAGNOSTIC COMPLETE")
    logging.info("🔬" * 60)
    
    return results

def dg_separation_diagnostic(dg_outputs, expert_labels, logger=None):
    """
    Computes DG separation metrics:
    - Silhouette Score
    - Davies–Bouldin Index
    - Mean intra- and inter-cluster distances
    """
    sil = silhouette_score(dg_outputs, expert_labels)
    db = davies_bouldin_score(dg_outputs, expert_labels)
    dists = pairwise_distances(dg_outputs)
    intra, inter = [], []
    for i in range(len(expert_labels)):
        for j in range(i+1, len(expert_labels)):
            if expert_labels[i] == expert_labels[j]:
                intra.append(dists[i, j])
            else:
                inter.append(dists[i, j])
    mean_intra = np.mean(intra)
    mean_inter = np.mean(inter)
    if logger:
        logger.info(f"DG Separation Diagnostic:")
        logger.info(f"  Silhouette Score: {sil:.3f} (≳ 0.5 is good)")
        logger.info(f"  Davies–Bouldin Index: {db:.3f} (< 1 is good)")
        logger.info(f"  Mean Intra-Cluster Distance: {mean_intra:.3f}")
        logger.info(f"  Mean Inter-Cluster Distance: {mean_inter:.3f}")
        logger.info(f"  Ratio (inter/intra): {mean_inter/mean_intra:.2f}")
    return {
        "silhouette": sil,
        "db_index": db,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "ratio": mean_inter/mean_intra
    }

def diagnose_prototype_status(model):
    """
    Diagnose the current status of prototypes in the model
    """
    logging.info("\n" + "🔍" * 50)
    logging.info("🔍 PROTOTYPE STATUS DIAGNOSTIC")
    logging.info("🔍" * 50)
    
    for expert_id in range(model.num_experts):
        if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None:
            if expert_id in model.dg_prototypes:
                proto = model.dg_prototypes[expert_id]
                if proto is not None:
                    norm = torch.norm(proto).item()
                    logging.info(f"✅ Expert {expert_id}: Prototype exists, norm={norm:.3f}")
                else:
                    logging.warning(f"⚠️ Expert {expert_id}: Prototype is None")
            else:
                logging.warning(f"⚠️ Expert {expert_id}: No prototype found")
        else:
            logging.warning(f"⚠️ Expert {expert_id}: No prototype system initialized")
    
    logging.info("🔍" * 50)

# ============================================================================
# PLOTTING FUNCTIONS FOR DIAGNOSTICS
# ============================================================================

def plot_prototype_stats(prototypes, phase, save_dir, logger=None):
    """Plot prototype statistics including norms and similarity matrix"""
    norms = np.linalg.norm(prototypes, axis=1)
    mean_norm = norms.mean()
    std_norm = norms.std()
    if logger:
        logger.info(f"[{phase}] Prototype norm: mean={mean_norm:.3f}, std={std_norm:.3f}")
    plt.figure()
    plt.hist(norms, bins=20, color='blue', alpha=0.7)
    plt.title(f'Prototype Norms ({phase})')
    plt.xlabel('L2 Norm')
    plt.ylabel('Count')
    plt.savefig(f'{save_dir}/prototype_norms_{phase}.png')
    plt.close()
    # Similarity matrix
    sim = np.dot(prototypes, prototypes.T) / (np.linalg.norm(prototypes, axis=1, keepdims=True) * np.linalg.norm(prototypes, axis=1, keepdims=True).T)
    off_diag = sim[~np.eye(sim.shape[0],dtype=bool)]
    if logger:
        logger.info(f"[{phase}] Prototype similarity: mean={off_diag.mean():.3f}, min={off_diag.min():.3f}, max={off_diag.max():.3f}")
    plt.figure()
    sns.heatmap(sim, annot=False, cmap='RdBu_r', center=0)
    plt.title(f'Prototype Cosine Similarity ({phase})')
    plt.savefig(f'{save_dir}/prototype_similarity_{phase}.png')
    plt.close()

def plot_gating_confidence(gate_logits, phase, save_dir, logger=None):
    """Plot gating confidence distribution"""
    conf = gate_logits.max(axis=1)
    entropy = -np.sum(np.exp(gate_logits) * gate_logits, axis=1)
    if logger:
        logger.info(f"[{phase}] Gating confidence: mean={conf.mean():.3f}, std={conf.std():.3f}")
        logger.info(f"[{phase}] Gating entropy: mean={entropy.mean():.3f}, std={entropy.std():.3f}")
    plt.figure()
    plt.hist(conf, bins=30, color='purple', alpha=0.7)
    plt.title(f'Gating Confidence ({phase})')
    plt.xlabel('Max Softmax')
    plt.ylabel('Count')
    plt.savefig(f'{save_dir}/gating_confidence_{phase}.png')
    plt.close()

def plot_cluster_purity(labels, preds, phase, save_dir, logger=None):
    """Plot cluster purity per expert"""
    cm = confusion_matrix(labels, preds)
    purity = np.max(cm, axis=1) / (cm.sum(axis=1) + 1e-8)
    if logger:
        logger.info(f"[{phase}] Cluster purity: mean={purity.mean():.3f}, min={purity.min():.3f}, max={purity.max():.3f}")
    plt.figure()
    plt.bar(range(len(purity)), purity)
    plt.title(f'Cluster Purity ({phase})')
    plt.xlabel('Cluster/Expert')
    plt.ylabel('Purity')
    plt.savefig(f'{save_dir}/cluster_purity_{phase}.png')
    plt.close()

def plot_per_class_accuracy(true, pred, n_classes, phase, save_dir, logger=None):
    """Plot per-class accuracy"""
    accs = []
    for c in range(n_classes):
        idx = (true == c)
        acc = (pred[idx] == c).mean() if idx.sum() > 0 else 0.0
        accs.append(acc)
    if logger:
        logger.info(f"[{phase}] Per-class accuracy: mean={np.mean(accs):.3f}, min={np.min(accs):.3f}, max={np.max(accs):.3f}")
    plt.figure()
    plt.bar(range(n_classes), accs)
    plt.title(f'Per-Class Accuracy ({phase})')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.savefig(f'{save_dir}/per_class_accuracy_{phase}.png')
    plt.close()

def log_and_plot_prototype_drift(proto_before, proto_after, phase_from, phase_to, save_dir, logger=None):
    """Logs and plots prototype drift between two phases"""
    drifts = np.linalg.norm(proto_after - proto_before, axis=1)
    mean_drift = drifts.mean()
    max_drift = drifts.max()
    min_drift = drifts.min()
    if logger:
        logger.info(f"Prototype drift from {phase_from} to {phase_to}: mean={mean_drift:.4f}, min={min_drift:.4f}, max={max_drift:.4f}")
    # Line plot
    plt.figure()
    plt.plot(drifts, marker='o')
    plt.title(f'Prototype Drift per Expert ({phase_from} → {phase_to})')
    plt.xlabel('Expert')
    plt.ylabel('L2 Drift')
    plt.savefig(f'{save_dir}/prototype_drift_{phase_from}_to_{phase_to}.png')
    plt.close()
    # Histogram
    plt.figure()
    plt.hist(drifts, bins=20, color='orange', alpha=0.7)
    plt.title(f'Prototype Drift Distribution ({phase_from} → {phase_to})')
    plt.xlabel('L2 Drift')
    plt.ylabel('Count')
    plt.savefig(f'{save_dir}/prototype_drift_hist_{phase_from}_to_{phase_to}.png')
    plt.close()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_and_visualize_dg_features(
    model,
    data_loaders,
    device,
    save_dir,
    max_samples=2000,
    reduction="tsne",
    use_correct_expert=True,
):
    """
    Extract final DG outputs and visualize them in 2D.

    Args:
        model:
            Trained DGGatedHippocampalMoE model.

        data_loaders:
            List of loaders, one per task/expert.

        device:
            CPU or CUDA device.

        save_dir:
            Directory in which outputs will be saved.

        max_samples:
            Maximum number of samples to collect.

        reduction:
            "pca" or "tsne".

        use_correct_expert:
            True:
                Pass task i's samples through expert i.
            False:
                Route each sample using DG prototype similarity and extract
                the selected expert's DG representation.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    collected_dg = []
    collected_classes = []
    collected_tasks = []
    collected_experts = []

    with torch.no_grad():
        for task_id, loader in enumerate(data_loaders):
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Raw flattened image values passed directly to DG
                features = model.prepare_dg_input(inputs)

                if use_correct_expert:
                    # Oracle expert: task i is processed by expert i
                    dg_output, _ = model.hippocampal_experts[task_id](features)

                    selected_experts = torch.full(
                        (inputs.size(0),),
                        task_id,
                        dtype=torch.long,
                        device=device,
                    )

                else:
                    # Obtain DG output from every expert
                    all_dg_outputs = []

                    for expert_id in range(model.trained_experts):
                        dg_output_i, _ = model.hippocampal_experts[expert_id](
                            features
                        )
                        all_dg_outputs.append(dg_output_i)

                    # [batch, experts, dg_dim]
                    all_dg_outputs = torch.stack(all_dg_outputs, dim=1)

                    normalized_dg = torch.nn.functional.normalize(
                        all_dg_outputs,
                        p=2,
                        dim=2,
                    )

                    normalized_prototypes = torch.nn.functional.normalize(
                        model.dg_prototypes[:model.trained_experts],
                        p=2,
                        dim=1,
                    ).to(device)

                    # [batch, experts]
                    similarities = torch.einsum(
                        "bed,ed->be",
                        normalized_dg,
                        normalized_prototypes,
                    )

                    selected_experts = similarities.argmax(dim=1)

                    # Select each sample's routed DG representation
                    batch_indices = torch.arange(
                        inputs.size(0),
                        device=device,
                    )

                    dg_output = all_dg_outputs[
                        batch_indices,
                        selected_experts,
                    ]

                collected_dg.append(dg_output.cpu())
                collected_classes.append(labels.cpu())
                collected_tasks.append(
                    torch.full(
                        (inputs.size(0),),
                        task_id,
                        dtype=torch.long,
                    )
                )
                collected_experts.append(selected_experts.cpu())

                current_count = sum(x.size(0) for x in collected_dg)

                if current_count >= max_samples:
                    break

            if sum(x.size(0) for x in collected_dg) >= max_samples:
                break

    dg_features = torch.cat(collected_dg, dim=0)[:max_samples]
    class_labels = torch.cat(collected_classes, dim=0)[:max_samples]
    task_labels = torch.cat(collected_tasks, dim=0)[:max_samples]
    expert_labels = torch.cat(collected_experts, dim=0)[:max_samples]

    # Save the original 512-dimensional DG representations
    torch.save(
        {
            "dg_features": dg_features,
            "class_labels": class_labels,
            "task_labels": task_labels,
            "selected_experts": expert_labels,
        },
        os.path.join(save_dir, "dg_features.pt"),
    )

    features_np = dg_features.numpy()

    # PCA before t-SNE makes t-SNE faster and less noisy
    if reduction.lower() == "tsne":
        intermediate_dim = min(
            50,
            features_np.shape[1],
            features_np.shape[0] - 1,
        )

        pca_features = PCA(
            n_components=intermediate_dim,
            random_state=42,
        ).fit_transform(features_np)

        embedded = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(features_np) // 10)),
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(pca_features)

    elif reduction.lower() == "pca":
        embedded = PCA(
            n_components=2,
            random_state=42,
        ).fit_transform(features_np)

    else:
        raise ValueError("reduction must be 'pca' or 'tsne'")

    # Plot by class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=class_labels.detach().cpu().numpy(),
        cmap="tab10",
        s=16,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Class")
    plt.xlabel(f"{reduction.upper()} dimension 1")
    plt.ylabel(f"{reduction.upper()} dimension 2")
    plt.title("Final DG representations colored by class")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"dg_by_class_{reduction}.png"),
        dpi=200,
    )
    plt.close()

    # Plot by true task
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=task_labels.detach().cpu().numpy(),
        cmap="tab10",
        s=16,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="True task")
    plt.xlabel(f"{reduction.upper()} dimension 1")
    plt.ylabel(f"{reduction.upper()} dimension 2")
    plt.title("Final DG representations colored by true task")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"dg_by_task_{reduction}.png"),
        dpi=200,
    )
    plt.close()

    # Plot by selected expert
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=expert_labels.detach().cpu().numpy(),
        cmap="tab10",
        s=16,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Selected expert")
    plt.xlabel(f"{reduction.upper()} dimension 1")
    plt.ylabel(f"{reduction.upper()} dimension 2")
    plt.title("Final DG representations colored by selected expert")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"dg_by_selected_expert_{reduction}.png"),
        dpi=200,
    )
    plt.close()

    print(f"Extracted DG shape: {tuple(dg_features.shape)}")
    print(f"Saved DG features and plots to: {save_dir}")

    return {
        "dg_features": dg_features,
        "class_labels": class_labels,
        "task_labels": task_labels,
        "selected_experts": expert_labels,
        "embedding_2d": embedded,
    }

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    pairwise_distances,
)


def analyze_dg_input_output_features(
    model,
    data_loader,
    device,
    save_dir,
    expert_id=0,
    max_samples=2000,
    num_images_to_show=16,
    reduction="pca",
    normalize_features=True,
    class_names=None,
    dataset_mean=(0.4914, 0.4822, 0.4465),
    dataset_std=(0.2470, 0.2435, 0.2616),
):
    """
    Compare:
        1. Original input images
        2. Flattened raw input values entering DG
        3. Final DG outputs leaving DG and entering CA3

    This version is suitable for a one-expert, two-class experiment.

    Args:
        model:
            Trained DGGatedHippocampalMoE model.

        data_loader:
            Loader containing samples for the selected expert/task.

        device:
            torch.device("cuda") or torch.device("cpu").

        save_dir:
            Directory for saved tensors, plots, and metrics.

        expert_id:
            Expert whose DG representations will be analyzed.

        max_samples:
            Maximum number of samples used in the feature analysis.

        num_images_to_show:
            Number of original input images shown in the image grid.

        reduction:
            "pca" or "tsne".

        normalize_features:
            L2-normalize each feature vector before computing metrics
            and dimensionality reduction.

        class_names:
            Optional dictionary or list mapping class IDs to names.

        dataset_mean, dataset_std:
            Normalization values used by the image dataset.
            Defaults are CIFAR-10 values. Change these if your
            dataloader uses different normalization.
    """
    os.makedirs(save_dir, exist_ok=True)

    if expert_id < 0 or expert_id >= model.num_experts:
        raise ValueError(
            f"expert_id={expert_id} is invalid for "
            f"{model.num_experts} expert(s)."
        )

    if reduction.lower() not in {"pca", "tsne"}:
        raise ValueError("reduction must be either 'pca' or 'tsne'")

    model.eval()

    original_inputs = []
    dg_input_features = []
    dg_output_features = []
    class_labels = []

    collected_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            remaining = max_samples - collected_samples

            if remaining <= 0:
                break

            if inputs.size(0) > remaining:
                inputs = inputs[:remaining]
                labels = labels[:remaining]

            # ---------------------------------------------------------
            # 1. Original inputs
            # ---------------------------------------------------------
            original_inputs.append(inputs.detach().cpu())

            # ---------------------------------------------------------
            # 2. Raw inputs entering DG
            # ---------------------------------------------------------
            # Raw flattened pixels now enter DG directly.
            features_entering_dg = model.prepare_dg_input(inputs)

            # ---------------------------------------------------------
            # 3. Features leaving DG
            # ---------------------------------------------------------
            # CustomEnhancedHippocampalExpert returns:
            #   dg_output, ca1_output
            #
            # Internally:
            #   dg_output = expert.dg(features_entering_dg)
            #   ca3_output = expert.ca3(dg_output)
            #
            # Therefore, dg_output is exactly what enters CA3.
            dg_output, _ = model.hippocampal_experts[expert_id](
                features_entering_dg
            )

            dg_input_features.append(features_entering_dg.detach().cpu())
            dg_output_features.append(dg_output.detach().cpu())
            class_labels.append(labels.detach().cpu())

            collected_samples += inputs.size(0)

    if collected_samples == 0:
        raise RuntimeError("The supplied data loader produced no samples.")

    original_inputs = torch.cat(original_inputs, dim=0)
    dg_input_features = torch.cat(dg_input_features, dim=0)
    dg_output_features = torch.cat(dg_output_features, dim=0)
    class_labels = torch.cat(class_labels, dim=0)

    print("=" * 70)
    print("DG FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Samples:                  {len(class_labels)}")
    print(f"Original input shape:     {tuple(original_inputs.shape)}")
    print(f"Raw inputs entering DG:     {tuple(dg_input_features.shape)}")
    print(f"Features leaving DG:      {tuple(dg_output_features.shape)}")
    print(f"Unique class labels:      {torch.unique(class_labels).tolist()}")
    print("=" * 70)

    # Save the full, unreduced tensors
    tensor_path = os.path.join(save_dir, "dg_feature_comparison.pt")

    torch.save(
        {
            "original_inputs": original_inputs,
            "dg_input_features": dg_input_features,
            "dg_output_features": dg_output_features,
            "class_labels": class_labels,
            "expert_id": expert_id,
        },
        tensor_path,
    )

    # =============================================================
    # Visualize original images
    # =============================================================
    display_count = min(num_images_to_show, len(original_inputs))

    mean = torch.tensor(dataset_mean).view(3, 1, 1)
    std = torch.tensor(dataset_std).view(3, 1, 1)

    # Reverse dataset normalization for display
    display_images = original_inputs[:display_count].clone()

    if display_images.size(1) == 3:
        display_images = display_images * std + mean

    display_images = display_images.clamp(0, 1)

    grid_cols = min(4, display_count)
    grid_rows = int(np.ceil(display_count / grid_cols))

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(3 * grid_cols, 3 * grid_rows),
    )

    axes = np.array(axes).reshape(-1)

    for index in range(len(axes)):
        axes[index].axis("off")

        if index >= display_count:
            continue

        image = display_images[index]

        if image.size(0) == 1:
            axes[index].imshow(
                image.squeeze(0).numpy(),
                cmap="gray",
            )
        else:
            axes[index].imshow(
                image.permute(1, 2, 0).numpy()
            )

        label_id = int(class_labels[index].item())

        if class_names is None:
            label_text = f"Class {label_id}"
        elif isinstance(class_names, dict):
            label_text = class_names.get(label_id, f"Class {label_id}")
        else:
            if 0 <= label_id < len(class_names):
                label_text = class_names[label_id]
            else:
                label_text = f"Class {label_id}"

        axes[index].set_title(label_text)

    fig.suptitle(
        f"Original inputs used by Expert {expert_id}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "original_input_images.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # =============================================================
    # Prepare feature matrices
    # =============================================================
    dg_input_analysis = dg_input_features.float()
    dg_output_analysis = dg_output_features.float()

    if normalize_features:
        dg_input_analysis = F.normalize(
            dg_input_analysis,
            p=2,
            dim=1,
        )

        dg_output_analysis = F.normalize(
            dg_output_analysis,
            p=2,
            dim=1,
        )

    dg_input_np = dg_input_analysis.numpy()
    dg_output_np = dg_output_analysis.numpy()
    labels_np = class_labels.detach().cpu().numpy()

    # =============================================================
    # Dimensionality reduction
    # =============================================================
    def reduce_to_2d(features, method):
        if method == "pca":
            reducer = PCA(
                n_components=2,
                random_state=42,
            )
            return reducer.fit_transform(features)

        # Reduce very high-dimensional features before t-SNE
        intermediate_dimensions = min(
            50,
            features.shape[1],
            features.shape[0] - 1,
        )

        if intermediate_dimensions < 2:
            raise RuntimeError(
                "Not enough samples to perform t-SNE."
            )

        features_pca = PCA(
            n_components=intermediate_dimensions,
            random_state=42,
        ).fit_transform(features)

        # Perplexity must be smaller than number of samples
        perplexity = min(
            30,
            max(5, features.shape[0] // 10),
            features.shape[0] - 1,
        )

        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(features_pca)

    input_embedding = reduce_to_2d(
        dg_input_np,
        reduction.lower(),
    )

    output_embedding = reduce_to_2d(
        dg_output_np,
        reduction.lower(),
    )

    # Save reduced coordinates
    np.savez(
        os.path.join(
            save_dir,
            f"dg_feature_embeddings_{reduction.lower()}.npz",
        ),
        input_embedding=input_embedding,
        output_embedding=output_embedding,
        labels=labels_np,
    )

    # =============================================================
    # Side-by-side feature visualization
    # =============================================================
    unique_classes = np.unique(labels_np)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    for class_id in unique_classes:
        class_mask = labels_np == class_id

        if class_names is None:
            class_label = f"Class {class_id}"
        elif isinstance(class_names, dict):
            class_label = class_names.get(
                int(class_id),
                f"Class {class_id}",
            )
        else:
            if 0 <= int(class_id) < len(class_names):
                class_label = class_names[int(class_id)]
            else:
                class_label = f"Class {class_id}"

        axes[0].scatter(
            input_embedding[class_mask, 0],
            input_embedding[class_mask, 1],
            s=18,
            alpha=0.7,
            label=class_label,
        )

        axes[1].scatter(
            output_embedding[class_mask, 0],
            output_embedding[class_mask, 1],
            s=18,
            alpha=0.7,
            label=class_label,
        )

    method_name = reduction.upper()

    axes[0].set_title(
        f"Raw inputs entering DG\n({dg_input_features.shape[1]} dimensions)"
    )
    axes[0].set_xlabel(f"{method_name} dimension 1")
    axes[0].set_ylabel(f"{method_name} dimension 2")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].set_title(
        f"Features leaving DG and entering CA3\n"
        f"({dg_output_features.shape[1]} dimensions)"
    )
    axes[1].set_xlabel(f"{method_name} dimension 1")
    axes[1].set_ylabel(f"{method_name} dimension 2")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"DG input versus DG output — Expert {expert_id}",
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            f"dg_input_vs_output_{reduction.lower()}.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # =============================================================
    # Compute separation metrics
    # =============================================================
    def calculate_feature_metrics(features, labels):
        unique = np.unique(labels)

        # Silhouette and Davies-Bouldin need at least two classes
        if len(unique) < 2:
            return {
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "mean_intra_class_distance": None,
                "mean_inter_class_distance": None,
                "separation_ratio": None,
            }

        # Need at least two samples in every class
        class_counts = [
            np.sum(labels == class_id)
            for class_id in unique
        ]

        if min(class_counts) < 2:
            return {
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "mean_intra_class_distance": None,
                "mean_inter_class_distance": None,
                "separation_ratio": None,
            }

        silhouette = silhouette_score(
            features,
            labels,
            metric="euclidean",
        )

        db_index = davies_bouldin_score(
            features,
            labels,
        )

        distances = pairwise_distances(
            features,
            metric="euclidean",
        )

        same_class = labels[:, None] == labels[None, :]
        different_class = ~same_class

        # Exclude diagonal self-distances
        diagonal = np.eye(len(labels), dtype=bool)
        same_class = same_class & ~diagonal

        intra_distances = distances[same_class]
        inter_distances = distances[different_class]

        mean_intra = (
            float(intra_distances.mean())
            if len(intra_distances) > 0
            else None
        )

        mean_inter = (
            float(inter_distances.mean())
            if len(inter_distances) > 0
            else None
        )

        if (
            mean_intra is not None
            and mean_inter is not None
            and mean_intra > 0
        ):
            separation_ratio = mean_inter / mean_intra
        else:
            separation_ratio = None

        return {
            "silhouette_score": float(silhouette),
            "davies_bouldin_index": float(db_index),
            "mean_intra_class_distance": mean_intra,
            "mean_inter_class_distance": mean_inter,
            "separation_ratio": separation_ratio,
        }

    input_metrics = calculate_feature_metrics(
        dg_input_np,
        labels_np,
    )

    output_metrics = calculate_feature_metrics(
        dg_output_np,
        labels_np,
    )

    metrics = {
        "expert_id": expert_id,
        "num_samples": int(len(labels_np)),
        "classes": [int(x) for x in unique_classes],
        "normalization_used": normalize_features,
        "features_entering_dg_dimension": int(
            dg_input_features.shape[1]
        ),
        "features_leaving_dg_dimension": int(
            dg_output_features.shape[1]
        ),
        "dg_input_metrics": input_metrics,
        "dg_output_metrics": output_metrics,
    }

    # Calculate explicit changes
    if (
        input_metrics["silhouette_score"] is not None
        and output_metrics["silhouette_score"] is not None
    ):
        metrics["silhouette_change"] = (
            output_metrics["silhouette_score"]
            - input_metrics["silhouette_score"]
        )

    if (
        input_metrics["davies_bouldin_index"] is not None
        and output_metrics["davies_bouldin_index"] is not None
    ):
        # Positive means improvement because lower DB is better
        metrics["davies_bouldin_improvement"] = (
            input_metrics["davies_bouldin_index"]
            - output_metrics["davies_bouldin_index"]
        )

    if (
        input_metrics["separation_ratio"] is not None
        and output_metrics["separation_ratio"] is not None
    ):
        metrics["separation_ratio_change"] = (
            output_metrics["separation_ratio"]
            - input_metrics["separation_ratio"]
        )

    metrics_path = os.path.join(
        save_dir,
        "dg_input_output_metrics.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # =============================================================
    # Plot metric comparison
    # =============================================================
    metric_names = []
    input_values = []
    output_values = []

    possible_metrics = [
        (
            "Silhouette\n(higher better)",
            "silhouette_score",
        ),
        (
            "Davies-Bouldin\n(lower better)",
            "davies_bouldin_index",
        ),
        (
            "Inter/Intra ratio\n(higher better)",
            "separation_ratio",
        ),
    ]

    for display_name, key in possible_metrics:
        input_value = input_metrics[key]
        output_value = output_metrics[key]

        if input_value is not None and output_value is not None:
            metric_names.append(display_name)
            input_values.append(input_value)
            output_values.append(output_value)

    if metric_names:
        positions = np.arange(len(metric_names))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(
            positions - width / 2,
            input_values,
            width,
            label="Entering DG",
        )
        plt.bar(
            positions + width / 2,
            output_values,
            width,
            label="Leaving DG",
        )

        plt.xticks(positions, metric_names)
        plt.ylabel("Metric value")
        plt.title(
            f"Class-separation metrics before and after DG — "
            f"Expert {expert_id}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_dir,
                "dg_input_output_metric_comparison.png",
            ),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # =============================================================
    # Print results
    # =============================================================
    print("\nCLASS-SEPARATION COMPARISON")
    print("-" * 70)

    print("Raw inputs entering DG:")
    for key, value in input_metrics.items():
        print(f"  {key}: {value}")

    print("\nFeatures leaving DG / entering CA3:")
    for key, value in output_metrics.items():
        print(f"  {key}: {value}")

    print("\nChanges:")
    print(
        "  Silhouette change:",
        metrics.get("silhouette_change"),
    )
    print(
        "  Davies-Bouldin improvement:",
        metrics.get("davies_bouldin_improvement"),
    )
    print(
        "  Separation-ratio change:",
        metrics.get("separation_ratio_change"),
    )

    print("\nSaved:")
    print(f"  Raw tensors:       {tensor_path}")
    print(f"  Metrics:           {metrics_path}")
    print(f"  Output directory:  {save_dir}")

    return {
        "original_inputs": original_inputs,
        "dg_input_features": dg_input_features,
        "dg_output_features": dg_output_features,
        "class_labels": class_labels,
        "input_embedding": input_embedding,
        "output_embedding": output_embedding,
        "metrics": metrics,
    }

def main():
    """Train model + Analyze DG-Gated model"""
    args = parse_arguments()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = setup_logging()
    
    logging.info("🚀 TRAIN + ANALYZE DG-GATED MODEL (2-Phase Contrastive)")
    logging.info("=" * 80)
    
    if args.num_experts != 1 or args.classes_per_task != 2:
        raise ValueError(
            "The custom RGB configuration in this file is designed for "
            "--num_experts 1 and --classes_per_task 2."
        )

    train_loaders, test_loaders, task_classes, class_names = (
        create_custom_rgb_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            test_split=args.test_split,
            num_workers=args.num_workers,
            seed=args.seed,
            normalize_mean=args.normalize_mean,
            normalize_std=args.normalize_std,
            disable_normalization=args.disable_normalization,
        )
    )

    if args.disable_normalization:
        display_mean = (0.0, 0.0, 0.0)
        display_std = (1.0, 1.0, 1.0)
    else:
        display_mean = tuple(args.normalize_mean)
        display_std = tuple(args.normalize_std)
    
    model = DGGatedHippocampalMoE(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        input_channels=3,
        target_sparsity=args.dg_sparsity,
        memory_size=args.memory_size,
        use_small_features=args.use_small_features,
        image_size=args.image_size
    ).to(device)
    
    # Set dropout rate for all experts
    for expert in model.hippocampal_experts:
        # Find dropout layers by type instead of index
        if hasattr(expert, 'ca1_integration') and hasattr(expert.ca1_integration, 'modules'):
            for module in expert.ca1_integration.modules():
                if isinstance(module, nn.Dropout):
                    module.p = args.dropout_rate
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters (DG-Gated): {trainable_params:,}")
    
    model.set_task_classes(task_classes)
    model.set_gating_temperature(args.gating_temperature)
    model.set_gating_strategy(args.gating_strategy)
    
    # === PHASE 1: Independent Expert Training with Online EMA Prototypes ===
    phase1_train_experts_sequentially(model, train_loaders, test_loaders, device, args,log_dir)

    analysis_results = analyze_dg_input_output_features(
    model=model,
    data_loader=test_loaders[0],
    device=device,
    save_dir=os.path.join(
        log_dir,
        "expert0_dg_feature_comparison",
    ),
    expert_id=0,
    max_samples=2000,
    num_images_to_show=16,
    reduction="pca",
    normalize_features=True,
    class_names=class_names,
    dataset_mean=display_mean,
    dataset_std=display_std,
)
    analysis_results_tsne = analyze_dg_input_output_features(
    model=model,
    data_loader=test_loaders[0],
    device=device,
    save_dir=os.path.join(
        log_dir,
        "expert0_dg_feature_comparison_tsne",
    ),
    expert_id=0,
    max_samples=2000,
    num_images_to_show=16,
    reduction="tsne",
    normalize_features=True,
    class_names=class_names,
    dataset_mean=display_mean,
    dataset_std=display_std,
)
    model.eval()

    dg_outputs = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in test_loaders[0]:
            inputs = inputs.to(device)

            features = model.prepare_dg_input(inputs)

            # This dg_output is passed directly into CA3
            dg_output, ca1_output = model.hippocampal_experts[0](features)

            dg_outputs.append(dg_output.cpu())
            labels_all.append(labels.cpu())

    dg_outputs = torch.cat(dg_outputs)
    labels_all = torch.cat(labels_all)

    print(dg_outputs.shape)


    dg_2d = PCA(n_components=2).fit_transform(dg_outputs.numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        dg_2d[:, 0],
        dg_2d[:, 1],
        c=labels_all.numpy(),
        cmap="tab10",
        alpha=0.7,
    )

    plt.colorbar(scatter, label="Class")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Expert 0 final DG representations")
    plt.tight_layout()
    plt.show()

    dg_results = extract_and_visualize_dg_features(
    model=model,
    data_loaders=test_loaders,
    device=device,
    save_dir=os.path.join(log_dir, "dg_feature_visualization_correct_expert"),
    max_samples=2000,
    reduction="tsne",
    use_correct_expert=True,
)
    dg_results = extract_and_visualize_dg_features(
        model=model,
        data_loaders=test_loaders,
        device=device,
        save_dir=os.path.join(log_dir, "dg_feature_visualization"),
        max_samples=2000,
        reduction="tsne",
        use_correct_expert=False,
    )
    # Prototypes are now computed online during Phase 1 using EMAs
    logging.info("✅ Online EMA prototypes computed during Phase 1 training")
    
    # === POST PHASE 1 DIAGNOSTICS ===
    # logging.info("\n" + "🔬" * 60)
    logging.info(" POST PHASE 1 DG PATTERN SEPARATION DIAGNOSTIC")
    # logging.info("🔬" * 60)
    phase1_results = analyze_dg_pattern_separation(model, test_loaders, device, log_dir)
    
    # Save Phase 1 prototype plots
    if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None:
        try:
            # Convert prototypes to numpy for plotting
            proto_array = []
            for expert_id in range(model.num_experts):
                proto_array.append(
                    model.dg_prototypes[expert_id].detach().cpu().numpy()
                )
            
            proto_array = np.array(proto_array)
            plot_prototype_stats(proto_array, phase='phase1', save_dir=log_dir, logger=logging)
            logging.info("✅ Phase 1 prototype plots saved")
        except Exception as e:
            logging.warning(f"⚠️ Failed to save Phase 1 prototype plots: {e}")

    # === PHASE 2: Joint Contrastive Fine-Tuning ===
    # Phase 2 separates multiple experts from one another. With one expert,
    # there are no negative expert/prototype pairs, so the phase is skipped.
    phase2_results = None
    if model.num_experts > 1 and args.contrastive_epochs > 0:
        phase2_contrastive_tuning(
            model,
            train_loaders,
            device,
            args,
            log_dir,
        )

        model.update_dg_prototypes_from_ema()
        logging.info("✅ Final prototype update after Phase 2 complete")
        model.log_prototype_stats()

        logging.info("\n" + "🔬" * 60)
        logging.info("🔬 POST PHASE 2 DG PATTERN SEPARATION DIAGNOSTIC")
        logging.info("🔬" * 60)
        phase2_results = analyze_dg_pattern_separation(
            model,
            test_loaders,
            device,
            log_dir,
        )
    else:
        logging.info(
            "ℹ️ Skipping Phase 2 expert-contrastive tuning: the current "
            "experiment has one expert, so expert-to-expert separation is "
            "not defined. Class separation is still measured before and "
            "after the DG by analyze_dg_input_output_features()."
        )

    # === ANALYSIS ===
    final_results = evaluate_dg_gated_model_standardized(
        model,
        test_loaders,
        task_classes,
        device,
    )
    dg_analysis = analyze_dg_gated_model(
        model,
        test_loaders,
        device,
        log_dir,
    )

    if model.num_experts > 1:
        deep_dive_analysis = analyze_dg_deep_dive(
            model,
            test_loaders,
            device,
            log_dir,
        )
    else:
        deep_dive_analysis = None
        logging.info(
            "ℹ️ Skipping the multi-expert intra/inter-task deep-dive plot "
            "because only one expert/task is present."
        )

    # === PHASE COMPARISON SUMMARY ===
    if phase2_results is not None:
        logging.info("\n" + "📊" * 60)
        logging.info("📊 PHASE 1 vs PHASE 2 DG SEPARATION COMPARISON")
        logging.info("📊" * 60)

        if (
            'silhouette_score' in phase1_results
            and 'silhouette_score' in phase2_results
        ):
            sil_improvement = (
                phase2_results['silhouette_score']
                - phase1_results['silhouette_score']
            )
            db_improvement = (
                phase1_results['davies_bouldin_index']
                - phase2_results['davies_bouldin_index']
            )
            ratio_improvement = (
                phase2_results['separation_ratio']
                - phase1_results['separation_ratio']
            )

            logging.info(
                f"📈 Silhouette Score: "
                f"{phase1_results['silhouette_score']:.4f} → "
                f"{phase2_results['silhouette_score']:.4f} "
                f"(Δ: {sil_improvement:+.4f})"
            )
            logging.info(
                f"📈 Davies-Bouldin Index: "
                f"{phase1_results['davies_bouldin_index']:.4f} → "
                f"{phase2_results['davies_bouldin_index']:.4f} "
                f"(Δ: {db_improvement:+.4f})"
            )
            logging.info(
                f"📈 Separation Ratio: "
                f"{phase1_results['separation_ratio']:.4f} → "
                f"{phase2_results['separation_ratio']:.4f} "
                f"(Δ: {ratio_improvement:+.4f})"
            )

        logging.info("📊" * 60)

    # === PROTOTYPE DRIFT ANALYSIS ===
    if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None:
        try:
            # Get Phase 1 prototypes (we need to save them earlier)
            # For now, we'll compute the drift using current prototypes vs a baseline
            proto_array_phase2 = []
            for expert_id in range(model.num_experts):
                proto_array_phase2.append(
                    model.dg_prototypes[expert_id].detach().cpu().numpy()
                )
            
            proto_array_phase2 = np.array(proto_array_phase2)
            
            # Create a baseline (random initialization) for comparison
            baseline_protos = np.random.randn(proto_array_phase2.shape[0], proto_array_phase2.shape[1]) * 0.1
            baseline_protos = baseline_protos / np.linalg.norm(baseline_protos, axis=1, keepdims=True)
            
            log_and_plot_prototype_drift(baseline_protos, proto_array_phase2, 'baseline', 'phase2', log_dir, logging)
            logging.info("✅ Prototype drift analysis completed")
        except Exception as e:
            logging.warning(f"⚠️ Failed to analyze prototype drift: {e}")
    
    # === CLASS-IL BREAKDOWN ANALYSIS ===
    analyze_class_il_breakdown(model, test_loaders, device, log_dir)
    # === TASK-IL BREAKDOWN ANALYSIS ===
    analyze_task_il_breakdown(model, test_loaders, device, log_dir)
    # === REPORTING ===
    report_text = f"""
# 🧠 DG-Gated Hippocampal MoE Analysis Report

## Key Findings:
- **Routing Accuracy (DG Similarity):** {dg_analysis['routing_accuracy']:.1%}
- **Expert Utilization:** { {i: f'{u:.1%}' for i, u in enumerate(dg_analysis['expert_utilization'])} }

## Status:
{'✅ EXCELLENT ROUTING' if dg_analysis['routing_accuracy'] > 0.8 else '⚠️ POOR ROUTING' if dg_analysis['routing_accuracy'] < 0.5 else '✅ GOOD ROUTING'}

This analysis uses gating based on DG pattern similarity, not a separate gating network.
    """
    
    with open(f'{log_dir}/DG_Gated_Analysis_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logging.info(f"\n🎉 TRAINING + ANALYSIS OF DG-GATED MODEL COMPLETE!")
    logging.info(f"💾 Results saved in: {log_dir}/")
    logging.info(f"📊 Analysis plots: {log_dir}/dg_gated_analysis/")
    logging.info(f"📊 Deep-dive plots: {log_dir}/DG_Deep_Dive_Analysis.png")
    logging.info(f"📋 Report: {log_dir}/DG_Gated_Analysis_Report.md")

    train_class_counts = {}
    test_class_counts = {}
    for expert_id, loader in enumerate(train_loaders):
        for _, labels in loader:
            for l in labels.detach().cpu().numpy():
                train_class_counts[l] = train_class_counts.get(l, 0) + 1
    for expert_id, loader in enumerate(test_loaders):
        for _, labels in loader:
            for l in labels.detach().cpu().numpy():
                test_class_counts[l] = test_class_counts.get(l, 0) + 1
    logging.info("=== TRAIN CLASS COUNTS ===\n" + str(train_class_counts))
    logging.info("=== TEST CLASS COUNTS ===\n" + str(test_class_counts))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and analyze DG-Gated Hippocampal MoE.")
    parser.add_argument('--num_experts', type=int, default=1)
    parser.add_argument('--classes_per_task', type=int, default=2)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing the two image classes, or train/test subfolders.')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Square RGB size used by the raw-input DG model. Default: 32.')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test fraction used only when train/test folders are not supplied.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--normalize_mean', type=float, nargs=3,
                        default=(0.5, 0.5, 0.5), metavar=('R', 'G', 'B'))
    parser.add_argument('--normalize_std', type=float, nargs=3,
                        default=(0.5, 0.5, 0.5), metavar=('R', 'G', 'B'))
    parser.add_argument('--disable_normalization', action='store_true',
                        help='Use image values in [0, 1] without channel normalization.')
    parser.add_argument('--expert_epochs', type=int, default=12)
    parser.add_argument('--gate_epochs', type=int, default=20) # Not used, but kept for compatibility
    parser.add_argument('--joint_epochs', type=int, default=8) # Not used
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dg_sparsity', type=float, default=0.03)  # Lowered from 0.05 to 0.03 for sparser DG
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--memory_size', type=int, default=200)
    parser.add_argument('--gating_temperature', type=float, default=0.03)  # Lowered from 0.07 to 0.03 for sharper gating
    parser.add_argument('--gating_strategy', type=str, default='soft', choices=['soft', 'hard', 'top2', 'soft_hard'])  # Default to soft gating only
    parser.add_argument('--distillation_coef', type=float, default=0.1)
    parser.add_argument('--feature_distillation_coef', type=float, default=0.05)
    parser.add_argument('--replay_loss_coef', type=float, default=0.1)
    parser.add_argument('--contrastive_epochs', type=int, default=0, help='Ignored for one expert; expert-contrastive Phase 2 requires at least two experts.')
    parser.add_argument('--contrastive_lr', type=float, default=1e-4)
    parser.add_argument('--contrastive_margin', type=float, default=0.6)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--use_class_balanced_loss', action='store_true', help='Enable class-balanced loss weighting')
    parser.add_argument('--class_balance_epsilon', type=float, default=0.1, help='Small constant to prevent division by zero in class weights')
    parser.add_argument('--class_balance_smoothing', type=float, default=0.1, help='Smoothing factor for class weights to prevent instability')
    parser.add_argument('--use_routing_confidence_penalty', action='store_true', help='Enable routing confidence penalty to encourage decisive gating')
    parser.add_argument('--routing_confidence_coef', type=float, default=0.05, help='Coefficient for routing confidence penalty')
    parser.add_argument('--use_prototype_regularization', action='store_true', help='Enable prototype regularization to anchor DG prototypes to class centers')
    parser.add_argument('--prototype_reg_coef', type=float, default=0.1, help='Coefficient for prototype regularization loss')
    parser.add_argument('--ewc_lambda', type=float, default=1000.0, help='EWC regularization strength to prevent forgetting')
    parser.add_argument('--use_small_features', action='store_true', help='Deprecated compatibility flag; ignored because the feature extractor is disabled')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    main() 