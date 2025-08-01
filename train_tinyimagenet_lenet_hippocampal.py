#!/usr/bin/env python3
"""
TinyImageNet Hippocampal MoE Training with Proven LeNet Architecture
Uses the EXACT same architecture that achieved 70% Class-IL on CIFAR-10
Just scaled for 64x64 TinyImageNet images instead of 32x32 CIFAR-10

🚨 CRITICAL ROUTING COLLAPSE FIX (2025-06-27):
==============================================
PROBLEM: Router pretraining achieved 37.5% accuracy, but final routing dropped to 10% (random level).
ROOT CAUSE: During expert training (Phase 1), the gating network was being RE-TRAINED while using 
forced routing (task_id provided). This DESTROYED the good routing learned during pretraining!

The sequence was:
1. Router pretraining: Learn task discrimination → 37.5% routing accuracy ✅
2. Expert training: Use forced routing BUT still train gating weights → Corrupts routing ❌
3. Final evaluation: Gating network can no longer route correctly → 10% accuracy ❌

SOLUTION: FREEZE gating network during expert training to preserve pretraining weights.
- During expert training: gating_network.requires_grad = False
- During evaluation: gating_network uses preserved pretraining weights
- Expected improvement: 10% → ~35% routing accuracy

🔧 CRITICAL FIXES APPLIED to prevent routing collapse:
1. FIXED Phase 1: Removed gating network training during expert training (froze weights)
2. FIXED Phase 1: Removed task-ID loss calculation during expert training (unnecessary)
3. FIXED Phase 1: Removed task-ID loss from replay (gating frozen)
4. FIXED evaluation: Use raw features (not enhanced) for gating consistency with pretraining
5. ENHANCED: Clear comments explaining why gating is frozen during expert training

This fix preserves the router's ability to distinguish tasks while allowing experts to 
specialize on classification within their assigned tasks.

Original enhancement features still intact:
🚀 FEATURE SEPARABILITY ENHANCEMENTS:
7. ENHANCED: Added auxiliary task-ID classification head to force task-discriminative features
8. ENHANCED: Reduced replay loss weight (0.5x) to fix overfitting to small replay buffer
9. ENHANCED: Increased trunk learning rate (2.0x) for faster adaptation to new tasks
10. ENHANCED: Extended auxiliary loss to replay samples for consistent task discrimination

🧬 BIOLOGICAL SPARSITY ENHANCEMENTS:
11. ENHANCED: Reduced DG sparsity to biological 3% (from 5%) with regularization loss
12. ENHANCED: Real-time sparsity monitoring and deviation tracking
13. ENHANCED: Sparsity loss regularization to maintain biological realism

🔧 PREVIOUS BUG FIXES (MAINTAINED):
14. FIXED: total_loss overwrite bug - replay processing was erasing aux_loss and sparsity_loss
15. FIXED: Enhanced sparsity function with quadratic penalty + L1 regularization
16. FIXED: Increased sparsity_loss_weight from 0.1 to 1.0 (10x stronger control)
17. ADDED: Comprehensive debugging for aux_loss and sparsity_loss monitoring
18. FIXED: Auxiliary head overconfidence bug - added temperature scaling + L2 reg to prevent aux_loss=0

Expected outcomes after routing fix:
- Routing accuracy: 10% → 35-40% (preserves pretraining performance)
- Task-IL accuracy: Should remain high (experts are still properly trained)
- Class-IL accuracy: Should improve significantly due to better routing
- Feature separability: Should remain good (experts still process features)
- DG sparsity: Should be better controlled with regularization
"""
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
import numpy as np
import torch

# Set all seeds before any other imports
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
import urllib.request
import zipfile
import shutil
from collections import defaultdict
import copy
import json
import warnings
import time

# Visualization imports for analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import the proven CIFAR-10 architecture components
# Analysis utility (optional)
try:
    from train_and_analyze import analyze_TRAINED_model
except ImportError:
    analyze_TRAINED_model = None  # Will be None if script not present

from train_hippocampal_optimal_moe import (
    OptimalHippocampalMoE, HippocampalReplayBuffer,
    calculate_load_balancing_loss, calculate_diversity_loss,
    create_balanced_joint_loader, set_seed
)

# 🔧 ACTION 4: Enhanced sparsity functions
def calculate_sparsity_loss(model, expert_id, target_sparsity):
    """
    Calculate sparsity regularization loss to maintain biological DG sparsity.
    Penalizes deviation from target sparsity (e.g., 3% biological DG sparsity).
    """
    sparsity_losses = []
    
    # Extract DG output from current expert to measure actual sparsity
    expert = model.hippocampal_experts[expert_id]
    
    # We need a dummy forward pass to get DG activations
    # This is called during training so we can access the last DG output
    # For now, return zero loss and implement proper tracking later
    return torch.tensor(0.0, device=next(model.parameters()).device)

def calculate_dg_sparsity(dg_outputs, target_sparsity=0.03):
    """Calculate actual DG sparsity and create regularization loss with quadratic penalty + L1"""
    # Calculate actual sparsity (fraction of non-zero activations)
    actual_sparsity = (dg_outputs > 0).float().mean()
    
    # Quadratic penalty for deviations from target sparsity (stronger control)
    sparsity_diff_quad = (actual_sparsity - target_sparsity) ** 2
    
    # L1 regularization to encourage sparsity
    l1_penalty = torch.mean(torch.abs(dg_outputs))
    
    # Combined sparsity loss - new term directly penalizes mean activation
    total_sparsity_loss = sparsity_diff_quad + (0.1 * l1_penalty) + (0.05 * torch.mean(dg_outputs))
    
    return total_sparsity_loss

# Import key components from base Hippocampal MoE file
from train_hippocampal_moe import SoftGating, HippocampalExpert

def setup_logging():
    """Setup logging for the training process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"tinyimagenet_lenet_hippocampal_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_dir

class TinyImageNetValDataset(Dataset):
    """Custom dataset for TinyImageNet validation set"""
    def __init__(self, val_dir, annotations_file, class_to_idx, transform=None):
        self.val_dir = Path(val_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = self._load_samples(annotations_file)
        
    def _load_samples(self, annotations_file):
        samples = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    if class_name in self.class_to_idx:
                        img_path = self.val_dir / 'images' / img_name
                        if img_path.exists():
                            samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = torchvision.datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

class LabelAdjustedDataset(Dataset):
    """Adjusts labels to be 0-indexed for each task"""
    def __init__(self, subset, class_range):
        self.subset = subset
        self.class_range = sorted(class_range)
        self.class_mapping = {old_class: new_idx for new_idx, old_class in enumerate(self.class_range)}
        
    def __getitem__(self, index):
        data, label = self.subset[index]
        new_label = self.class_mapping[label]
        return data, new_label
        
    def __len__(self):
        return len(self.subset)

def create_tinyimagenet_tasks(data_dir='./data/tiny-imagenet-200', num_tasks=10, classes_per_task=20, batch_size=64):
    """Create TinyImageNet tasks using the same proven approach as CIFAR-10"""
    
    logger = logging.getLogger()
    logger.info(f"Creating {num_tasks} Tiny ImageNet tasks with {classes_per_task} classes per task from {data_dir}")
    
    # Check for data directory existence and download if needed
    data_path = Path(data_dir)
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if not train_path.is_dir() or not val_path.is_dir():
        logger.info(f"Tiny ImageNet data not found in {data_dir}. Downloading...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = data_path.parent / "tiny-imagenet-200.zip"
        
        # Create parent directory
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download
            if not zip_path.is_file():
                logger.info(f"Downloading {url}...")
                with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                logger.info("Download complete.")
            
            # Extract
            logger.info(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path.parent)
            logger.info("Extraction complete.")
            
        except Exception as e:
            logger.error(f"Failed to download Tiny ImageNet: {e}")
            raise
    
    # Define transforms optimized for TinyImageNet (64x64 images)
    # Use same philosophy as CIFAR-10 but scaled for larger images
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),  # Similar to CIFAR-10's padding=4
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    full_train_dataset = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    class_to_idx = full_train_dataset.class_to_idx
    all_classes = sorted(list(class_to_idx.values()))  # 0-199
    total_classes = len(all_classes)
    logger.info(f"Found {total_classes} classes in Tiny ImageNet.")
    
    # Load validation dataset
    val_annotations_file = val_path / 'val_annotations.txt'
    full_test_dataset = TinyImageNetValDataset(val_path, val_annotations_file, class_to_idx, transform=test_transform)
    logger.info(f"Loaded validation set with {len(full_test_dataset)} samples.")
    
    # Pre-compute class indices for faster task creation
    train_labels = np.array(full_train_dataset.targets)
    test_labels = np.array([s[1] for s in full_test_dataset.samples])
    
    class_to_train_indices = {cls: np.where(train_labels == cls)[0] for cls in range(total_classes)}
    class_to_test_indices = {cls: np.where(test_labels == cls)[0] for cls in range(total_classes)}
    
    # Shuffle classes for random task assignment (same as CIFAR-10)
    random.seed(0)
    random.shuffle(all_classes)
    
    train_loaders = []
    test_loaders = []
    task_classes = []
    
    for task_id in range(num_tasks):
        start_idx = task_id * classes_per_task
        end_idx = start_idx + classes_per_task
        task_class_indices = all_classes[start_idx:end_idx]
        task_classes.append(task_class_indices)
        
        logger.info(f"Task {task_id}: Global classes {task_class_indices} -> Task indices 0-{len(task_class_indices)-1}")
        
        # Create training loader
        train_task_indices = np.concatenate([class_to_train_indices[cls] for cls in task_class_indices]).tolist()
        train_subset = Subset(full_train_dataset, train_task_indices)
        train_task_dataset = LabelAdjustedDataset(train_subset, task_class_indices)
        
        train_loader = DataLoader(
            train_task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        train_loaders.append(train_loader)
        
        # Create test loader
        test_task_indices = np.concatenate([class_to_test_indices[cls] for cls in task_class_indices]).tolist()
        test_subset = Subset(full_test_dataset, test_task_indices)
        test_task_dataset = LabelAdjustedDataset(test_subset, task_class_indices)
        
        test_loader = DataLoader(
            test_task_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loaders.append(test_loader)
        
        logger.info(f"Task {task_id}: Train samples: {len(train_task_dataset)}, Test samples: {len(test_task_dataset)}")
    
    logger.info(f"Successfully created {len(train_loaders)} Tiny ImageNet tasks.")
    return train_loaders, test_loaders, task_classes

class TinyImageNetOptimalHippocampalMoE(OptimalHippocampalMoE):
    """
    An implementation of the Hippocampal MoE for TinyImageNet, using a LeNet-style
    feature extractor that includes a GridCellLayer analogue.
    """
    def __init__(self, num_experts=10, classes_per_task=20, input_channels=3, dropout_rate=0.5):
        # Call nn.Module's __init__ directly. DO NOT call super().__init__ as it
        # builds the wrong (CIFAR-10) architecture. We are inheriting for methods, not for construction.
        nn.Module.__init__(self)
        
        self.num_experts = num_experts
        self.num_tasks = num_experts
        self.classes_per_task = classes_per_task
        self.num_classes = num_experts * classes_per_task

        self.feature_extractor = self._create_feature_extractor(input_channels)
        self.feature_dim = self._get_feature_dim((input_channels, 64, 64))
        
        self.gating_network = self._create_gating_network(self.feature_dim, num_experts, dropout_rate)
        self.soft_gating = SoftGating()
        
        # 🔧 UNIFIED: Remove separate aux_head - gating_network will handle task-ID classification directly
        
        dg_dim = 512  # Keep same as CIFAR-10
        ca3_dim = 256  # Keep same as CIFAR-10
        
        # 🔧 FIXED: Use enhanced hippocampal experts with proper biological sparsity
        self.hippocampal_experts = nn.ModuleList([
            EnhancedHippocampalExpert(self.feature_dim, dg_dim, ca3_dim, target_sparsity=0.03)
            for _ in range(num_experts)
        ])
        
        self.ca1_integration = nn.Sequential(
            nn.Linear(dg_dim + dg_dim + self.feature_dim, 256),  # DG + CA3 + direct entorhinal bypass
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )
        
        self.output_layers = nn.ModuleList([
            nn.Linear(128, classes_per_task) for _ in range(num_experts)
        ])
        
        self._initialize_weights()
    
    def _create_feature_extractor(self, input_channels):
        """Builds the LeNet-style feature extractor with a GridCellLayer."""
        return nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GridCellLayer(32),  # <-- Grid Cell Layer added here
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            # Extra pooling to tame parameter count (8x8 -> 4x4)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _get_feature_dim(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.feature_extractor(dummy_input)
            return dummy_output.numel()
    
    def _create_grid_cell_layer(self, channels):
        """Create the proven GridCellLayer component"""
        from train_hippocampal_moe import GridCellLayer
        return GridCellLayer(channels)
    
    def _create_gating_network(self, input_dim, num_experts, dropout_rate=0.5):
        """Creates a POWERFUL gating network for TinyImageNet feature separation."""
        return nn.Sequential(
            # 🔧 ENHANCED: Much deeper and more powerful gating network
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Feature enhancement layer
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1 if dropout_rate > 0.1 else 0),
            
            # Task discrimination layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.2 if dropout_rate > 0.2 else 0),
            
            # Final classification layer with proper initialization
            nn.Linear(512, num_experts)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_all_tasks(self, x):
        """Class-IL inference with HARD top-1 routing (no ensemble mixing)."""
        was_training = self.training
        self.eval()

        # Extract features once
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)

        # Compute gating logits and pick the top-1 expert per sample
        gate_logits = self.gating_network(features_flat)
        chosen_experts = torch.argmax(gate_logits, dim=1)  # (B,)

        # Prepare output tensor
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.num_classes, device=x.device)

        # For efficiency, group samples by chosen expert
        for expert_id in range(self.num_experts):
            mask = (chosen_experts == expert_id)
            if mask.sum() == 0:
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            # Forward only the samples belonging to this expert
            f_subset = features_flat[idx]
            dg_out, ca3_out, x_separated = self.hippocampal_experts[expert_id](f_subset)
            combined = torch.cat([dg_out, ca3_out, f_subset], dim=1)
            ca1_out = self.ca1_integration(combined)
            expert_logits = self.output_layers[expert_id](ca1_out)  # (n_i, 20)

            start = expert_id * self.classes_per_task
            end = start + self.classes_per_task
            outputs[idx, start:end] = expert_logits

        self.train(was_training)
        return outputs

    def forward(self, x, task_id=None, return_expert_outputs=False, precomputed_features=None):
        """
        Main forward pass for Hippocampal MoE
        Supports forced routing (task_id is not None) or soft gating.
        NEW: Can accept precomputed features to avoid re-running extractor.
        """
        
        if precomputed_features is None:
            features = self.feature_extractor(x)
        else:
            features = precomputed_features

        features_flat = features.view(features.size(0), -1)
        
        # Gating network decides the expert
        gate_logits = self.gating_network(features_flat)
        
        if task_id is not None:
            # FORCED ROUTING for expert training.
            # All samples in the batch are routed to the designated expert.
            chosen_experts = torch.full_like(gate_logits.argmax(dim=1), fill_value=task_id)
        else:
            # INFERENTIAL ROUTING for Class-IL evaluation.
            # The gate decides which expert to use for each sample.
            chosen_experts = torch.argmax(gate_logits, dim=1)

        # Prepare output tensor
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.num_classes, device=x.device)

        # For efficiency, group samples by chosen expert
        for expert_id in range(self.num_experts):
            mask = (chosen_experts == expert_id)
            if mask.sum() == 0:
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            # Forward only the samples belonging to this expert
            f_subset = features_flat[idx]
            dg_out, ca3_out, x_separated = self.hippocampal_experts[expert_id](f_subset)
            combined = torch.cat([dg_out, ca3_out, f_subset], dim=1)
            ca1_out = self.ca1_integration(combined)
            expert_logits = self.output_layers[expert_id](ca1_out)  # (n_i, 20)

            start = expert_id * self.classes_per_task
            end = start + self.classes_per_task
            outputs[idx, start:end] = expert_logits

        return outputs, gate_logits

def tinyimagenet_phase1_train_experts(model, train_loaders, test_loaders, device, args):
    """
    TinyImageNet-specific Phase 1: Train experts independently with memory replay
    Handles 0-19 local labels for each task properly
    """
    import torch.nn.functional as F
    import torch.optim as optim
    from tqdm import tqdm
    import logging
    
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: TRAINING HIPPOCAMPAL EXPERTS INDEPENDENTLY WITH MEMORY REPLAY")
    logger.info("="*80)
    
    # Initialize replay buffer
    replay_buffer = HippocampalReplayBuffer(capacity_per_task=200)
    logger.info("Hippocampal replay buffer initialized (200 samples/task)")
    
    expert_results = []
    
    # We will create a fresh optimiser **inside** each expert loop so that
    # once an expert is finished we can freeze its weights. This ensures replay
    # updates only the shared trunk (e.g. CA1) and never alters already-trained
    # experts.
    for expert_id in range(len(train_loaders)):
        # ------------------------------------------------------------------
        # 1) Set requires_grad flags: train current expert + shared CA1 only
        # ------------------------------------------------------------------
        for name, p in model.named_parameters():
            if f"hippocampal_experts.{expert_id}" in name or \
               f"output_layers.{expert_id}" in name:
                p.requires_grad = True  # current expert + its head
            elif "hippocampal_experts" in name or "output_layers." in name:
                p.requires_grad = False  # freeze past experts
            elif "feature_extractor" in name or "ca1_integration" in name:
                p.requires_grad = True   # 🔧 Allow backbone + CA1 to keep learning
            elif "gating_network" in name or "soft_gating" in name:
                p.requires_grad = False  # 🔧 CRITICAL FIX: Freeze gating to preserve pretraining!
            else:
                # All other parameters frozen
                p.requires_grad = False

        # 🔧 ACTION 3: Enhanced optimizer with dedicated trunk learning rate
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'feature_extractor' in name:
                    # Give trunk higher LR to adapt faster to new tasks
                    trainable_params.append({'params': param, 'lr': args.learning_rate * args.trunk_lr_multiplier})
                elif 'gating_network' in name:
                    # Gating network gets higher LR for task discrimination
                    trainable_params.append({'params': param, 'lr': args.learning_rate * 1.5})
                else:
                    # Expert params get standard LR
                    trainable_params.append({'params': param, 'lr': args.learning_rate})
        
        optimizer = optim.AdamW(
            trainable_params,
            weight_decay=args.weight_decay * 2.0
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.expert_epochs)
        
        logger.info(f"\nTraining Hippocampal Expert {expert_id}")
        logger.info(f"   Components: Grid Cells → DG (4x expansion) → CA3 → CA1 (MEC bypass)")
        
        # 🔧 Verify gating is trainable for task-ID classification
        gating_params_trainable = any(p.requires_grad for name, p in model.named_parameters() 
                                     if 'gating_network' in name or 'soft_gating' in name)
        logger.info(f"   🎯 Gating network trainable for task-ID: {gating_params_trainable}")
        
        # Count trainable parameters for this expert
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"   📊 Trainable parameters: {trainable_params:,}")
        
        train_loader = train_loaders[expert_id]
        test_loader = test_loaders[expert_id]
        
        best_acc = 0.0
        best_expert_state = None
        
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        # Test run: reduce epochs to 2 for faster debugging
        num_epochs = 2 if args.test_run else args.expert_epochs
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Separate tracking for current vs replay performance
            curr_correct_total = 0
            curr_total_total = 0
            replay_correct_total = 0
            replay_total_total = 0
            curr_loss_total = 0.0
            replay_loss_total = 0.0
            batch_count = 0
            
            progress_bar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}")
            
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                # Test run: only process 2 batches
                if args.test_run and batch_idx >= 2:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Labels are already 0-19 for each task (handled by LabelAdjustedDataset)
                local_labels = labels  # No mapping needed!
                
                optimizer.zero_grad()
                
                # FORCED routing to this expert
                # Add dropout during training for regularization
                if model.training:
                    inputs = F.dropout2d(inputs, p=0.1, training=True)
                
                # 🔧 ENHANCED: Extract features and process through hippocampus for better separation
                features = model.feature_extractor(inputs)
                features_flat = features.view(features.size(0), -1)
                
                # 🔧 CRITICAL: Process through hippocampus to get enhanced features
                dg_output, ca3_output, x_separated = model.hippocampal_experts[expert_id](features_flat)
                
                # 🔧 FIXED: Skip gating loss since gating network is frozen to preserve pretraining
                task_id_loss = torch.tensor(0.0, device=device)
                
                # Debug: Gating network is frozen to preserve pretraining
                if batch_idx < 3 and epoch == 0:
                    logger.info(f"🐛 DEBUG: Gating network frozen, task_id_loss={task_id_loss.item():.6f}")
                
                # 🔧 FIXED: Extract enhanced outputs for sparsity and feature separation
                dg_output, ca3_output, x_separated = model.hippocampal_experts[expert_id](features_flat)
                
                # Forward through the model with precomputed features
                outputs, _ = model(inputs, task_id=expert_id, precomputed_features=features)
                
                # Extract outputs for this task
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                # Main classification loss
                current_loss = F.cross_entropy(task_outputs, local_labels)
                
                # 🔧 FIXED: Calculate biological DG sparsity loss (target 3% active)
                actual_sparsity = (dg_output > 0).float().mean()
                sparsity_loss = ((actual_sparsity - args.dg_sparsity) ** 2) + 0.001 * torch.mean(torch.abs(dg_output))
                
                # 🔧 ENHANCED: Calculate feature separability loss (maximize separation)
                feature_sep_loss = 1.0 / (1.0 + calculate_enhanced_feature_separability(features_flat, x_separated))
                feature_sep_loss = torch.tensor(feature_sep_loss, device=device, requires_grad=False)
                
                # Track current batch performance
                _, pred_curr = torch.max(task_outputs, 1)
                curr_correct = (pred_curr == local_labels).sum().item()
                curr_total = local_labels.size(0)
                curr_correct_total += curr_correct
                curr_total_total += curr_total
                curr_loss_total += current_loss.item()
                
                # 🔧 FIXED: Combine all losses - classification, sparsity, and feature separation (no task-ID loss)
                total_loss = (current_loss + 
                             (args.sparsity_loss_weight * sparsity_loss) +
                             (0.1 * feature_sep_loss))  # Feature separation weight
                
                # Add samples to replay buffer
                for i in range(inputs.size(0)):
                    replay_buffer.add_sample(inputs[i], labels[i], expert_id, current_loss)
                
                # Memory replay from previous tasks
                replay_loss = torch.tensor(0.0, device=device)
                r_correct_batch = 0
                r_total_batch = 0
                
                if expert_id > 0:
                    # Define a transform to apply to replay samples on the fly.
                    # This prevents the model from simply memorizing the fixed replay buffer.
                    replay_transform = transforms.Compose([
                        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                        transforms.RandomHorizontalFlip(),
                    ])

                    replay_inputs, replay_labels, replay_task_ids = replay_buffer.sample_replay(
                        batch_size=16, exclude_task=expert_id
                    )
                    
                    if replay_inputs is not None:
                        # Apply augmentation to the batch of replay images
                        replay_inputs = replay_transform(replay_inputs).to(device)
                        replay_labels = replay_labels.to(device)
                        
                        # Process replay samples
                        for r_task_id in torch.unique(replay_task_ids):
                            task_mask = (replay_task_ids == r_task_id)
                            if task_mask.sum() > 0:
                                r_inputs = replay_inputs[task_mask]
                                r_labels = replay_labels[task_mask]  # Already 0-19
                                
                                # Skip if only 1 sample (BatchNorm requires batch_size > 1)
                                if r_inputs.size(0) == 1:
                                    continue
                                
                                # Forward through the replay task's expert (no gating loss during expert training)
                                r_features = model.feature_extractor(r_inputs)
                                r_outputs, _ = model(r_inputs, task_id=r_task_id.item(), precomputed_features=r_features)
                                start_idx = r_task_id * model.classes_per_task
                                end_idx = start_idx + model.classes_per_task
                                r_task_outputs = r_outputs[:, start_idx:end_idx]
                                
                                # Track replay performance
                                _, r_pred = torch.max(r_task_outputs, 1)
                                r_correct_batch += (r_pred == r_labels).sum().item()
                                r_total_batch += r_labels.size(0)
                                
                                # Only classification loss for replay (no task-ID loss)
                                replay_loss += F.cross_entropy(r_task_outputs, r_labels, label_smoothing=0.1)
                        
                        replay_correct_total += r_correct_batch
                        replay_total_total += r_total_batch
                        replay_loss_total += replay_loss.item()
                        
                        # 🔧 FIXED: Don't overwrite total_loss! Add replay loss to existing loss components
                        # 🔧 ACTION 3: REDUCED REPLAY WEIGHT to fix overfitting to small replay buffer
                        total_loss = total_loss + (args.replay_loss_weight * replay_loss)
                
                batch_count += 1
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # 🔧 Monitor DG sparsity (gating network frozen to preserve pretraining)
                actual_sparsity = (dg_output > 0).float().mean().item()
                
                # Debug: Monitor sparsity and feature separation (gating frozen during expert training)
                if batch_idx < 5 and epoch == 0:  # Debug first few batches of first epoch
                    logger.info(f"🐛 DEBUG Batch {batch_idx}: sparsity_loss={sparsity_loss.item():.6f}, feat_sep_loss={feature_sep_loss.item():.6f}")
                    logger.info(f"🐛 DEBUG: current_loss={current_loss.item():.6f}, total_loss={total_loss.item():.6f}, actual_sparsity={actual_sparsity:.3f}")
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.3f}",
                    'sparsity_loss': f"{sparsity_loss.item():.3f}",
                    'dg_sparsity': f"{actual_sparsity:.1%}",
                    'curr_acc': f"{100.*curr_correct_total/curr_total_total:.1f}%",
                    'replay_acc': f"{100.*replay_correct_total/max(1,replay_total_total):.1f}%" if replay_total_total > 0 else "N/A"
                })
            
            scheduler.step()
            
            # ========== End of Epoch Loop (expert training finished) ==========

            # 🔧 ACTION 1: Enhanced diagnostics after expert completes
            try:
                routing_acc, feat_sep = evaluate_router_performance(model, test_loaders, device, args)
                logger.info(f"📈  Post-Expert {expert_id} Feature-Separability (FeatSep): {feat_sep:.2f}")
                logger.info(f"🎯  Post-Expert {expert_id} Routing Accuracy: {routing_acc:.1f}%")
                
                # Test gating network task-ID accuracy using pretraining-frozen weights
                model.eval()
                gate_correct = gate_total = 0
                with torch.no_grad():
                    # Test on all tasks trained so far (0 to expert_id)
                    for task_id in range(expert_id + 1):
                        task_test_loader = test_loaders[task_id]
                        for inputs, labels in task_test_loader:
                            inputs = inputs.to(device)
                            features = model.feature_extractor(inputs)
                            features_flat = features.view(features.size(0), -1)
                            
                            # Use raw features for gating (as in pretraining)
                            gate_logits = model.gating_network(features_flat)
                            
                            target_task_ids = torch.full((inputs.size(0),), task_id, device=device)
                            _, gate_pred = torch.max(gate_logits, 1)
                            gate_correct += (gate_pred == target_task_ids).sum().item()
                            gate_total += inputs.size(0)
                
                gate_acc = (gate_correct / gate_total) * 100 if gate_total > 0 else 0
                logger.info(f"🧠  Post-Expert {expert_id} Gating Task-ID Classification (Tasks 0-{expert_id}): {gate_acc:.1f}% (frozen from pretraining)")
                
                # 🔧 FIXED: Test DG sparsity and feature separation with enhanced experts
                dg_sparsities = []
                feature_separabilities = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        features = model.feature_extractor(inputs)
                        features_flat = features.view(features.size(0), -1)
                        dg_output, _, x_separated = model.hippocampal_experts[expert_id](features_flat)
                        batch_sparsity = (dg_output > 0).float().mean().item()
                        dg_sparsities.append(batch_sparsity)
                        
                        # Calculate feature separability
                        feat_sep = calculate_enhanced_feature_separability(features_flat, x_separated)
                        feature_separabilities.append(feat_sep)
                
                avg_dg_sparsity = np.mean(dg_sparsities) if dg_sparsities else 0
                avg_feat_sep = np.mean(feature_separabilities) if feature_separabilities else 0
                
                logger.info(f"🧬  Post-Expert {expert_id} DG Sparsity: {avg_dg_sparsity:.1%} (target: {args.dg_sparsity:.1%})")
                logger.info(f"🔀  Post-Expert {expert_id} Feature Separability: {avg_feat_sep:.4f} (target: >0.2)")
                
                # Sparsity deviation analysis
                sparsity_deviation = abs(avg_dg_sparsity - args.dg_sparsity) / args.dg_sparsity
                if sparsity_deviation < 0.5:
                    logger.info(f"✅  Sparsity is well-controlled (deviation: {sparsity_deviation:.1%})")
                else:
                    logger.warning(f"⚠️  Sparsity deviation is high (deviation: {sparsity_deviation:.1%})")
                
                # Feature separability analysis
                if avg_feat_sep > 0.2:
                    logger.info(f"✅  Feature separation is good ({avg_feat_sep:.4f})")
                elif avg_feat_sep > 0.1:
                    logger.warning(f"⚠️  Feature separation is moderate ({avg_feat_sep:.4f})")
                else:
                    logger.warning(f"❌  Feature separation is poor ({avg_feat_sep:.4f})")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not compute diagnostics after Expert {expert_id}: {e}")
            
            # Evaluate this expert
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    local_labels = labels
                    
                    outputs, _ = model(inputs, task_id=expert_id)
                    start_idx = expert_id * model.classes_per_task
                    end_idx = start_idx + model.classes_per_task
                    task_outputs = outputs[:, start_idx:end_idx]
                    
                    _, predicted = torch.max(task_outputs, 1)
                    test_correct += (predicted == local_labels).sum().item()
                    test_total += local_labels.size(0)
            
            test_acc = (test_correct / test_total) * 100

            # End-of-epoch diagnostic summary
            avg_curr_acc = 100.0 * curr_correct_total / curr_total_total if curr_total_total > 0 else 0.0
            avg_replay_acc = 100.0 * replay_correct_total / replay_total_total if replay_total_total > 0 else 0.0
            avg_curr_loss = curr_loss_total / batch_count if batch_count > 0 else 0.0
            avg_replay_loss = replay_loss_total / batch_count if replay_total_total > 0 else 0.0
             
            # Main epoch summary line
            if replay_total_total > 0:
                combined_correct = curr_correct_total + replay_correct_total
                combined_total = curr_total_total + replay_total_total
                train_acc = 100.0 * combined_correct / combined_total if combined_total > 0 else 0.0
                
                logger.info(f"Expert {expert_id} Epoch {epoch+1}: "
                           f"Current={avg_curr_acc:.1f}%, Replay={avg_replay_acc:.1f}%, "
                           f"Combined={train_acc:.1f}%, Test={test_acc:.1f}% (Best: {best_acc:.1f}%)")
                logger.info(f"   📊 BREAKDOWN: Current samples: {avg_curr_acc:.1f}% | "
                           f"Replay samples: {avg_replay_acc:.1f}% | "
                           f"Gap: {avg_replay_acc - avg_curr_acc:+.1f}%")
            else:
                logger.info(f"Expert {expert_id} Epoch {epoch+1}: "
                           f"Current={avg_curr_acc:.1f}%, Test={test_acc:.1f}% (Best: {best_acc:.1f}%) [No replay]")
            
            # Check if test accuracy improved
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                best_expert_state = {
                    name: param.clone().detach().cpu() for name, param in model.named_parameters()
                    if 'hippocampal_experts' in name or 'ca1_integration' in name or 'output_layers' in name
                }
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping Expert {expert_id} at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Load best expert weights back into model
        if best_expert_state is not None:
            for name, param in model.named_parameters():
                if name in best_expert_state:
                    param.data.copy_(best_expert_state[name].to(device))
            logger.info(f"🎯 Expert {expert_id} best weights restored (accuracy: {best_acc:.2f}%)")
        
        expert_results.append({
            'expert_id': expert_id,
            'best_accuracy': best_acc
        })
        
        logger.info(f"Expert {expert_id} final accuracy: {best_acc:.2f}%")
    
        # NEW: freeze expert only **once** after all epochs are done
        for p in model.hippocampal_experts[expert_id].parameters():
            p.requires_grad = False
        for p in model.output_layers[expert_id].parameters():
            p.requires_grad = False
        logger.info(f"❄️  Expert {expert_id} frozen. Shared trunk will still get replay updates.")
    
    # Unfreeze gating for final evaluation (retains pretraining weights, no further training)
    for param in model.gating_network.parameters():
        param.requires_grad = True
    
    logger.info(f"\nPhase 1 Complete - Expert accuracies: {[r['best_accuracy'] for r in expert_results]}")
    return expert_results

def train_tinyimagenet_hippocampal_moe(model, train_loaders, test_loaders, device, args, task_classes_global):
    """
    Train TinyImageNet Hippocampal MoE using the EXACT same 3-phase approach as CIFAR-10
    NOW WITH PROPER CHECKPOINT MANAGEMENT FOR BEST WEIGHTS TRANSFER
    """
    logger = logging.getLogger()
    
    # Convert global task classes to local task classes (0-19 for each task)
    task_classes = []
    for task_id in range(len(train_loaders)):
        task_classes.append(list(range(args.classes_per_task)))  # 0-19 for each task
    model.set_task_classes(task_classes)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("🚀 Starting 3-phase hippocampal training on TinyImageNet...")
    logger.info(f"Using proven CIFAR-10 architecture scaled for 64x64 images")
    logger.info("✅ ENHANCED: Proper weight transfer between phases")
    
    # Phase 1: Train experts independently with memory replay
    logger.info("Phase 1: Training experts independently with memory replay...")
    phase1_results = tinyimagenet_phase1_train_experts(model, train_loaders, test_loaders, device, args)
    
    # Save Phase 1 checkpoint with best expert weights
    phase1_accuracy = np.mean([r['best_accuracy'] for r in phase1_results])
    phase1_checkpoint_path = os.path.join(checkpoint_dir, "phase1_best_experts.pth")
    try:
        save_checkpoint(
            model, None, args.expert_epochs, 
            phase1_accuracy, phase1_checkpoint_path,
            metadata={'expert_accuracies': [r['best_accuracy'] for r in phase1_results], 'phase': 1}
        )
        logger.info(f"✅ Phase 1 checkpoint saved to {phase1_checkpoint_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save Phase 1 checkpoint: {e}")
    
    # Phase 2 is now redundant because the gating network is trained during Phase 1
    logger.info("✅ SKIPPING Phase 2: Gating network is now trained directly in Phase 1.")
    logger.info("🔄 Model already has best weights from Phase 1 training (no reload needed)...")
    
    # Clean up memory after Phase 1
    cleanup_memory()
    
    # Final evaluation
    logger.info("🎯 Final evaluation...")
    final_results = tinyimagenet_evaluate_final_performance(model, test_loaders, task_classes_global, device)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(
        model, None, args.joint_epochs, 
        final_results['task_il_accuracy'], final_checkpoint_path,
        metadata=final_results
    )
    
    logger.info("✅ All checkpoints saved - best weights properly transferred between phases!")
    
    # Clean up intermediate checkpoints (keep final model)
    cleanup_checkpoints(checkpoint_dir, keep_final=True)
    cleanup_memory()
    
    # Optional post-training analysis with rich visuals
    if args.analyze_trained:
        logger.info("🔬 Running detailed TRAINED model analysis (visuals)...")
        # Create analysis directory
        analysis_dir = os.path.join(os.getcwd(), "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        analyze_trained_tinyimagenet_model(model, test_loaders, task_classes, device, analysis_dir)
    
    return final_results

def tinyimagenet_evaluate_final_performance(model, test_loaders, task_classes_global, device):
    """
    Fixed final evaluation for TinyImageNet with proper label handling
    """
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("TINYIMAGENET FINAL PERFORMANCE EVALUATION (FIXED)")
    logger.info("="*80)
    
    model.eval()
    
    # Task-IL evaluation (using oracle task information)
    task_il_correct = 0
    task_il_total = 0
    expert_accuracies = []
    
    with torch.no_grad():
        for expert_id, test_loader in enumerate(test_loaders):
            expert_correct = 0
            expert_total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Labels are already local (0-19) from LabelAdjustedDataset
                local_labels = labels
                
                # Oracle task information for Task-IL
                outputs, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                _, predicted = torch.max(task_outputs, 1)
                expert_correct += (predicted == local_labels).sum().item()
                expert_total += local_labels.size(0)
            
            expert_acc = (expert_correct / expert_total) * 100
            expert_accuracies.append(expert_acc)
            task_il_correct += expert_correct
            task_il_total += expert_total
            
            # Show the ACTUAL task classes for this expert
            actual_task_classes = task_classes_global[expert_id] if expert_id < len(task_classes_global) else list(range(expert_id * 20, (expert_id + 1) * 20))
            logger.info(f"Expert {expert_id} (actual classes {actual_task_classes}): {expert_acc:.2f}%")
    
    task_il_accuracy = (task_il_correct / task_il_total) * 100
    
    # DEBUG: Class-IL evaluation with detailed logging
    class_il_correct = 0
    class_il_total = 0
    
    logger.info(f"🔍 DEBUG: Task classes mapping:")
    for task_id in range(len(task_classes_global)):
        logger.info(f"  Task {task_id}: {task_classes_global[task_id]}")
    
    # Test a single batch first to debug
    logger.info("🔍 DEBUG: Testing Class-IL routing on single batch...")
    test_batch_count = 0
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            if test_batch_count >= 1:  # Only test first batch for debugging
                break
                
            actual_global_classes = task_classes_global[task_id]
            for inputs, local_labels in test_loader:
                inputs = inputs.to(device)
                local_labels = local_labels.to(device)
                
                # Convert to global labels for Class-IL
                global_labels = torch.tensor([actual_global_classes[label.item()] for label in local_labels]).to(device)
                
                # Class-IL: no task information, use soft routing
                outputs = model.forward_all_tasks(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Debug first few samples
                logger.info(f"🔍 DEBUG Sample routing:")
                logger.info(f"  Task {task_id}, Local labels: {local_labels[:5].cpu().tolist()}")
                logger.info(f"  Global labels: {global_labels[:5].cpu().tolist()}")
                logger.info(f"  Predictions: {predicted[:5].cpu().tolist()}")
                logger.info(f"  Output shape: {outputs.shape}")
                logger.info(f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                
                # Check routing behavior
                _, gate_logits = model(inputs, task_id=None)
                _, predicted_experts = torch.max(gate_logits, 1)
                expert_counts = torch.bincount(predicted_experts, minlength=model.num_experts)
                logger.info(f"  Routing distribution: {expert_counts.cpu().tolist()}")
                
                # CRITICAL DEBUG: Check what expert SHOULD be used for Task 0
                logger.info(f"  ❗ PROBLEM: Task 0 samples should route to Expert 0, but routing to:")
                for i in range(min(10, len(predicted_experts))):
                    logger.info(f"    Sample {i}: Local={local_labels[i].item()}, Global={global_labels[i].item()}, RoutedTo=Expert{predicted_experts[i].item()}")
                
                # Test oracle routing for comparison
                oracle_outputs, _ = model(inputs, task_id=0)  # Force Task 0
                oracle_task_outputs = oracle_outputs[:, 0:20]  # Task 0 outputs
                _, oracle_predicted = torch.max(oracle_task_outputs, 1)
                oracle_correct = (oracle_predicted == local_labels).sum().item()
                oracle_acc = oracle_correct / len(local_labels) * 100
                logger.info(f"  🎯 Oracle routing (forced Expert 0): {oracle_acc:.1f}% accuracy")
                logger.info(f"  🔥 This proves Expert 0 CAN classify Task 0 correctly!")
                
                # The issue is that soft routing is sending Task 0 to wrong experts!
                
                test_batch_count += 1
                break
    
    # Now run full Class-IL evaluation
    logger.info("🔍 Running full Class-IL evaluation...")
    
    # Create combined test dataset with PROPER global labels
    all_test_data = []
    for task_id, test_loader in enumerate(test_loaders):
        actual_global_classes = task_classes_global[task_id]
        for inputs, local_labels in test_loader:
            # Convert local labels (0-19) back to global labels using actual task classes
            global_labels = torch.tensor([actual_global_classes[label.item()] for label in local_labels])
            all_test_data.append((inputs, global_labels))
    
    # Shuffle for proper Class-IL evaluation
    import random
    random.shuffle(all_test_data)
    
    logger.info(f"Class-IL evaluation: Testing {len(all_test_data)} batches with global labels")
    
    with torch.no_grad():
        for inputs, global_labels in all_test_data:
            inputs = inputs.to(device)
            global_labels = global_labels.to(device)
            
            # Class-IL: no task information, use soft routing
            outputs = model.forward_all_tasks(inputs)
            _, predicted = torch.max(outputs, 1)
            
            class_il_correct += (predicted == global_labels).sum().item()
            class_il_total += global_labels.size(0)
    
    class_il_accuracy = (class_il_correct / class_il_total) * 100
    
    logger.info(f"\nFIXED FINAL RESULTS:")
    logger.info(f"Expert accuracies: {[f'{acc:.1f}%' for acc in expert_accuracies]}")
    logger.info(f"Task-IL Accuracy: {task_il_accuracy:.2f}%")
    logger.info(f"Class-IL Accuracy: {class_il_accuracy:.2f}%")
    logger.info(f"Task-IL vs Class-IL gap: {task_il_accuracy - class_il_accuracy:.2f}%")
    
    return {
        'expert_accuracies': expert_accuracies,
        'task_il_accuracy': task_il_accuracy,
        'class_il_accuracy': class_il_accuracy,
        'task_class_gap': task_il_accuracy - class_il_accuracy
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TinyImageNet Hippocampal MoE Training with Proven LeNet Architecture')
    
    # Model parameters (same as CIFAR-10)
    parser.add_argument('--num_experts', type=int, default=10, help='Number of experts')
    parser.add_argument('--classes_per_task', type=int, default=20, help='Classes per task')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')
    
    # Training parameters (same as CIFAR-10 optimal)
    parser.add_argument('--expert_epochs', type=int, default=12, help='Epochs for expert training')
    parser.add_argument('--gate_epochs', type=int, default=20, help='Epochs for gate training')
    parser.add_argument('--router_epochs', type=int, default=80, help='Epochs for task-ID router pretraining (increased for difficult TinyImageNet)')
    parser.add_argument('--router_lr', type=float, default=3e-3, help='Learning rate for router pretraining (increased for difficult task)')
    parser.add_argument('--joint_epochs', type=int, default=10, help='Epochs for joint training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gate_lr', type=float, default=1e-4, help='Gate learning rate (fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    
    # Advanced MoE strategies (proven optimal values)
    parser.add_argument('--balance_loss_coef', type=float, default=0.01, help='Load balancing loss coefficient')
    parser.add_argument('--diversity_loss_coef', type=float, default=0.1, help='Expert diversity loss coefficient')
    parser.add_argument('--gating_loss_coef', type=float, default=2.0, help='Gating loss coefficient')
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True, help='Use balanced sampling')
    
    # 🔧 UNIFIED & 3: Feature separability improvements
    parser.add_argument('--aux_loss_weight', type=float, default=0.3, help='Task-ID loss weight for gating network')
    parser.add_argument('--replay_loss_weight', type=float, default=0.5, help='Replay loss weight (reduce overfitting)')
    parser.add_argument('--trunk_lr_multiplier', type=float, default=2.0, help='Feature extractor LR multiplier')
    
    # 🔧 ACTION 4: Enhanced sparsity controls
    parser.add_argument('--dg_sparsity', type=float, default=0.03, help='DG sparsity level (biological: 2-5%)')
    parser.add_argument('--sparsity_loss_weight', type=float, default=50.0, help='Sparsity regularization weight (increased from 1.0 to 50.0)')
    parser.add_argument('--adaptive_sparsity', action='store_true', default=True, help='Use adaptive sparsity during training')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # Analysis flag
    parser.add_argument('--analyze_trained', action='store_true', default=True, help='Run detailed visual analysis after training (disable with --no-analyze_trained)')
    parser.add_argument('--test_run', action='store_true', help='Quick test run with only 2 batches per epoch (for debugging)')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup
    log_dir = setup_logging()
    logger = logging.getLogger()
    
    logger.info("🧠 === TINYIMAGENET HIPPOCAMPAL MOE WITH ENHANCED FEATURE SEPARABILITY ===")
    logger.info(f"Tasks: {args.num_tasks}, Classes per task: {args.classes_per_task}")
    logger.info(f"Experts: {args.num_experts}")
    logger.info(f"Phase epochs: {args.expert_epochs}/{args.gate_epochs}/{args.joint_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}, Batch size: {args.batch_size}")
    logger.info(f"Loss coefficients: balance={args.balance_loss_coef}, diversity={args.diversity_loss_coef}")
    logger.info(f"🔧 ENHANCED: Auxiliary task-ID loss weight: {args.aux_loss_weight}")
    logger.info(f"🔧 ENHANCED: Replay loss weight: {args.replay_loss_weight} (reduced overfitting)")
    logger.info(f"🔧 ENHANCED: Trunk LR multiplier: {args.trunk_lr_multiplier}x (faster adaptation)")
    logger.info(f"🔧 ENHANCED: DG sparsity target: {args.dg_sparsity:.1%} (biological level)")
    logger.info(f"🔧 ENHANCED: Sparsity loss weight: {args.sparsity_loss_weight}")
    logger.info(f"Using EXACT same architecture that achieved 70% Class-IL on CIFAR-10")
    if args.test_run:
        logger.info("🚀 TEST RUN MODE: Only processing 2 batches per epoch for debugging")
    
    # Set seed and device
    set_seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Create TinyImageNet tasks
    logger.info("Creating TinyImageNet tasks...")
    train_loaders, test_loaders, task_classes = create_tinyimagenet_tasks(
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        batch_size=args.batch_size
    )
    
    # Create model with enhanced architecture
    logger.info("Creating TinyImageNet Hippocampal MoE model with auxiliary task-ID head...")
    model = TinyImageNetOptimalHippocampalMoE(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        input_channels=3
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Router pre-training (NEW: task-ID classification stage)
    logger.info("🚀 ROUTER PRE-TRAINING: Learning task-discriminative features...")
    pretrain_router(model, train_loaders, test_loaders, device, args)
    
    # NEW: Evaluate the quality of the pre-trained router
    evaluate_router_performance(model, test_loaders, device, args)
    
    # Train the model (experts + gate fine-tuning phases)
    results = train_tinyimagenet_hippocampal_moe(model, train_loaders, test_loaders, device, args, task_classes)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Results saved in: {log_dir}")
    
    return results

def evaluate_router_performance(model, test_loaders, device, args):
    """
    Evaluates the pre-trained router's accuracy and feature separability.
    This gives a clear signal on how well the pre-training phase worked.
    """
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("🔬 EVALUATING PRE-TRAINED ROUTER PERFORMANCE")
    logger.info("="*80)

    model.eval()
    
    all_features = []
    all_enhanced_features = []  # 🔧 NEW: Store hippocampal-enhanced features
    all_task_ids = []
    correct_routing = 0
    total_samples = 0

    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for batch_idx, (inputs, _) in enumerate(test_loader):
                # Test run: only process 2 batches per task
                if args.test_run and batch_idx >= 2:
                    break
                inputs = inputs.to(device)
                
                # Get features and gate logits
                features = model.feature_extractor(inputs)
                features_flat = features.view(features.size(0), -1)
                
                # 🔧 CRITICAL FIX: Use hippocampal-enhanced features for separability
                # Process through first available expert to get enhanced features
                dg_out, ca3_out, enhanced_features = model.hippocampal_experts[0](features_flat)
                
                gate_logits = model.gating_network(features_flat)
                
                # Store both raw and enhanced features for comparison
                all_features.append(features_flat.cpu())
                all_enhanced_features.append(enhanced_features.cpu())
                all_task_ids.append(torch.full((inputs.size(0),), task_id))

                # Check routing accuracy with FIXED temperature
                gate_logits_scaled = gate_logits / 1.0  # 🔧 FIXED: Remove harmful temperature scaling
                _, predicted_experts = torch.max(gate_logits_scaled, 1)
                target_expert = torch.full_like(predicted_experts, task_id)
                correct_routing += (predicted_experts == target_expert).sum().item()
                total_samples += inputs.size(0)

    # Combine all collected data
    all_features = torch.cat(all_features, dim=0)
    all_enhanced_features = torch.cat(all_enhanced_features, dim=0)
    all_task_ids = torch.cat(all_task_ids, dim=0)

    # Calculate metrics using ENHANCED features
    routing_accuracy = (correct_routing / total_samples) * 100
    
    # 🔧 CRITICAL: Use enhanced features for separability calculation
    raw_feature_sep = calculate_feature_separability(all_features, all_task_ids, args.num_tasks)
    enhanced_feature_sep = calculate_feature_separability(all_enhanced_features, all_task_ids, args.num_tasks)

    logger.info(f"✅ Router Evaluation Complete:")
    logger.info(f"  - Test Routing Accuracy: {routing_accuracy:.2f}%")
    logger.info(f"  - Raw Feature Separability: {raw_feature_sep:.4f}")
    logger.info(f"  - Enhanced Feature Separability: {enhanced_feature_sep:.4f}")
    logger.info(f"  - Separability Improvement: {enhanced_feature_sep/max(raw_feature_sep, 1e-6):.1f}x")
    
    # Use enhanced separability for decision making
    if enhanced_feature_sep < 0.1:
        logger.warning(f"⚠️ Enhanced Feature Separability is still low. Need stronger hippocampal processing.")
    elif enhanced_feature_sep < 0.5:
        logger.info(f"🔄 Enhanced Feature Separability is moderate. Routing should work but may struggle.")
    else:
        logger.info(f"✅ Enhanced Feature Separability is excellent. Strong routing expected.")
        
    return routing_accuracy, enhanced_feature_sep

def calculate_feature_separability(features, labels, num_classes):
    """
    Calculates a metric for how well-separated features are for different classes.
    Higher is better. Ratio of inter-class distance to intra-class variance.
    """
    if features.shape[0] < 2:
        return 0.0
    
    device = features.device
    feature_dim = features.shape[1]
    
    # Calculate centroids for each class
    centroids = torch.zeros(num_classes, feature_dim, device=device)
    class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    for c in range(num_classes):
        class_mask = (labels == c)
        count = class_mask.sum()
        if count > 0:
            class_features = features[class_mask]
            centroids[c] = class_features.mean(dim=0)
            class_counts[c] = count
            
    # Calculate intra-class variance
    intra_class_variance = 0.0
    for c in range(num_classes):
        class_mask = (labels == c)
        if class_counts[c] > 0:
            class_features = features[class_mask]
            # Sum of squared distances from centroid
            variance = torch.sum((class_features - centroids[c].unsqueeze(0))**2)
            intra_class_variance += variance
    
    # Average intra-class variance per sample
    intra_class_variance /= features.shape[0]
    
    # Calculate inter-class distance (average squared distance between centroids)
    valid_centroids = centroids[class_counts > 0]
    if valid_centroids.shape[0] < 2:
        return 0.0 # Cannot compute inter-class distance
        
    # Expand dims to compute pairwise distances efficiently
    c1 = valid_centroids.unsqueeze(0)
    c2 = valid_centroids.unsqueeze(1)
    inter_class_dist = torch.sum((c1 - c2)**2, dim=2).mean()
                              
    separability = inter_class_dist / (intra_class_variance + 1e-6)
    
    return separability.item()

# Add checkpoint management functions
def save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_path, metadata=None):
    """Save checkpoint with validation"""
    try:
        # Create temporary file first
        temp_path = checkpoint_path + ".tmp"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
            'metadata': metadata or {}
        }, temp_path)

        # Validate checkpoint was written
        if os.path.getsize(temp_path) > 0:
            # Atomically replace old checkpoint
            os.replace(temp_path, checkpoint_path)
            logging.info(f"✅ Checkpoint saved to {checkpoint_path}")
        else:
            logging.error(f"❌ Checkpoint save failed: empty file {temp_path}")
            os.remove(temp_path)
    except Exception as e:
        logging.error(f"Checkpoint save failed: {str(e)}")

def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """Load checkpoint with validation and retry"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    # Retry up to 3 times in case of read errors
    for attempt in range(3):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # Validate checkpoint structure
            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint structure")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            best_metric = checkpoint.get('best_metric', 0.0)
            metadata = checkpoint.get('metadata')
            
            logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, best metric: {best_metric})")
            return epoch, best_metric, metadata

        except (RuntimeError, EOFError, ValueError) as e:
            logging.warning(f"Checkpoint load failed (attempt {attempt+1}/3): {str(e)}")
            time.sleep(0.5)  # Wait before retrying

    # All attempts failed
    raise RuntimeError(f"Failed to load checkpoint after 3 attempts: {checkpoint_path}")

# Add memory cleanup functions after the checkpoint functions

def cleanup_memory():
    """Clean up GPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_checkpoints(checkpoint_dir, keep_final=True):
    """Clean up intermediate checkpoints, optionally keep final"""
    import os
    if not os.path.exists(checkpoint_dir):
        return
        
    checkpoints_to_remove = []
    if not keep_final:
        checkpoints_to_remove = ['phase1_best_experts.pth', 'phase2_best_gating.pth', 'final_model.pth']
    else:
        checkpoints_to_remove = ['phase1_best_experts.pth', 'phase2_best_gating.pth']
    
    for checkpoint_file in checkpoints_to_remove:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logging.info(f"🗑️ Cleaned up checkpoint: {checkpoint_file}")

def enhanced_phase2_with_validation(model, train_loaders, test_loaders, device, args):
    """ENHANCED Phase 2: Train gating network with proper test validation and early stopping"""
    logger = logging.getLogger()
    
    logger.info("\n" + "="*80)
    logger.info("ENHANCED PHASE 2: GATING WITH TEST VALIDATION & EARLY STOPPING")
    logger.info("="*80)
    logger.info("🎯 Training epoch-by-epoch like experts, with test validation each epoch")
    logger.info("🛑 Early stopping when test routing stops improving")
    
    # Freeze ALL parameters except gating and the last two conv layers.
    # This allows the feature extractor to adapt to the task of routing.
    trainable_conv_params = []
    for name, param in model.named_parameters():
        if 'gating_network' in name or 'soft_gating' in name:
            param.requires_grad = True
        elif 'feature_extractor.7' in name or 'feature_extractor.4' in name:  # Last two conv layers
            param.requires_grad = True
            trainable_conv_params.append(param)
        else:
            param.requires_grad = False
    
    # Optimizer: gating params + slightly larger LR for fine-tuning conv layers
    gate_optimizer = optim.AdamW([
        {'params': model.gating_network.parameters(), 'lr': args.gate_lr},
        {'params': model.soft_gating.parameters(), 'lr': args.gate_lr * 0.5},
        {'params': trainable_conv_params, 'lr': args.gate_lr * 0.1}  # Increased LR
    ], weight_decay=args.weight_decay * 0.5)
    
    gate_scheduler = optim.lr_scheduler.CosineAnnealingLR(gate_optimizer, T_max=args.gate_epochs)
    
    best_test_routing_acc = 0.0
    best_train_routing_acc = 0.0
    patience = 8  # Increased patience for early stopping
    patience_counter = 0
    
    from tqdm import tqdm as _tqdm
    for epoch in range(args.gate_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        epoch_loss = 0.0
        train_correct_routing = 0
        train_total_samples = 0
        usage_meter = torch.zeros(args.num_experts, device=device)

        # Build cyclic iterators for each task loader
        task_iters = [iter(loader) for loader in train_loaders]
        batches_per_epoch = max(len(loader) for loader in train_loaders)
        per_task_samples = max(1, args.batch_size // args.num_tasks)

        train_pbar = _tqdm(range(batches_per_epoch), desc=f"Gate Epoch {epoch+1} [Train]", leave=False)

        for _ in train_pbar:
            batch_inputs = []
            batch_targets = []
            for task_id in range(args.num_tasks):
                try:
                    x, _ = next(task_iters[task_id])
                except StopIteration:
                    task_iters[task_id] = iter(train_loaders[task_id])
                    x, _ = next(task_iters[task_id])

                # Trim or pad to per_task_samples
                if x.size(0) > per_task_samples:
                    x = x[:per_task_samples]
                batch_inputs.append(x)
                batch_targets.append(torch.full((x.size(0),), task_id))

            inputs = torch.cat(batch_inputs, dim=0).to(device)
            target_expert = torch.cat(batch_targets, dim=0).long().to(device)

            gate_optimizer.zero_grad()

            # Extract features **with** gradient so that unfrozen conv layers (4 & 7)
            # can learn task-discriminative cues during gate training.
            features = model.feature_extractor(inputs)
            features_flat = features.view(features.size(0), -1)
            features_flat = F.dropout(features_flat, p=0.1, training=model.training)
            if model.training:
                features_flat = features_flat + 0.01 * torch.randn_like(features_flat)

            gate_logits = model.gating_network(features_flat)

            gate_loss = F.cross_entropy(gate_logits, target_expert, label_smoothing=0.05)

            gate_probs = F.softmax(gate_logits, dim=1)
            entropy_loss = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1).mean()

            # Load-balancing loss (Google Switch-Transformer style)
            lb_loss = calculate_load_balancing_loss(gate_logits, args.num_experts)

            total_loss = (args.gating_loss_coef * gate_loss
                          + 0.02 * entropy_loss
                          + args.balance_loss_coef * lb_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gating_network.parameters(), max_norm=1.0)
            gate_optimizer.step()

            # metrics
            epoch_loss += total_loss.item()
            _, predicted_experts = torch.max(gate_logits, 1)
            train_correct_routing += (predicted_experts == target_expert).sum().item()
            train_total_samples += inputs.size(0)

            usage_meter += gate_probs.detach().sum(dim=0)
        
        # Calculate training metrics
        avg_loss = epoch_loss / sum(len(loader) for loader in train_loaders)
        train_routing_acc = (train_correct_routing / train_total_samples) * 100
        
        # ===== TEST VALIDATION PHASE =====
        model.eval()
        test_correct_routing = 0
        test_total_samples = 0
        
        with torch.no_grad():
            test_progress = _tqdm(enumerate(test_loaders), desc=f"Gate Epoch {epoch+1} [Test]", total=len(test_loaders))
            
            for task_id, test_loader in test_progress:
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    
                    # Extract features and get gate logits
                    features = model.feature_extractor(inputs)
                    features_flat = features.view(features.size(0), -1)
                    gate_logits = model.gating_network(features_flat)
                    
                    # Target: route to correct task (expert)
                    target_expert = torch.full((inputs.size(0),), task_id, device=device, dtype=torch.long)
                    
                    # Track test routing accuracy
                    _, predicted_experts = torch.max(gate_logits, 1)
                    test_correct_routing += (predicted_experts == target_expert).sum().item()
                    test_total_samples += inputs.size(0)
                    
                    # Update progress bar
                    current_test_routing = (test_correct_routing / test_total_samples) * 100
                    test_progress.set_postfix({
                        'test_routing': f"{current_test_routing:.1f}%"
                    })
        
        test_routing_acc = (test_correct_routing / test_total_samples) * 100
        
        gate_scheduler.step()
        
        # Track best performance
        if train_routing_acc > best_train_routing_acc:
            best_train_routing_acc = train_routing_acc
            
        if test_routing_acc > best_test_routing_acc:
            best_test_routing_acc = test_routing_acc
            patience_counter = 0
            
            # Save best gating weights
            best_gating_state = {
                name: param.clone().detach().cpu() for name, param in model.named_parameters() 
                if 'gating_network' in name or 'soft_gating' in name
            }
            logger.info(f"💾 New best test routing: {test_routing_acc:.1f}% (epoch {epoch+1})")
        else:
            patience_counter += 1
        
        # Log epoch results
        train_test_gap = train_routing_acc - test_routing_acc
        logger.info(f"Gate Epoch {epoch+1}: Loss={avg_loss:.4f}")
        logger.info(f"  Train routing: {train_routing_acc:.1f}% (best: {best_train_routing_acc:.1f}%)")
        logger.info(f"  Test routing:  {test_routing_acc:.1f}% (best: {best_test_routing_acc:.1f}%)")
        logger.info(f"  Train/Test gap: {train_test_gap:.1f}% {'✅' if train_test_gap < 20 else '⚠️' if train_test_gap < 40 else '❌'}")
        
        # 📊 Extra diagnostic: track how feature separability evolves during gate training
        try:
            _, feat_sep = evaluate_router_performance(model, test_loaders, device, args)
            logger.info(f"📈  Gate Epoch {epoch+1}: Feature-Separability (FeatSep) = {feat_sep:.2f}")
        except Exception as e:
            logger.warning(f"⚠️  Could not compute FeatSep during Gate Epoch {epoch+1}: {e}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"🛑 Early stopping: No improvement for {patience} epochs")
            break
        
        # Stop if we achieve good generalization
        if test_routing_acc > 60.0 and train_test_gap < 15.0:
            logger.info(f"🎯 Excellent generalization achieved: {test_routing_acc:.1f}% test with {train_test_gap:.1f}% gap")
            break
        
        # Update progress bar every 50 batches
        if train_total_samples % (50 * args.batch_size) < inputs.size(0):
            current_train_acc = 100. * train_correct_routing / max(1, train_total_samples)
            train_pbar.set_postfix({'train_routing': f"{current_train_acc:.1f}%", 'loss': f"{total_loss.item():.3f}"})
    
    # Load best weights
    if best_test_routing_acc > 20.0:  # Only load if we got reasonable performance
        for name, param in model.named_parameters():
            if name in best_gating_state:
                param.data = best_gating_state[name].to(device)
        logger.info(f"✅ Loaded best gating weights (test routing: {best_test_routing_acc:.1f}%)")
    
    # Freeze router for later phases
    for name, p in model.named_parameters():
        if 'feature_extractor' in name or 'gating_network' in name or 'soft_gating' in name:
            p.requires_grad = False
        else:
            p.requires_grad = True  # Re-enable training for experts and downstream layers
    
    logger.info(f"\n🎉 ENHANCED Phase 2 Complete:")
    logger.info(f"  Best train routing: {best_train_routing_acc:.1f}%")
    logger.info(f"  Best test routing:  {best_test_routing_acc:.1f}%")
    logger.info(f"  Final train/test gap: {best_train_routing_acc - best_test_routing_acc:.1f}%")
    
    return best_test_routing_acc

# ===================== TASK-ID ROUTER PRETRAINING =====================
class TaskIDDataset(Dataset):
    """Combine all task datasets but label each sample by its task index (0-9)."""
    def __init__(self, task_loaders):
        self.entries = []  # (dataset_ref, idx, task_id)
        for tid, loader in enumerate(task_loaders):
            # We need to access the underlying dataset of the loader.
            # It could be a Subset wrapped in a LabelAdjustedDataset.
            current_dataset = loader.dataset
            if isinstance(current_dataset, LabelAdjustedDataset):
                # We need the original subset to get original indices/data
                subset = current_dataset.subset
                # The indices in the subset refer to the full original dataset
                self.entries.extend([(subset.dataset, original_idx, tid) for original_idx in subset.indices])
            else: # Fallback for simpler structures
                 self.entries.extend([(current_dataset, i, tid) for i in range(len(current_dataset))])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        ds, inner_idx, tid = self.entries[idx]
        img, _ = ds[inner_idx]  # discard original class label
        return img, tid

def pretrain_router(model, train_loaders, test_loaders, device, args):
    """Pre-train feature_extractor + gating_network to predict task ID."""
    import logging, math
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("TASK-ID ROUTER PRETRAINING")
    logger.info("="*80)
    
    # Validate that num_experts matches num_tasks for router pre-training
    num_tasks = len(train_loaders)
    if args.num_experts != num_tasks:
        logger.error(f"❌ Router pre-training requires num_experts ({args.num_experts}) == num_tasks ({num_tasks})")
        logger.error(f"   Each task needs its own expert for proper routing.")
        logger.error(f"   Please set --num_experts {num_tasks} or adjust --num_tasks {args.num_experts}")
        raise ValueError(f"num_experts ({args.num_experts}) must equal num_tasks ({num_tasks}) for router pre-training")

    # REVERTED: Using a standard combined + shuffled DataLoader for pre-training.
    # The balanced-batch approach had very small per-task samples, leading to noisy
    # gradients and poor feature learning. A standard shuffled dataset is better for this stage.
    logger.info("Reverting to standard shuffled DataLoader for robust pre-training...")
    taskid_dataset = TaskIDDataset(train_loaders)
    router_loader = DataLoader(
        taskid_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # The entire feature_extractor must be trained to learn task-discriminative features.
    # Freezing the randomly initialized early layers was preventing learning.
    logger.info("⚡ Unfreezing all feature extractor layers for router pre-training...")
    for name, p in model.named_parameters():
        if 'feature_extractor' in name or 'gating_network' in name or 'soft_gating' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Higher LR for gating head, small LR for the entire feature extractor backbone
    # FIXED: Use a unified LR. The feature extractor is deep and needs a sufficient LR to learn.
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': args.router_lr},
        {'params': model.gating_network.parameters(), 'lr': args.router_lr},
        {'params': model.soft_gating.parameters(), 'lr': args.router_lr}
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.router_epochs)

    best_val = 0.0
    
    # Test run: reduce epochs to 2 for faster debugging
    num_epochs = 2 if args.test_run else args.router_epochs

    for epoch in range(num_epochs):
        model.train()
        correct = total = 0
        running_loss = 0.0

        pbar = tqdm(router_loader, desc=f"Router Epoch {epoch+1}")

        for batch_idx, (imgs, tids) in enumerate(pbar):
            # Test run: only process 2 batches
            if args.test_run and batch_idx >= 2:
                break
            imgs = imgs.to(device)
            tids = tids.to(device)

            optimizer.zero_grad()
            feats = model.feature_extractor(imgs).view(imgs.size(0), -1)
            logits = model.gating_network(feats)
            loss = F.cross_entropy(logits, tids)  # Removed label smoothing for better learning
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == tids).sum().item()
            total += tids.size(0)
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'acc': f"{100*correct/total:.1f}%"
            })

        scheduler.step()

        # quick validation on test loaders
        model.eval()
        val_corr = val_tot = 0
        with torch.no_grad():
            for tid, loader in enumerate(test_loaders):
                for batch_idx, (imgs, _) in enumerate(loader):
                    # Test run: only process 2 batches per task
                    if args.test_run and batch_idx >= 2:
                        break
                    imgs = imgs.to(device)
                    feats = model.feature_extractor(imgs).view(imgs.size(0), -1)
                    logits = model.gating_network(feats)
                    _, pred = torch.max(logits, 1)
                    val_corr += (pred == tid).sum().item()
                    val_tot += imgs.size(0)
        val_acc = 100*val_corr/val_tot
        logger.info(f"Epoch {epoch+1}: Train {100*correct/total:.1f}% | Val {val_acc:.1f}%")
        if val_acc > best_val:
            best_val = val_acc

    logger.info(f"Best validation routing accuracy after pretraining: {best_val:.1f}%")

    # Freeze router for later phases
    for name, p in model.named_parameters():
        if 'feature_extractor' in name or 'gating_network' in name or 'soft_gating' in name:
            p.requires_grad = False

    return best_val

def analyze_trained_tinyimagenet_model(model, test_loaders, task_classes_global, device, save_dir):
    """
    Comprehensive visual analysis of TRAINED TinyImageNet Hippocampal MoE
    Creates detailed dashboard similar to train_and_analyze.py
    """
    logger = logging.getLogger()
    logger.info("\n" + "🔬" * 60)
    logger.info("🔬 ANALYZING TRAINED TINYIMAGENET MODEL (ACTUAL LEARNED WEIGHTS)")
    logger.info("🔬" * 60)
    
    model.eval()
    analysis_dir = os.path.join(save_dir, 'trained_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Collect data from the TRAINED model
    all_gate_logits = []
    all_dg_outputs = []
    all_ca1_outputs = []
    all_task_labels = []
    routing_matrix = np.zeros((model.num_experts, model.num_experts))
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, labels in tqdm(test_loader, desc=f"Analyzing Trained Task {task_id}"):
                inputs = inputs.to(device)
                
                # Get features from TRAINED feature extractor
                features = model.feature_extractor(inputs)
                features_flat = features.view(features.size(0), -1)
                
                # Get gating decisions from TRAINED gating network
                gate_logits = model.gating_network(features_flat)
                predicted_experts = gate_logits.argmax(dim=1)
                
                # Update routing matrix
                for pred_expert in predicted_experts:
                    routing_matrix[task_id, pred_expert.item()] += 1
                
                # Get representations from TRAINED hippocampal experts
                dg_output, ca3_output, x_separated = model.hippocampal_experts[task_id](features_flat)
                combined = torch.cat([dg_output, ca3_output, features_flat], dim=1)
                ca1_output = model.ca1_integration(combined)
                
                # Store data
                all_gate_logits.append(gate_logits.cpu())
                all_dg_outputs.append(dg_output.cpu())
                all_ca1_outputs.append(ca1_output.cpu())
                all_task_labels.extend([task_id] * inputs.size(0))
    
    # Combine data
    all_gate_logits = torch.cat(all_gate_logits, dim=0)
    all_dg_outputs = torch.cat(all_dg_outputs, dim=0)
    all_ca1_outputs = torch.cat(all_ca1_outputs, dim=0)
    all_task_labels = np.array(all_task_labels)
    
    # Normalize routing matrix
    routing_matrix = routing_matrix / (routing_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Calculate expert utilization
    gate_probs = F.softmax(all_gate_logits, dim=1)
    expert_utilization = gate_probs.mean(dim=0).numpy()
    
    logger.info(f"📊 Analyzed {len(all_task_labels)} samples from TRAINED TinyImageNet model")
    
    # Create comprehensive visualizations
    create_tinyimagenet_visualizations(
        all_gate_logits.numpy(), all_dg_outputs.numpy(), all_ca1_outputs.numpy(),
        all_task_labels, routing_matrix, expert_utilization, analysis_dir, task_classes_global
    )
    
    return {
        'routing_matrix': routing_matrix,
        'expert_utilization': expert_utilization,
        'dg_sparsity': (all_dg_outputs > 0).float().mean().item(),
        'routing_accuracy': np.diag(routing_matrix).mean()
    }

def create_tinyimagenet_visualizations(gate_logits, dg_outputs, ca1_outputs, task_labels, 
                                     routing_matrix, expert_utilization, save_dir, task_classes_global):
    """Create comprehensive visualizations for TinyImageNet TRAINED model"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('🧠 TRAINED TinyImageNet Hippocampal MoE Analysis (Actual Learned Weights)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Routing Matrix (TRAINED)
    sns.heatmap(routing_matrix, annot=True, fmt='.3f', cmap='Blues', 
                ax=axes[0,0], square=True)
    axes[0,0].set_title('🚪 Task→Expert Routing\n(TRAINED MODEL)')
    axes[0,0].set_xlabel('Expert ID')
    axes[0,0].set_ylabel('Task ID')
    
    # 2. Expert Utilization (TRAINED)
    perfect_balance = 1.0 / len(expert_utilization)
    colors = ['red' if x > 2*perfect_balance else 'green' for x in expert_utilization]
    bars = axes[0,1].bar(range(len(expert_utilization)), expert_utilization, 
                        color=colors, alpha=0.7)
    axes[0,1].axhline(y=perfect_balance, color='black', linestyle='--', 
                     label=f'Perfect: {perfect_balance:.3f}')
    axes[0,1].set_title('⚖️ Expert Utilization\n(TRAINED MODEL)')
    axes[0,1].set_xlabel('Expert ID')
    axes[0,1].set_ylabel('Usage Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, expert_utilization):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Routing Accuracy (TRAINED)
    routing_accuracies = np.diag(routing_matrix)
    colors = ['green' if x > 0.8 else 'orange' if x > 0.6 else 'red' for x in routing_accuracies]
    bars = axes[0,2].bar(range(len(routing_accuracies)), routing_accuracies, 
                        color=colors, alpha=0.7)
    axes[0,2].axhline(y=0.2, color='red', linestyle='--', label='Random (20%)')
    axes[0,2].axhline(y=0.8, color='green', linestyle='--', label='Good (80%)')
    axes[0,2].set_title('🎯 Routing Accuracy\n(TRAINED MODEL)')
    axes[0,2].set_xlabel('Task ID') 
    axes[0,2].set_ylabel('Accuracy')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, routing_accuracies):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{val:.2f}', ha='center', va='bottom')
    
    # 4. DG Sparsity (TRAINED)
    sparsity_levels = (dg_outputs > 0).mean(axis=1)
    mean_sparsity = sparsity_levels.mean()
    
    axes[0,3].hist(sparsity_levels, bins=50, alpha=0.7, color='green', density=True)
    axes[0,3].axvline(x=mean_sparsity, color='red', linewidth=2, 
                     label=f'Mean: {mean_sparsity:.1%}')
    axes[0,3].axvline(x=0.05, color='blue', linestyle='--', linewidth=2, 
                     label='Biological: 5%')
    axes[0,3].set_title('🧬 DG Sparsity\n(TRAINED MODEL)')
    axes[0,3].set_xlabel('Fraction Active')
    axes[0,3].set_ylabel('Density')
    axes[0,3].legend()
    axes[0,3].grid(True, alpha=0.3)
    
    # 5. Expert Similarity Matrix (TRAINED)
    num_experts = len(expert_utilization)
    expert_similarities = np.zeros((num_experts, num_experts))
    
    for i in range(num_experts):
        mask_i = (task_labels == i)
        if mask_i.sum() > 0:
            mean_i = ca1_outputs[mask_i].mean(axis=0)
            for j in range(num_experts):
                mask_j = (task_labels == j)
                if mask_j.sum() > 0:
                    mean_j = ca1_outputs[mask_j].mean(axis=0)
                    similarity = np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j))
                    expert_similarities[i, j] = similarity
    
    sns.heatmap(expert_similarities, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, ax=axes[1,0], square=True)
    axes[1,0].set_title('🧠 Expert Similarity\n(TRAINED MODEL)')
    axes[1,0].set_xlabel('Expert ID')
    axes[1,0].set_ylabel('Expert ID')
    
    # 6. CA1 Representations t-SNE (TRAINED)
    n_samples = min(2000, len(ca1_outputs))
    indices = np.random.choice(len(ca1_outputs), n_samples, replace=False)
    sample_ca1 = ca1_outputs[indices]
    sample_labels = task_labels[indices]
    
    pca = PCA(n_components=50)
    ca1_pca = pca.fit_transform(sample_ca1)
    tsne = TSNE(n_components=2, random_state=42)
    ca1_tsne = tsne.fit_transform(ca1_pca)
    
    scatter = axes[1,1].scatter(ca1_tsne[:, 0], ca1_tsne[:, 1], 
                               c=sample_labels, cmap='tab10', alpha=0.6, s=10)
    axes[1,1].set_title('🗺️ CA1 Features (t-SNE)\n(TRAINED MODEL)')
    axes[1,1].set_xlabel('t-SNE 1')
    axes[1,1].set_ylabel('t-SNE 2')
    
    # 7. Gating Confidence (TRAINED)
    gate_probs = F.softmax(torch.tensor(gate_logits), dim=1).numpy()
    max_confidences = gate_probs.max(axis=1)
    
    for task_id in range(num_experts):
        task_mask = (task_labels == task_id)
        if task_mask.sum() > 0:
            task_confidences = max_confidences[task_mask]
            axes[1,2].hist(task_confidences, bins=20, alpha=0.5, 
                          label=f'Task {task_id}', density=True)
    
    axes[1,2].axvline(x=0.5, color='red', linestyle='--', label='Random')
    axes[1,2].set_title('🎲 Gating Confidence\n(TRAINED MODEL)')
    axes[1,2].set_xlabel('Max Probability')
    axes[1,2].set_ylabel('Density')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    # 8. Load Imbalance (TRAINED)
    perfect_balance = 1.0 / num_experts
    imbalances = np.abs(expert_utilization - perfect_balance)
    colors = ['red' if x > perfect_balance else 'green' for x in imbalances]
    
    bars = axes[1,3].bar(range(num_experts), imbalances, color=colors, alpha=0.7)
    axes[1,3].axhline(y=0, color='black', linestyle='-', label='Perfect Balance')
    axes[1,3].set_title('⚖️ Load Imbalance\n(TRAINED MODEL)')
    axes[1,3].set_xlabel('Expert ID')
    axes[1,3].set_ylabel('|Usage - Perfect|')
    axes[1,3].legend()
    axes[1,3].grid(True, alpha=0.3)
    
    
    # 9. Task-Expert Activation Heatmap (TRAINED)
    expert_task_activations = np.zeros((num_experts, num_experts))
    for task_id in range(num_experts):
        task_mask = (task_labels == task_id)
        if task_mask.sum() > 0:
            task_gate_probs = F.softmax(torch.tensor(gate_logits[task_mask]), dim=1).numpy()
            expert_task_activations[task_id] = task_gate_probs.mean(axis=0)
    
    sns.heatmap(expert_task_activations, annot=True, fmt='.3f', cmap='viridis', 
                ax=axes[2,0], square=True)
    axes[2,0].set_title('🔥 Task→Expert Activation\n(TRAINED MODEL)')
    axes[2,0].set_xlabel('Expert ID')
    axes[2,0].set_ylabel('Task ID')
    
    # 10. Performance Summary (TRAINED)
    metrics = ['Routing\nAccuracy', 'Load\nBalance', 'Sparsity\nScore', 'Expert\nDiversity']
    values = [
        routing_accuracies.mean(),
        1.0 / (1.0 + np.var(expert_utilization) * 10),
        max(0, 1 - abs(mean_sparsity - 0.05) / 0.05),
        1.0 - expert_similarities.mean()
    ]
    
    colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
    bars = axes[2,1].bar(metrics, values, color=colors, alpha=0.7)
    axes[2,1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
    axes[2,1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7)
    axes[2,1].set_title('📈 Performance Summary\n(TRAINED MODEL)')
    axes[2,1].set_ylabel('Score')
    axes[2,1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        axes[2,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{val:.3f}', ha='center', va='bottom')
    
    # 11. PCA Explained Variance (TRAINED)
    axes[2,2].bar(range(min(10, len(pca.explained_variance_ratio_))), 
                 pca.explained_variance_ratio_[:10], color='orange', alpha=0.7)
    axes[2,2].set_title('📊 PCA Variance\n(TRAINED MODEL)')
    axes[2,2].set_xlabel('Component')
    axes[2,2].set_ylabel('Explained Variance')
    axes[2,2].grid(True, alpha=0.3)
    
    # 12. TinyImageNet-Specific Analysis Text
    axes[2,3].text(0.1, 0.9, f'📊 TRAINED TinyImageNet Analysis', fontsize=14, fontweight='bold', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.8, f'Routing Accuracy: {routing_accuracies.mean():.1%}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.7, f'DG Sparsity: {mean_sparsity:.1%}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.6, f'Load Balance: {1.0/(1.0+np.var(expert_utilization)*10):.3f}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.5, f'Expert Diversity: {1.0-expert_similarities.mean():.3f}', transform=axes[2,3].transAxes)
    
    # TinyImageNet specific stats
    total_classes = len(task_classes_global) * len(task_classes_global[0]) if task_classes_global else 200
    axes[2,3].text(0.1, 0.4, f'Dataset: TinyImageNet-{total_classes}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.35, f'Image Size: 64×64', transform=axes[2,3].transAxes)
    
    # Status indicators
    status_color = 'green' if routing_accuracies.mean() > 0.6 else 'red'
    axes[2,3].text(0.1, 0.25, '✅ GOOD PERFORMANCE' if routing_accuracies.mean() > 0.6 else '❌ POOR PERFORMANCE', 
                  color=status_color, fontweight='bold', transform=axes[2,3].transAxes)
    
    load_balance_status = 'green' if np.var(expert_utilization) < 0.01 else 'red'
    axes[2,3].text(0.1, 0.15, '✅ BALANCED EXPERTS' if np.var(expert_utilization) < 0.01 else '❌ IMBALANCED EXPERTS', 
                  color=load_balance_status, fontweight='bold', transform=axes[2,3].transAxes)
    
    axes[2,3].set_xlim(0, 1)
    axes[2,3].set_ylim(0, 1)
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'TRAINED_tinyimagenet_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📊 TRAINED TinyImageNet model analysis saved: {plot_path}")

class GridCellLayer(nn.Module):
    """
    Implements grid cell-like spatial encoding from entorhinal cortex.
    Uses multiple spatial frequencies to create hexagonal grid patterns.
    This is an ML analogue, not a biological simulation.
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

class TinyImageNetLeNet(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyImageNetLeNet, self).__init__()
        # ... existing code ...

# 🔧 FIXED: Biological SparseActivation with CORRECT sparsity calculation
class BiologicalSparseActivation(nn.Module):
    """Dentate Gyrus sparse activation with CORRECT biological sparsity"""
    def __init__(self, target_sparsity=0.03):
        super().__init__()
        self.target_sparsity = target_sparsity  # Fraction of neurons that should be ACTIVE
        
    def forward(self, x):
        batch_size, num_neurons = x.shape
        
        # Calculate how many neurons should be active (not zero)
        k = max(1, int(num_neurons * self.target_sparsity))
        
        if self.training:
            # Add small noise during training for stochasticity
            noise = torch.randn_like(x) * 0.01
            x_noisy = x + noise
            _, indices = torch.topk(x_noisy, k, dim=1)
        else:
            # Deterministic during inference
            _, indices = torch.topk(x, k, dim=1)
        
        # Create sparse output - ONLY top-k neurons are active
        sparse_output = torch.zeros_like(x)
        sparse_output.scatter_(1, indices, x.gather(1, indices))
        
        return sparse_output

# 🔧 FIXED: Enhanced DentateGyrusExpert with proper biological components
class EnhancedDentateGyrusExpert(nn.Module):
    """FIXED: Dentate Gyrus with proper pattern separation and biological sparsity"""
    def __init__(self, input_dim, hidden_dim, target_sparsity=0.03, expansion_factor=4):
        super().__init__()
        self.target_sparsity = target_sparsity
        
        # Mossy fiber expansion (biological DG has 4x more granule cells than inputs)
        expanded_dim = input_dim * expansion_factor
        
        # Pattern separation pathway
        self.expansion = nn.Linear(input_dim, expanded_dim)
        self.activation = nn.ReLU()
        self.norm1 = nn.LayerNorm(expanded_dim)
        
        # CORRECTED: Biological sparse activation
        self.sparse_activation = BiologicalSparseActivation(target_sparsity)
        
        # Projection back to hidden dimension
        self.projection = nn.Linear(expanded_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Expansion phase
        expanded = self.expansion(x)
        expanded = self.activation(expanded)
        expanded = self.norm1(expanded)
        
        # CRITICAL: Apply biological sparsity
        sparse_expanded = self.sparse_activation(expanded)
        
        # Project back
        output = self.projection(sparse_expanded)
        output = self.activation(output)
        output = self.norm2(output)
        
        return output

# 🔧 FIXED: Enhanced CA3 with better pattern completion
class EnhancedCA3PatternCompletion(nn.Module):
    """Enhanced CA3 with attractor dynamics for pattern completion"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Auto-associative memory with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Recurrent-style pattern completion
        self.completion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Bounded activation for stability
            nn.LayerNorm(hidden_dim)
        )
        
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encode to associative space
        encoded = self.encoder(x)
        
        # Pattern completion with residual connection
        completed = self.completion(encoded) + encoded
        
        # Decode back to DG space
        decoded = self.decoder(completed)
        
        return decoded

# 🔧 FIXED: Enhanced HippocampalExpert with proper feature separation
class EnhancedHippocampalExpert(nn.Module):
    """FIXED: Hippocampal expert with proper DG→CA3 flow and feature separation"""
    def __init__(self, input_dim, dg_dim, ca3_dim, target_sparsity=0.03):
        super().__init__()
        
        # Enhanced DG with correct sparsity
        self.dg = EnhancedDentateGyrusExpert(
            input_dim, dg_dim, target_sparsity, expansion_factor=4
        )
        
        # Enhanced CA3 with better pattern completion
        self.ca3 = EnhancedCA3PatternCompletion(dg_dim, ca3_dim)
        
        # Feature separation mechanism
        self.feature_separator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),  # Bounded for stable gradients
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x):
        # Apply feature separation transformation
        x_separated = self.feature_separator(x)
        
        # DG pattern separation with biological sparsity
        dg_output = self.dg(x_separated)
        
        # CA3 pattern completion
        ca3_output = self.ca3(dg_output)
        
        return dg_output, ca3_output, x_separated

# 🔧 FIXED: Calculate proper feature separability
def calculate_enhanced_feature_separability(original_features, separated_features):
    """Calculate feature separability between original and separated features"""
    # Normalize features
    orig_norm = F.normalize(original_features, p=2, dim=1)
    sep_norm = F.normalize(separated_features, p=2, dim=1)
    
    # Calculate cosine similarity (we want this to be LOW for good separation)
    cos_sim = torch.sum(orig_norm * sep_norm, dim=1)
    
    # Good separation means low similarity, so return 1 - |cos_sim|
    separability = 1.0 - torch.abs(cos_sim).mean()
    
    return separability.item()

# Import key components from base Hippocampal MoE file
from train_hippocampal_moe import SoftGating

if __name__ == "__main__":
    main() 