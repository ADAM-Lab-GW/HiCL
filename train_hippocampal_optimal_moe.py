#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
from datetime import datetime
import copy

# Import our Enhanced Hippocampal MoE architecture
from train_hippocampal_moe import (
    HippocampalMoE, GridCellLayer, SparseActivation, SoftGating,
    DentateGyrusExpert, CA3PatternCompletion, HippocampalExpert,
    HippocampalGatingNetwork, set_seed, count_model_flops
)

# ============================================================================
# ADVANCED MOE STRATEGIES
# ============================================================================

def calculate_load_balancing_loss(gate_logits, num_experts):
    """
    Load Balancing Loss: Encourages even distribution of inputs across experts.
    Based on Switch Transformer load balancing from Google Research.
    """
    if gate_logits is None:
        return torch.tensor(0.0, device=gate_logits.device if torch.is_tensor(gate_logits) else 'cpu')
    
    # Convert to probabilities
    gate_probs = F.softmax(gate_logits, dim=1)
    
    # Calculate fraction of examples routed to each expert
    fraction_routed = torch.mean(gate_probs, dim=0)
    
    # Load balancing loss = num_experts * sum(fraction_i^2)
    # This penalizes uneven distributions
    loss = num_experts * torch.sum(fraction_routed * fraction_routed)
    return loss

def calculate_diversity_loss(expert_outputs):
    """
    Expert Diversity Loss: Encourages experts to learn different representations.
    Minimizes cosine similarity between expert outputs.
    """
    if expert_outputs is None or expert_outputs.shape[1] <= 1:
        return torch.tensor(0.0, device=expert_outputs.device if torch.is_tensor(expert_outputs) else 'cpu')
    
    # expert_outputs shape: (batch_size, num_experts, feature_dim)
    # Normalize along feature dimension
    normalized_outputs = F.normalize(expert_outputs, p=2, dim=2)
    
    # Calculate pairwise cosine similarities
    # (batch_size, num_experts, num_experts)
    similarity_matrix = torch.bmm(normalized_outputs, normalized_outputs.transpose(1, 2))
    
    # Take upper triangle (excluding diagonal) and average
    # We want to minimize similarity between different experts
    batch_size, num_experts, _ = similarity_matrix.shape
    mask = torch.triu(torch.ones(num_experts, num_experts), diagonal=1).to(similarity_matrix.device)
    
    # Average similarity across batch and expert pairs
    diversity_loss = torch.mean(similarity_matrix * mask.unsqueeze(0))
    return diversity_loss

def create_balanced_joint_loader(task_loaders, max_batches_per_epoch=100):
    """
    Balanced Sampling: Creates batches with equal representation from each task.
    This prevents the model from becoming biased toward later tasks during joint training.
    """
    def balanced_generator():
        # Create iterators for each task
        task_iters = [iter(loader) for loader in task_loaders]
        num_tasks = len(task_loaders)
        batch_count = 0
        
        while batch_count < max_batches_per_epoch:
            batch_inputs = []
            batch_labels = []
            batch_task_ids = []
            
            # Sample one batch from each task
            all_exhausted = True
            for task_id, task_iter in enumerate(task_iters):
                try:
                    inputs, labels = next(task_iter)
                    batch_inputs.append(inputs)
                    batch_labels.append(labels)
                    batch_task_ids.extend([task_id] * inputs.size(0))
                    all_exhausted = False
                except StopIteration:
                    # Restart this task's iterator
                    task_iters[task_id] = iter(task_loaders[task_id])
                    try:
                        inputs, labels = next(task_iters[task_id])
                        batch_inputs.append(inputs)
                        batch_labels.append(labels)
                        batch_task_ids.extend([task_id] * inputs.size(0))
                        all_exhausted = False
                    except StopIteration:
                        # Task loader is completely empty, skip
                        continue
            
            if all_exhausted or not batch_inputs:
                break
                
            # Combine all task batches
            combined_inputs = torch.cat(batch_inputs, dim=0)
            combined_labels = torch.cat(batch_labels, dim=0)
            combined_task_ids = torch.tensor(batch_task_ids)
            
            yield combined_inputs, combined_labels, combined_task_ids
            batch_count += 1
    
    return balanced_generator

class HippocampalReplayBuffer:
    """Biologically-inspired replay buffer using hippocampal pattern storage"""
    def __init__(self, capacity_per_task=200, prioritized=True):
        self.capacity_per_task = capacity_per_task
        self.prioritized = prioritized
        self.buffer = defaultdict(list)  # Task-specific buffers
        self.priorities = defaultdict(list)  # Task-specific priorities
        
    def add_sample(self, inputs, labels, task_id, loss=None):
        """Add sample to hippocampal memory"""
        # Ensure task_id is always a Python int
        if isinstance(task_id, torch.Tensor):
            if task_id.dim() == 0:
                task_id = task_id.item()
            elif task_id.numel() == 1:
                task_id = task_id.view(-1)[0].item()
            else:
                task_id = int(task_id.cpu().numpy().flatten()[0])
        sample = (inputs.cpu().detach(), labels.cpu().detach())
        priority = loss.item() if loss is not None else 1.0
        
        # If buffer full, remove lowest priority sample
        if len(self.buffer[task_id]) >= self.capacity_per_task:
            if self.prioritized and self.priorities[task_id]:
                min_idx = np.argmin(self.priorities[task_id])
                self.buffer[task_id].pop(min_idx)
                self.priorities[task_id].pop(min_idx)
            else:
                self.buffer[task_id].pop(0)  # FIFO
                if self.priorities[task_id]:
                    self.priorities[task_id].pop(0)
        
        self.buffer[task_id].append(sample)
        self.priorities[task_id].append(priority)
    
    def sample_replay(self, batch_size=32, exclude_task=None):
        """Sample from all tasks except current one"""
        all_samples = []
        all_task_ids = []

        # Robustly convert exclude_task to int if it's a tensor
        if exclude_task is not None and isinstance(exclude_task, torch.Tensor):
            print(f"[DEBUG] exclude_task type: {type(exclude_task)}, value: {exclude_task}")
            try:
                exclude_task = exclude_task.item()
            except Exception:
                exclude_task = int(exclude_task.cpu().numpy().flatten()[0])

        for task_id, samples in self.buffer.items():
            # Ensure both task_id and exclude_task are Python ints for safe comparison
            if isinstance(task_id, torch.Tensor):
                if task_id.dim() == 0:
                    task_id = task_id.item()
                elif task_id.numel() == 1:
                    task_id = task_id.view(-1)[0].item()
                else:
                    task_id = int(task_id.cpu().numpy().flatten()[0])
            
            if exclude_task is not None and task_id == exclude_task:
                continue
            if not samples:
                continue
                
            # Sample from this task based on priorities
            if self.prioritized and self.priorities[task_id]:
                priorities = np.array(self.priorities[task_id])
                probs = priorities / priorities.sum()
                num_samples = min(batch_size // len(self.buffer), len(samples))
                if num_samples > 0:
                    indices = np.random.choice(len(samples), size=num_samples, 
                                             replace=False, p=probs)
                    for idx in indices:
                        all_samples.append(samples[idx])
                        all_task_ids.append(task_id)
            else:
                # Random sampling
                num_samples = min(batch_size // len(self.buffer), len(samples))
                if num_samples > 0:
                    indices = np.random.choice(len(samples), size=num_samples, replace=False)
                    sampled = [samples[i] for i in indices]
                    for sample in sampled:
                        all_samples.append(sample)
                        all_task_ids.append(task_id)
        
        if not all_samples:
            return None, None, None
            
        # Convert back to tensors - ensure all have same batch size
        if all_samples:
            # Get the target batch size from the first sample
            target_batch_size = all_samples[0][0].size(0)
            
            # Pad or truncate all samples to match target batch size
            normalized_samples = []
            for sample in all_samples:
                inputs, labels = sample
                if inputs.size(0) < target_batch_size:
                    # Pad with the last sample
                    pad_size = target_batch_size - inputs.size(0)
                    inputs = torch.cat([inputs, inputs[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
                    labels = torch.cat([labels, labels[-1:].repeat(pad_size)], dim=0)
                elif inputs.size(0) > target_batch_size:
                    # Truncate to target size
                    inputs = inputs[:target_batch_size]
                    labels = labels[:target_batch_size]
                normalized_samples.append((inputs, labels))
            
            # Concatenate all samples instead of stacking to avoid extra dimension
            inputs = torch.cat([s[0] for s in normalized_samples], dim=0)
            labels = torch.cat([s[1] for s in normalized_samples], dim=0)
            task_ids = torch.tensor(all_task_ids)
        else:
            return None, None, None
        
        return inputs, labels, task_ids

# ============================================================================
# HIPPOCAMPAL EXPERTS (DG + CA3)
# ============================================================================

class EnhancedHippocampalExpert(nn.Module):
    """
    An enhanced Hippocampal Expert that now includes its own CA1 integration.
    This makes each expert a self-contained processing stream from DG to CA1.
    """
    def __init__(self, input_dim, dg_dim, ca3_dim, feature_dim, target_sparsity=0.05):
        super().__init__()
        self.dg = DentateGyrusExpert(input_dim, dg_dim, target_sparsity)
        self.ca3 = CA3PatternCompletion(dg_dim, ca3_dim)
        
        # Each expert now has its own CA1 integration layer.
        self.ca1_integration = nn.Sequential(
            nn.Linear(dg_dim + ca3_dim + feature_dim, 256),  # DG + CA3 + direct entorhinal bypass
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )
        
    def forward(self, features):
        # The expert now takes the raw features from the feature_extractor
        dg_output = self.dg(features)
        ca3_output = self.ca3(dg_output)
        
        # CA1 Integration with MEC->CA1 direct bypass
        combined = torch.cat([dg_output, ca3_output, features], dim=1)
        ca1_output = self.ca1_integration(combined)
        
        return dg_output, ca1_output # Return DG for contrastive loss, and CA1 for final output

# ============================================================================
# ENHANCED HIPPOCAMPAL MOE WITH ADVANCED STRATEGIES
# ============================================================================

class OptimalHippocampalMoE(HippocampalMoE):
    """
    An optimal MoE with a more powerful feature extractor and advanced training.
    NOW uses a LeNet-style feature extractor with Grid Cells.
    """
    def __init__(self, num_experts, classes_per_task, input_channels, 
                 feature_dim=256, dg_dim=512, ca3_dim=256, ca1_dim=128, target_sparsity=0.05):
        # Call parent init but we will overwrite all major components
        super().__init__(num_experts=num_experts, classes_per_task=classes_per_task, input_channels=input_channels)
        
        # --- FEATURE EXTRACTOR ---
        # Replace the ResNet feature extractor with a LeNet-style one
        logging.info("🧠 Using LeNet-style feature extractor with Grid Cells.")
        self.feature_extractor = nn.Sequential(
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
        
        # Dynamically calculate the output dimension of the new feature extractor
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.feature_extractor(dummy_input)
            feature_extractor_output_dim = dummy_output.numel()

        logging.info(f"  - LeNet feature extractor output dimension: {feature_extractor_output_dim}")

        # --- HIPPOCAMPAL EXPERTS ---
        # Re-create experts to match the new feature extractor's output dimension
        self.hippocampal_experts = nn.ModuleList([
            EnhancedHippocampalExpert(
                input_dim=feature_extractor_output_dim, 
                dg_dim=dg_dim, 
                ca3_dim=ca3_dim, 
                feature_dim=ca1_dim, # feature_dim is the output dim of the CA1 layer
                target_sparsity=target_sparsity
            ) for _ in range(num_experts)
        ])

        # --- GATING NETWORK ---
        # Re-create gating network to match the new feature extractor's output
        self.gating_network = HippocampalGatingNetwork(
            input_dim=feature_extractor_output_dim,
            num_experts=num_experts
        )

        # --- OUTPUT LAYERS ---
        # Re-create output layers, they depend on the CA1 output from the expert
        self.output_layers = nn.ModuleList([
            nn.Linear(ca1_dim, classes_per_task) for _ in range(num_experts)
        ])

        # Store task classes for later use
        self.task_classes = []

    def set_task_classes(self, task_classes):
        self.task_classes = task_classes
        
    def forward(self, x, task_id=None, return_expert_outputs=False):
        """
        Enhanced forward pass with optional expert output return for diversity loss
        """
        # Feature Extraction + Grid Cells (Entorhinal Cortex)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Gating Network
        gate_logits = self.gating_network(features)
        gate_weights = self.soft_gating(gate_logits, training=self.training)
        
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        if task_id is not None:
            # Task-specific inference (oracle)
            dg_output, ca1_output = self.hippocampal_experts[task_id](features)
            
            # Output
            expert_outputs = self.output_layers[task_id](ca1_output)
            start_idx = task_id * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            final_outputs[:, start_idx:end_idx] = expert_outputs
            
            if return_expert_outputs:
                # Return single expert output in proper format (use CA1 output)
                all_expert_outputs = torch.zeros(batch_size, self.num_experts, ca1_output.size(1), device=x.device)
                all_expert_outputs[:, task_id, :] = ca1_output
                return final_outputs, gate_logits, all_expert_outputs
            else:
                return final_outputs, gate_logits
        else:
            # Soft or hard routing based on training mode
            if self.training:
                # Soft routing during training - weighted combination
                all_expert_ca1_outputs = []
                
                for expert_id in range(self.num_experts):
                    weight = gate_weights[:, expert_id].unsqueeze(1)
                    
                    # Get integrated output from each expert's internal CA1
                    _, ca1_output = self.hippocampal_experts[expert_id](features)
                    all_expert_ca1_outputs.append(ca1_output)
                    
                    # Output
                    expert_outputs = self.output_layers[expert_id](ca1_output)
                    
                    start_idx = expert_id * self.classes_per_task
                    end_idx = (expert_id + 1) * self.classes_per_task
                    final_outputs[:, start_idx:end_idx] += weight * expert_outputs
                
                if return_expert_outputs:
                    # Stack expert CA1 outputs for diversity loss
                    expert_outputs_tensor = torch.stack(all_expert_ca1_outputs, dim=1)
                    return final_outputs, gate_logits, expert_outputs_tensor
                else:
                    return final_outputs, gate_logits
            else:
                # Hard routing for inference
                _, predicted_gates = torch.max(gate_logits, 1)
                
                for expert_id in range(self.num_experts):
                    expert_mask = (predicted_gates == expert_id)
                    if expert_mask.sum() > 0:
                        expert_features = features[expert_mask]
                        
                        # Get integrated output from the expert's internal CA1
                        _, ca1_output = self.hippocampal_experts[expert_id](expert_features)
                        
                        # Output
                        expert_outputs = self.output_layers[expert_id](ca1_output)
                        
                        start_idx = expert_id * self.classes_per_task
                        end_idx = (expert_id + 1) * self.classes_per_task
                        final_outputs[expert_mask, start_idx:end_idx] = expert_outputs
                
                if return_expert_outputs:
                    # For inference, just return empty expert outputs
                    empty_expert_outputs = torch.zeros(batch_size, self.num_experts, 128, device=x.device)
                    return final_outputs, gate_logits, empty_expert_outputs
                else:
                    return final_outputs, gate_logits

def setup_logging():
    """Setup logging configuration for the training run"""
    log_dir_base = "hippocampal_optimal"
    log_dir = f"{log_dir_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    
    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/training.log")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Set logging level to INFO (less verbose)
    root_logger.setLevel(logging.INFO)
    
    # Silence matplotlib font warnings
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    
    return log_dir

def create_task_specific_dataloaders(num_experts=5, classes_per_task=2, batch_size=64):
    """Create task-specific dataloaders like the successful approach"""
    
    # CIFAR-10 transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load full datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)

    # Define task classes (same as successful run)
    task_classes = [
        [0, 1],  # Task 0: airplane, automobile
        [2, 3],  # Task 1: bird, cat
        [4, 5],  # Task 2: deer, dog
        [6, 7],  # Task 3: frog, horse
        [8, 9],  # Task 4: ship, truck
    ]
    
    # Create task-specific datasets
    train_loaders = []
    test_loaders = []
    
    for task_id, classes in enumerate(task_classes):
        # Filter training data for this task
        train_indices = [i for i, (_, label) in enumerate(trainset) if label in classes]
        train_subset = torch.utils.data.Subset(trainset, train_indices)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
        
        # Filter test data for this task
        test_indices = [i for i, (_, label) in enumerate(testset) if label in classes]
        test_subset = torch.utils.data.Subset(testset, test_indices)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=100,
                                                shuffle=False, num_workers=2)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
        logging.info(f"Task {task_id} ({classes}): {len(train_indices)} train, {len(test_indices)} test samples")
    
    return train_loaders, test_loaders, task_classes

def phase1_train_experts_independently(model, train_loaders, test_loaders, device, args):
    """Phase 1: Train each hippocampal expert independently with forced routing + replay"""
    logging.info("\n" + "="*80)
    logging.info("PHASE 1: TRAINING HIPPOCAMPAL EXPERTS INDEPENDENTLY WITH MEMORY REPLAY")
    logging.info("="*80)
    
    # Freeze gating network during expert training
    for param in model.gating_network.parameters():
        param.requires_grad = False
    
    # Initialize hippocampal replay buffer
    replay_buffer = HippocampalReplayBuffer(capacity_per_task=200, prioritized=True)
    logging.info("Hippocampal replay buffer initialized (200 samples/task)")
    
    expert_results = []
    
    for expert_id in range(args.num_experts):
        logging.info(f"\nTraining Hippocampal Expert {expert_id}")
        logging.info(f"   Components: Grid Cells → DG → CA3 → CA1 (Integrated within Expert)")
        
        # Optimizer for this expert + shared feature extractor
        expert_params = list(model.hippocampal_experts[expert_id].parameters()) + \
                       list(model.output_layers[expert_id].parameters())
        shared_params = list(model.feature_extractor.parameters())
        
        # Give a lower learning rate to the shared feature extractor to reduce forgetting
        optimizer = optim.AdamW([
            {'params': shared_params, 'lr': args.learning_rate * 0.1}, 
            {'params': expert_params, 'lr': args.learning_rate}
        ], weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.expert_epochs)
        
        best_acc = 0.0
        train_loader = train_loaders[expert_id]
        test_loader = test_loaders[expert_id]
        
        for epoch in range(args.expert_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}/{args.expert_epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Map global labels to local task labels (0 to classes_per_task-1)
                local_labels = labels % model.classes_per_task

                optimizer.zero_grad()
                
                # Forced routing to the current expert
                outputs, _ = model(inputs, task_id=expert_id)
                
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                # Only classification loss - no gating loss during expert training
                current_loss = F.cross_entropy(task_outputs, local_labels)
                total_loss = current_loss
                
                # Add samples to replay buffer
                for i in range(inputs.size(0)):
                    replay_buffer.add_sample(inputs[i], labels[i], expert_id, current_loss)
                
                # Memory replay from previous tasks
                replay_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only replay for tasks 1+
                    replay_inputs, replay_labels, replay_task_ids = replay_buffer.sample_replay(
                        batch_size=16, exclude_task=expert_id
                    )
                    
                    if replay_inputs is not None:
                        replay_inputs = replay_inputs.to(device)
                        replay_labels = replay_labels.to(device)
                        
                        # Process replay samples through their respective experts
                        for r_task_id in torch.unique(replay_task_ids):
                            task_mask = (replay_task_ids == r_task_id)
                            if task_mask.sum() > 0:
                                r_inputs = replay_inputs[task_mask]
                                r_labels = replay_labels[task_mask]
                                
                                # Map to local labels for replay task
                                r_local_labels = r_labels % model.classes_per_task
                                
                                # Forward through the replay task's expert
                                r_outputs, _ = model(r_inputs, task_id=r_task_id.item())
                                start_idx = r_task_id * model.classes_per_task
                                end_idx = start_idx + model.classes_per_task
                                r_task_outputs = r_outputs[:, start_idx:end_idx]
                                
                                replay_loss += F.cross_entropy(r_task_outputs, r_local_labels)
                        
                        # Combine current and replay losses
                        total_loss = current_loss + 0.5 * replay_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                _, predicted = torch.max(task_outputs, 1)
                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.3f}",
                    'acc': f"{100.*correct/total:.1f}%",
                    'replay': f"{replay_loss.item():.3f}" if expert_id > 0 else "N/A"
                })
            
            scheduler.step()
            
            # Evaluate this expert
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Map to local labels
                    local_labels = labels % model.classes_per_task
                    
                    # Oracle routing for this task
                    outputs, _ = model(inputs, task_id=expert_id)
                    
                    # Extract outputs for the current task
                    start_idx = expert_id * model.classes_per_task
                    end_idx = start_idx + model.classes_per_task
                    task_outputs = outputs[:, start_idx:end_idx]
                    
                    _, predicted = torch.max(task_outputs, 1)
                    test_correct += (predicted == local_labels).sum().item()
                    test_total += local_labels.size(0)
            
            test_acc = (test_correct / test_total) * 100
            if test_acc > best_acc:
                best_acc = test_acc
            
            avg_loss = epoch_loss / len(train_loader)
            train_acc = (correct / total) * 100
            
            logging.info(f"Expert {expert_id} Epoch {epoch+1}: "
                        f"Loss={avg_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}% (Best: {best_acc:.1f}%)")
        
        expert_results.append({
            'expert_id': expert_id,
            'best_accuracy': best_acc
        })
        
        logging.info(f"Expert {expert_id} final accuracy: {best_acc:.2f}%")
    
    # Unfreeze gating for next phase
    for param in model.gating_network.parameters():
        param.requires_grad = True
    
    logging.info(f"\nPhase 1 Complete - Expert accuracies: {[r['best_accuracy'] for r in expert_results]}")
    return expert_results

def phase2_train_gating_network(model, train_loaders, test_loaders, device, args):
    """Phase 2: Train gating network separately with frozen experts"""
    logging.info("\n" + "="*80)
    logging.info("PHASE 2: TRAINING HIPPOCAMPAL GATING NETWORK")
    logging.info("="*80)
    
    # Freeze ALL parameters except gating network
    for name, param in model.named_parameters():
        if 'gating_network' not in name and 'soft_gating' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Collect all training data with task labels
    all_gating_data = []
    
    logging.info("Collecting features for gating network training...")
    model.eval()
    
    with torch.no_grad():
        for task_id, train_loader in enumerate(train_loaders):
            for inputs, labels in tqdm(train_loader, desc=f"Collecting Task {task_id} features"):
                inputs = inputs.to(device)
                
                # Extract features using current feature extractor
                features = model.feature_extractor(inputs)
                features_flat = features.view(features.size(0), -1)
                
                # Store features with correct task ID
                for i in range(features_flat.size(0)):
                    all_gating_data.append((features_flat[i].cpu(), task_id))
    
    logging.info(f"Collected {len(all_gating_data)} samples for gating training")
    
    # Gating network optimizer with higher learning rate
    gate_optimizer = optim.AdamW([
        {'params': model.gating_network.parameters(), 'lr': args.gate_lr},
        {'params': model.soft_gating.parameters(), 'lr': args.gate_lr * 0.5}
    ], weight_decay=args.weight_decay * 0.1)
    
    gate_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        gate_optimizer, T_0=5, T_mult=2
    )
    
    best_routing_acc = 0.0
    
    for epoch in range(args.gate_epochs):
        model.train()
        
        # Shuffle gating data
        np.random.shuffle(all_gating_data)
        
        epoch_loss = 0.0
        correct_routing = 0
        total_samples = 0
        
        # Process in batches
        batch_size = args.batch_size
        num_batches = len(all_gating_data) // batch_size
        
        progress_bar = tqdm(range(num_batches), desc=f"Gate Epoch {epoch+1}")
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_gating_data))
            batch_data = all_gating_data[start_idx:end_idx]
            
            if len(batch_data) == 0:
                continue
            
            # Prepare batch
            batch_features = torch.stack([data[0] for data in batch_data]).to(device)
            batch_task_ids = torch.tensor([data[1] for data in batch_data], device=device, dtype=torch.long)
            
            gate_optimizer.zero_grad()
            
            # Forward through gating network only
            gate_logits = model.gating_network(batch_features)
            
            # Strong gating loss - force correct routing
            gate_loss = F.cross_entropy(gate_logits, batch_task_ids)
            
            # Additional routing constraints
            gate_probs = F.softmax(gate_logits, dim=1)
            
            # Confidence penalty - encourage confident decisions
            entropy_loss = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1).mean()
            
            # Expert balance loss - encourage using all experts
            expert_usage = gate_probs.mean(dim=0)
            target_usage = torch.ones_like(expert_usage) / args.num_experts
            balance_loss = F.mse_loss(expert_usage, target_usage)
            
            # Total gating loss
            total_loss = gate_loss + 0.1 * entropy_loss + 0.5 * balance_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            gate_optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            _, predicted_experts = torch.max(gate_logits, 1)
            correct_routing += (predicted_experts == batch_task_ids).sum().item()
            total_samples += batch_features.size(0)
            
            # Update progress bar
            current_routing_acc = (correct_routing / total_samples) * 100
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.3f}",
                'routing': f"{current_routing_acc:.1f}%"
            })
        
        gate_scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        routing_acc = (correct_routing / total_samples) * 100 if total_samples > 0 else 0
        
        if routing_acc > best_routing_acc:
            best_routing_acc = routing_acc
        
        logging.info(f"Gate Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                    f"Routing={routing_acc:.1f}% (Best: {best_routing_acc:.1f}%)")
    
    # Save best gating weights before unfreezing all parameters
    best_gating_state = None
    if best_routing_acc > 50.0:  # Only save if routing is reasonably good
        best_gating_state = {
            name: param.clone().detach().cpu() for name, param in model.named_parameters() 
            if 'gating_network' in name or 'soft_gating' in name
        }
        logging.info(f"💾 Best gating weights saved (routing: {best_routing_acc:.1f}%)")
    
    # Unfreeze all parameters for joint training
    for param in model.parameters():
        param.requires_grad = True

    logging.info(f"\nPhase 2 Complete - Best routing accuracy: {best_routing_acc:.1f}%")
    return best_routing_acc

def phase3_joint_fine_tuning(model, train_loaders, test_loaders, device, args):
    """Phase 3: Joint fine-tuning with advanced MoE strategies"""
    logging.info("\n" + "="*80)
    logging.info("PHASE 3: JOINT FINE-TUNING WITH ADVANCED MOE STRATEGIES")
    logging.info("="*80)
    logging.info(f"Advanced strategies enabled:")
    logging.info(f"  • Load Balancing Loss (coef: {args.balance_loss_coef})")
    logging.info(f"  • Expert Diversity Loss (coef: {args.diversity_loss_coef})")
    logging.info(f"  • Balanced Sampling: {args.use_balanced_sampling}")
    
    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Create data source based on sampling strategy
    if args.use_balanced_sampling:
        # Use balanced sampling generator
        balanced_data_gen = create_balanced_joint_loader(train_loaders, max_batches_per_epoch=100)
        logging.info("Using balanced sampling for fair task representation")
    else:
        # Create mixed training data (original approach)
        all_joint_data = []
        for task_id, train_loader in enumerate(train_loaders):
            for inputs, labels in train_loader:
                all_joint_data.append((inputs, labels, task_id))
        logging.info("Using standard mixed sampling")
    
    # Joint optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate * 0.3, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.joint_epochs)
    
    joint_epochs = args.joint_epochs
    
    for epoch in range(joint_epochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_gate_loss = 0.0
        epoch_balance_loss = 0.0
        epoch_diversity_loss = 0.0
        correct_routing = 0
        correct_classification = 0
        total_samples = 0
        
        if args.use_balanced_sampling:
            # Use balanced sampling
            balanced_generator = balanced_data_gen()
            batches_data = list(balanced_generator)
            num_batches = len(batches_data)
        else:
            # Shuffle joint data (original approach)
            np.random.shuffle(all_joint_data)
            batch_size = args.batch_size
            num_batches = len(all_joint_data) // batch_size
            batches_data = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_joint_data))
                batch_data = all_joint_data[start_idx:end_idx]
                batches_data.append(batch_data)
        
        progress_bar = tqdm(enumerate(batches_data), desc=f"Joint Epoch {epoch+1}", total=num_batches)
        
        for batch_idx, batch_data in progress_bar:
            if len(batch_data) == 0:
                continue
            
            # Handle different batch formats
            if args.use_balanced_sampling and len(batch_data) == 3:
                # Balanced sampling format: (inputs, labels, task_ids)
                batch_inputs, batch_labels, batch_task_ids = batch_data
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_task_ids = batch_task_ids.to(device)
            else:
                # Original format: list of (inputs, labels, task_id) tuples
                batch_inputs = []
                batch_labels = []
                batch_task_ids = []
                
                for inputs, labels, task_id in batch_data:
                    batch_inputs.append(inputs)
                    batch_labels.append(labels)
                    batch_task_ids.extend([task_id] * len(labels))  # One task_id per sample
                
                batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
                batch_labels = torch.cat(batch_labels, dim=0).to(device)
                batch_task_ids = torch.tensor(batch_task_ids, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass with expert outputs for diversity loss
            outputs, gate_logits, expert_outputs = model(batch_inputs, task_id=None, return_expert_outputs=True)
            
            # Create target tensor for multi-task output
            # This is complex because each expert outputs for its own small set of classes
            targets = torch.zeros_like(outputs)
            
            # Generate task classes dynamically based on model configuration
            # For TinyImageNet: Data provides LOCAL labels (0-19) for each task
            # No need for complex verification - labels should be 0 to classes_per_task-1
            for i, (label, task_id) in enumerate(zip(batch_labels, batch_task_ids)):
                global_label = label.item()
                task_id_val = task_id.item()

                # Remap global CIFAR-10 label (0-9) to the 0-(classes_per_task-1) range
                local_label = global_label % model.classes_per_task

                # Compute the correct position within the big output vector
                output_idx = task_id_val * model.classes_per_task + local_label
                targets[i, output_idx] = 1.0
            
            # 1. Classification loss
            class_loss = F.cross_entropy(outputs, targets.argmax(dim=1))
            
            # 2. Gating loss (moderate weight in joint phase)
            gate_loss = F.cross_entropy(gate_logits, batch_task_ids)
            
            # 3. Advanced MoE losses
            # Load balancing loss
            balance_loss = calculate_load_balancing_loss(gate_logits, args.num_experts)
            
            # Expert diversity loss
            diversity_loss = calculate_diversity_loss(expert_outputs)
            
            # Total loss with all components
            total_loss = (class_loss + 
                         args.gating_loss_coef * gate_loss +
                         args.balance_loss_coef * balance_loss +
                         args.diversity_loss_coef * diversity_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_class_loss += class_loss.item()
            epoch_gate_loss += gate_loss.item()
            epoch_balance_loss += balance_loss.item()
            epoch_diversity_loss += diversity_loss.item()
            
            _, predicted = torch.max(outputs, 1)
            _, predicted_experts = torch.max(gate_logits, 1)
            
            correct_classification += (predicted == targets.argmax(dim=1)).sum().item()
            correct_routing += (predicted_experts == batch_task_ids).sum().item()
            total_samples += batch_inputs.size(0)
            
            # Update progress bar
            class_acc = (correct_classification / total_samples) * 100
            routing_acc = (correct_routing / total_samples) * 100
            progress_bar.set_postfix({
                'class': f"{class_acc:.1f}%",
                'route': f"{routing_acc:.1f}%",
                'balance': f"{balance_loss.item():.3f}",
                'diversity': f"{diversity_loss.item():.3f}"
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_class_loss = epoch_class_loss / num_batches if num_batches > 0 else 0
        avg_gate_loss = epoch_gate_loss / num_batches if num_batches > 0 else 0
        avg_balance_loss = epoch_balance_loss / num_batches if num_batches > 0 else 0
        avg_diversity_loss = epoch_diversity_loss / num_batches if num_batches > 0 else 0
        final_class_acc = (correct_classification / total_samples) * 100 if total_samples > 0 else 0
        final_routing_acc = (correct_routing / total_samples) * 100 if total_samples > 0 else 0
        
        logging.info(f"Joint Epoch {epoch+1}: Loss={avg_loss:.4f} "
                    f"(Class={avg_class_loss:.4f}, Gate={avg_gate_loss:.4f}, "
                    f"Balance={avg_balance_loss:.4f}, Diversity={avg_diversity_loss:.4f}) "
                    f"ClassAcc={final_class_acc:.1f}%, RouteAcc={final_routing_acc:.1f}%")
    
    logging.info(f"\nPhase 3 Complete - Joint fine-tuning finished")

def evaluate_final_performance(model, test_loaders, task_classes, device):
    """Final evaluation with Task-IL and Class-IL metrics"""
    logging.info("\n" + "="*80)
    logging.info("FINAL PERFORMANCE EVALUATION")
    logging.info("="*80)
    
    model.eval()
    
    # Task-IL evaluation
    task_il_correct = 0
    task_il_total = 0
    expert_accuracies = []
    
    with torch.no_grad():
        for expert_id, test_loader in enumerate(test_loaders):
            expert_correct = 0
            expert_total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 🔧 FIX: Convert global labels to local labels properly
                task_class_list = task_classes[expert_id]
                global_to_local_map = {global_class: local_idx for local_idx, global_class in enumerate(task_class_list)}
                
                local_labels = torch.zeros_like(labels)
                for i, global_label in enumerate(labels):
                    if global_label.item() in global_to_local_map:
                        local_labels[i] = global_to_local_map[global_label.item()]
                    else:
                        # Fallback for unexpected labels
                        local_labels[i] = global_label % model.classes_per_task
                
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
            
            logging.info(f"Expert {expert_id} (classes {task_classes[expert_id]}): {expert_acc:.2f}%")
    
    task_il_accuracy = (task_il_correct / task_il_total) * 100
    
    # Class-IL evaluation
    class_il_correct = 0
    class_il_total = 0
    
    # Create combined test loader
    all_test_data = []
    for test_loader in test_loaders:
        for inputs, labels in test_loader:
            all_test_data.append((inputs, labels))
    
    # Shuffle and evaluate
    np.random.shuffle(all_test_data)
    
    with torch.no_grad():
        for inputs, labels in all_test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Class-IL: no task information
            outputs = model.forward_all_tasks(inputs)
            _, predicted = torch.max(outputs, 1)
            
            class_il_correct += (predicted == labels).sum().item()
            class_il_total += labels.size(0)
    
    class_il_accuracy = (class_il_correct / class_il_total) * 100
    
    logging.info(f"\nFINAL RESULTS:")
    logging.info(f"Expert accuracies: {expert_accuracies}")
    logging.info(f"Task-IL Accuracy: {task_il_accuracy:.2f}%")
    logging.info(f"Class-IL Accuracy: {class_il_accuracy:.2f}%")
    logging.info(f"Task-IL vs Class-IL gap: {task_il_accuracy - class_il_accuracy:.2f}%")
    
    return {
        'expert_accuracies': expert_accuracies,
        'task_il_accuracy': task_il_accuracy,
        'class_il_accuracy': class_il_accuracy,
        'task_class_gap': task_il_accuracy - class_il_accuracy
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='3-Phase Hippocampal MoE Training')
    
    # Model configuration
    parser.add_argument('--num_experts', type=int, default=5)
    parser.add_argument('--classes_per_task', type=int, default=2)
    
    # Training configuration
    parser.add_argument('--expert_epochs', type=int, default=12)
    parser.add_argument('--gate_epochs', type=int, default=20)
    parser.add_argument('--joint_epochs', type=int, default=8)
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gate_lr', type=float, default=5e-3)
    parser.add_argument('--gating_loss_coef', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Advanced MoE strategy hyperparameters
    parser.add_argument('--balance_loss_coef', type=float, default=0.01,
                        help='Coefficient for load balancing loss')
    parser.add_argument('--diversity_loss_coef', type=float, default=0.1,
                        help='Coefficient for expert diversity loss')
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True,
                        help='Use balanced sampling in Phase 3 joint training')
    
    return parser.parse_args()

def main():
    """Main function with 3-phase hippocampal training"""
    args = parse_arguments()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = setup_logging()
    
    logging.info("3-PHASE OPTIMAL HIPPOCAMPAL MoE TRAINING")
    logging.info("=" * 80)
    logging.info("Enhanced with Advanced MoE Strategies:")
    logging.info("  • Load Balancing Loss (prevents expert collapse)")
    logging.info("  • Expert Diversity Loss (encourages specialization)")
    logging.info("  • Balanced Sampling (fair task representation)")
    logging.info("MEC→CA1 Direct Bypass (biological perforant path)")
    logging.info("Soft Gating During Training (prevents dead experts)")
    logging.info("DG Mossy Fiber Expansion (4x better pattern separation)")
    logging.info("3-Phase Training (Expert → Gating → Joint)")
    logging.info(f"Device: {device}")
    logging.info(f"Training phases:")
    logging.info(f"  1. Expert Training: {args.expert_epochs} epochs per expert")
    logging.info(f"  2. Gate Training: {args.gate_epochs} epochs")
    logging.info(f"  3. Joint Fine-tuning: {args.joint_epochs} epochs")
    
    # Create task-specific dataloaders
    train_loaders, test_loaders, task_classes = create_task_specific_dataloaders(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        batch_size=args.batch_size
    )
    
    # Create Optimal Hippocampal MoE
    model = OptimalHippocampalMoE(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        input_channels=3,
        target_sparsity=args.target_sparsity # Pass target_sparsity to the model
    ).to(device)
    
    model.set_task_classes(task_classes)
    
    # Count FLOPs
    logging.info("\nFLOP Analysis:")
    flops, params = count_model_flops(model, device=device)
    if flops:
        logging.info(f"Enhanced Hippocampal MoE: {flops/1e6:.2f}M FLOPs, {params/1e6:.2f}M params")
    
    # Phase 1: Train experts independently
    expert_results = phase1_train_experts_independently(model, train_loaders, test_loaders, device, args)
    
    # Phase 2: Train gating network separately
    routing_acc = phase2_train_gating_network(model, train_loaders, test_loaders, device, args)
    
    # Phase 3: Joint fine-tuning
    phase3_joint_fine_tuning(model, train_loaders, test_loaders, device, args)
    
    # Final evaluation
    final_results = evaluate_final_performance(model, test_loaders, task_classes, device)
    
    # Save results
    import json
    results = {
        **final_results,
        'expert_training_results': expert_results,
        'final_routing_accuracy': routing_acc,
        'args': vars(args),
        'flops': flops/1e6 if flops else None,
        'params': params/1e6 if params else None
    }
    
    with open(f'{log_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\n💾 Results saved to {log_dir}/")
    logging.info(f"🎉 3-Phase Optimal Hippocampal MoE Training Complete!")

if __name__ == "__main__":
    main() 