#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
import json
import logging
import argparse
import time
from datetime import datetime

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# HIPPOCAMPAL COMPONENTS (from cifar10_hippocampal_lenet.py)
# ============================================================================

class GridCellLayer(nn.Module):
    """Grid cell-like spatial encoding from entorhinal cortex"""
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

class SparseActivation(nn.Module):
    """Dentate Gyrus sparse activation for pattern separation"""
    def __init__(self, percent_on=0.1):
        super().__init__()
        self.percent_on = percent_on
        
    def forward(self, x):
        # Get top k activations (sparse coding like DG)
        k = int(x.size(1) * self.percent_on)
        k = max(1, k)
        topk, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x).to(x.device)
        mask.scatter_(1, indices, 1)
        return x * mask

class SoftGating(nn.Module):
    """Soft gating during training, hard during inference"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, gate_logits, training=True):
        if training:
            # Soft gating during training - all experts contribute
            return F.softmax(gate_logits / self.temperature, dim=1)
        else:
            # Hard gating during inference - winner takes all
            hard_gates = torch.zeros_like(gate_logits)
            _, max_indices = torch.max(gate_logits, 1)
            hard_gates.scatter_(1, max_indices.unsqueeze(1), 1.0)
            return hard_gates

# ============================================================================
# MOE GATING NETWORK
# ============================================================================

class HippocampalGatingNetwork(nn.Module):
    """Gating network for routing to different DG-CA3 experts"""
    def __init__(self, input_dim, num_experts=5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_experts)
        )
        
    def forward(self, x):
        return self.gate(x)

# ============================================================================
# HIPPOCAMPAL EXPERTS (DG + CA3)
# ============================================================================

class DentateGyrusExpert(nn.Module):
    """Dentate Gyrus - Pattern Separation Expert with Mossy Fiber Expansion"""
    def __init__(self, input_dim, hidden_dim, sparsity=0.1, expansion_factor=4):
        super().__init__()
        # DG expansion for better pattern separation (biological mossy fibers)
        expanded_dim = input_dim * expansion_factor
        self.pattern_separation = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.ReLU(),
            nn.LayerNorm(expanded_dim),
            SparseActivation(percent_on=sparsity),
            nn.Linear(expanded_dim, hidden_dim),  # Project back to hidden_dim
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.out_features = hidden_dim
        
    def forward(self, x):
        return self.pattern_separation(x)

class CA3PatternCompletion(nn.Module):
    """CA3 - Pattern Completion Expert"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encode = nn.Linear(input_dim, hidden_dim)
        self.decode = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # Auto-associative pattern completion
        encoded = F.relu(self.encode(x))
        completed = self.norm(self.decode(encoded))
        return completed

class HippocampalExpert(nn.Module):
    """
    An enhanced Hippocampal Expert that now includes its own CA1 integration.
    This makes each expert a self-contained processing stream from DG to CA1.
    """
    def __init__(self, input_dim, dg_dim, ca3_dim, feature_dim, sparsity=0.1):
        super().__init__()
        self.dg = DentateGyrusExpert(input_dim, dg_dim, sparsity)
        self.ca3 = CA3PatternCompletion(dg_dim, ca3_dim)
        
        # Each expert now has its own CA1 integration layer.
        self.ca1_integration = nn.Sequential(
            nn.Linear(dg_dim + dg_dim + feature_dim, 256),  # DG(512) + CA3(512) + Features(1600) = 2624
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
# COMPLETE HIPPOCAMPAL MOE MODEL
# ============================================================================

class HippocampalMoE(nn.Module):
    """
    Complete Hippocampal MoE:
    Input → Feature Extractor + Grid Cells → Gating → DG → CA3 → Close Gating → CA1 → Output
    """
    def __init__(self, num_experts=5, classes_per_task=2, input_channels=3):
        super().__init__()
        self.num_experts = num_experts
        self.classes_per_task = classes_per_task
        self.num_classes = num_experts * classes_per_task
        
        # Feature Extractor (Entorhinal Cortex) + Grid Cells
        self.feature_extractor = nn.Sequential(
            # LeNet-style feature extraction
            nn.Conv2d(input_channels, 32, 5),  # 32x32 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            GridCellLayer(32),  # Add grid cell processing
            
            nn.Conv2d(32, 64, 5),  # 14x14 -> 10x10
            nn.ReLU(), 
            nn.MaxPool2d(2),  # 10x10 -> 5x5
        )
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_dim = dummy_output.numel()
        
        print(f"🧠 Hippocampal MoE: Feature dimension = {self.feature_dim}")
        
        # Gating Network (decides which DG-CA3 expert to use)
        self.gating_network = HippocampalGatingNetwork(self.feature_dim, num_experts)
        self.soft_gating = SoftGating(temperature=1.0)
        
        # Hippocampal Experts (DG + CA3)
        dg_dim = 512  # DG sparse representation
        ca3_dim = 256  # CA3 associative memory
        self.hippocampal_experts = nn.ModuleList([
            HippocampalExpert(self.feature_dim, dg_dim, ca3_dim, self.feature_dim, sparsity=0.05)
            for _ in range(num_experts)
        ])
        
        # Output layers per task
        self.output_layers = nn.ModuleList([
            nn.Linear(128, classes_per_task) for _ in range(num_experts)
        ])
        
        self.current_task = 0
        self.task_classes = None
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_task_classes(self, task_classes):
        """Set the task-to-CIFAR class mapping"""
        self.task_classes = task_classes
        
        # Create CIFAR class to output index mapping
        self.cifar_to_output = {}
        for task_id, cifar_classes in enumerate(task_classes):
            for local_idx, cifar_class in enumerate(cifar_classes):
                output_idx = task_id * self.classes_per_task + local_idx
                self.cifar_to_output[cifar_class] = output_idx
    
    def forward(self, x, task_id=None):
        # Feature Extraction + Grid Cells (Entorhinal Cortex)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        direct_features = features # For MEC->CA1 bypass
        
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
        else:
            # Soft or hard routing based on training mode
            if self.training:
                # Soft routing during training - weighted combination
                for expert_id in range(self.num_experts):
                    weight = gate_weights[:, expert_id].unsqueeze(1)
                    
                    # Get integrated output from each expert's internal CA1
                    _, ca1_output = self.hippocampal_experts[expert_id](features)
                    
                    # Output
                    expert_outputs = self.output_layers[expert_id](ca1_output)
                    
                    start_idx = expert_id * self.classes_per_task
                    end_idx = (expert_id + 1) * self.classes_per_task
                    final_outputs[:, start_idx:end_idx] += weight * expert_outputs
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
        
        return final_outputs, gate_logits, None

    def forward_all_tasks(self, x):
        """Class-IL evaluation with proper CIFAR class mapping"""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        gate_logits = self.gating_network(features)
        gate_probs = F.softmax(gate_logits / 2.0, dim=1)
        
        batch_size = x.size(0)
        
        if self.task_classes is None:
            # Fallback to task-based output
            conf_outputs = torch.zeros(batch_size, self.num_classes, device=x.device)
            
            for expert_id in range(self.num_experts):
                _, ca1_output = self.hippocampal_experts[expert_id](features)
                expert_outputs = self.output_layers[expert_id](ca1_output)
                expert_probs = F.softmax(expert_outputs, dim=1)
                expert_confidence = expert_probs.max(dim=1)[0]
                
                gate_weight = gate_probs[:, expert_id]
                combined_weight = (gate_weight * expert_confidence).unsqueeze(1)
                
                start_idx = expert_id * self.classes_per_task
                end_idx = (expert_id + 1) * self.classes_per_task
                conf_outputs[:, start_idx:end_idx] += combined_weight * expert_probs
            
            return conf_outputs
        
        # Create outputs indexed by CIFAR class
        cifar_outputs = torch.zeros(batch_size, 10, device=x.device)
        
        for expert_id in range(self.num_experts):
            # Full hippocampal processing
            _, ca1_output = self.hippocampal_experts[expert_id](features)
            expert_outputs = self.output_layers[expert_id](ca1_output)
            expert_probs = F.softmax(expert_outputs, dim=1)
            expert_confidence = expert_probs.max(dim=1)[0]
            
            gate_weight = gate_probs[:, expert_id]
            combined_weight = (gate_weight * expert_confidence).unsqueeze(1)
            
            # Map to CIFAR class indices
            cifar_classes = self.task_classes[expert_id]
            for local_idx, cifar_class in enumerate(cifar_classes):
                expert_class_prob = expert_probs[:, local_idx]
                cifar_outputs[:, cifar_class] += combined_weight.squeeze() * expert_class_prob
        
        return cifar_outputs
    
    def set_task(self, task_id):
        self.current_task = task_id

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def count_model_flops(model, input_shape=(1, 3, 32, 32), device=None):
    """Count FLOPs with hippocampal components"""
    try:
        import thop
    except ImportError:
        print("⚠️ thop not available, installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'thop'])
        import thop
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🧠 FLOP Counting for {model.__class__.__name__}")
    print(f"   Input shape: {input_shape}")
    print(f"   Device: {device}")
    
    try:
        model = model.to(device)
        model.eval()
        
        test_input = torch.randn(input_shape).to(device)
        print(f"   Test input created: {test_input.shape}")
        
        # Test forward pass
        print("   Testing hippocampal forward pass...")
        with torch.no_grad():
            output = model.forward_all_tasks(test_input)
            print(f"   ✅ Forward pass successful! Output shape: {output.shape}")
        
        # Count FLOPs
        print("   Counting FLOPs...")
        flops, params = thop.profile(model, inputs=(test_input,), verbose=False)
        
        print(f"   ✅ FLOP counting successful!")
        print(f"   📊 Total FLOPs: {flops/1e6:.2f}M")
        print(f"   📊 Total Params: {params/1e6:.2f}M")
        
        return flops, params
        
    except Exception as e:
        print(f"   ❌ FLOP counting failed: {e}")
        return None, None

def create_cifar10_dataloaders():
    """Create CIFAR-10 dataloaders"""
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                           shuffle=False, num_workers=2)

    return trainloader, testloader

def train_hippocampal_moe():
    """Train the Enhanced Hippocampal MoE with biological improvements"""
    print("🧠 Enhanced Hippocampal MoE Training")
    print("=====================================")
    print("✅ MEC→CA1 Direct Bypass (biological perforant path)")
    print("✅ Soft Gating During Training (prevents dead experts)")
    print("✅ DG Mossy Fiber Expansion (4x better pattern separation)")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataloaders
    trainloader, testloader = create_cifar10_dataloaders()
    
    # Create model
    model = HippocampalMoE(num_experts=5, classes_per_task=2, input_channels=3)
    model = model.to(device)
    
    # Define task classes for CIFAR-10
    task_classes = [
        [0, 1],  # Task 0: airplane, automobile
        [2, 3],  # Task 1: bird, cat
        [4, 5],  # Task 2: deer, dog
        [6, 7],  # Task 3: frog, horse
        [8, 9],  # Task 4: ship, truck
    ]
    model.set_task_classes(task_classes)
    
    # Count FLOPs
    print("\n🔬 FLOP Analysis:")
    flops, params = count_model_flops(model, device=device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    num_epochs = 10  # Full biological circuit test
    
    # Track metrics
    task_il_accuracies = []
    class_il_accuracies = []
    
    print(f"\n🚀 Starting {num_epochs}-epoch training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Determine task for each sample
            task_ids = []
            for label in labels:
                for task_id, classes in enumerate(task_classes):
                    if label.item() in classes:
                        task_ids.append(task_id)
                        break
            task_ids = torch.tensor(task_ids, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass with task-specific routing
            outputs, gate_logits, _ = model(inputs, task_id=task_ids)
            
            # Create target tensor for multi-task output
            targets = torch.zeros_like(outputs)
            for j, (label, task_id) in enumerate(zip(labels, task_ids)):
                local_class = task_classes[task_id].index(label.item())
                output_idx = task_id * model.classes_per_task + local_class
                targets[j, output_idx] = 1.0
            
            # Loss calculation
            loss = F.cross_entropy(outputs, targets.argmax(dim=1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == targets.argmax(dim=1)).sum().item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                accuracy = 100 * correct_predictions / total_predictions
                print(f'[Epoch {epoch+1}, Batch {i+1:5d}] Loss: {avg_loss:.3f}, Acc: {accuracy:.1f}%')
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        scheduler.step()
        
        # Evaluation
        print(f"\n📊 Epoch {epoch+1} Evaluation:")
        
        # Task-IL evaluation
        model.eval()
        task_il_correct = 0
        task_il_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Oracle task information for Task-IL
                for task_id, classes in enumerate(task_classes):
                    task_mask = torch.tensor([label.item() in classes for label in labels], device=device)
                    if task_mask.sum() > 0:
                        task_inputs = inputs[task_mask]
                        task_labels = labels[task_mask]
                        
                        outputs, _ = model(task_inputs, task_id=task_id)
                        
                        # Extract predictions for this task
                        start_idx = task_id * model.classes_per_task
                        end_idx = start_idx + model.classes_per_task
                        task_outputs = outputs[:, start_idx:end_idx]
                        
                        # Convert CIFAR labels to local task labels
                        local_labels = torch.tensor([classes.index(label.item()) for label in task_labels], device=device)
                        
                        _, predicted = torch.max(task_outputs, 1)
                        task_il_correct += (predicted == local_labels).sum().item()
                        task_il_total += task_labels.size(0)
        
        task_il_accuracy = 100 * task_il_correct / task_il_total if task_il_total > 0 else 0
        task_il_accuracies.append(task_il_accuracy)
        
        # Class-IL evaluation
        class_il_correct = 0
        class_il_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Class-IL: no task information
                outputs = model.forward_all_tasks(inputs)
                _, predicted = torch.max(outputs, 1)
                
                class_il_correct += (predicted == labels).sum().item()
                class_il_total += labels.size(0)
        
        class_il_accuracy = 100 * class_il_correct / class_il_total
        class_il_accuracies.append(class_il_accuracy)
        
        print(f"Task-IL Accuracy: {task_il_accuracy:.2f}%")
        print(f"Class-IL Accuracy: {class_il_accuracy:.2f}%")
    
    # Final results
    print(f"\n🎯 Final Results After {num_epochs} Epoch(s):")
    print(f"Task-IL Accuracy: {task_il_accuracies[-1]:.2f}%")
    print(f"Class-IL Accuracy: {class_il_accuracies[-1]:.2f}%")
    
    if flops is not None:
        print(f"Training Cost: {flops/1e12:.2f} TFLOPs")
    
    print("\n🧠 Enhanced Hippocampal Architecture:")
    print("• Grid Cells: Spatial frequency encoding")
    print("• DG: 4x expansion + 10% sparse activation")
    print("• CA3: Auto-associative pattern completion")
    print("• MEC→CA1 Bypass: Direct perforant path")
    print("• Soft Gating: Prevents expert death during training")
    
    return model, task_il_accuracies, class_il_accuracies

if __name__ == "__main__":
    set_seed(42)
    model, task_il, class_il = train_hippocampal_moe() 