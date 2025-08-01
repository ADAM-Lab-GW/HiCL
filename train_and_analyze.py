#!/usr/bin/env python3
"""
Train Hippocampal MoE + Analyze TRAINED Model
This script fixes the issue where we were analyzing random weights instead of trained weights!
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

# Import from the optimal training script
from train_hippocampal_optimal_moe import *

def analyze_TRAINED_model(model, test_loaders, task_classes, device, save_dir):
    """
    Analyze the TRAINED model (not random weights!)
    """
    logging.info("\n" + "🔬" * 60)
    logging.info("🔬 ANALYZING THE TRAINED MODEL (ACTUAL LEARNED WEIGHTS)")
    logging.info("🔬" * 60)
    
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
                # MODIFIED: Get ca1_output directly from the expert
                dg_output, ca1_output = model.hippocampal_experts[task_id](features_flat)
                
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
    
    logging.info(f"📊 Analyzed {len(all_task_labels)} samples from TRAINED model")
    
    # Create visualizations
    create_TRAINED_visualizations(
        all_gate_logits.numpy(), all_dg_outputs.numpy(), all_ca1_outputs.numpy(),
        all_task_labels, routing_matrix, expert_utilization, analysis_dir
    )
    
    return {
        'routing_matrix': routing_matrix,
        'expert_utilization': expert_utilization,
        'dg_sparsity': (all_dg_outputs > 0).float().mean().item(),
        'routing_accuracy': np.diag(routing_matrix).mean()
    }

def create_TRAINED_visualizations(gate_logits, dg_outputs, ca1_outputs, task_labels, 
                                 routing_matrix, expert_utilization, save_dir):
    """Create visualizations from TRAINED model data"""
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('🧠 TRAINED Hippocampal MoE Analysis (Actual Learned Weights)', 
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
    
    # 12. Key Insights Text
    axes[2,3].text(0.1, 0.9, f'📊 TRAINED Model Analysis', fontsize=14, fontweight='bold', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.8, f'Routing Accuracy: {routing_accuracies.mean():.1%}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.7, f'DG Sparsity: {mean_sparsity:.1%}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.6, f'Load Balance: {1.0/(1.0+np.var(expert_utilization)*10):.3f}', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.5, f'Expert Diversity: {1.0-expert_similarities.mean():.3f}', transform=axes[2,3].transAxes)
    
    # Status indicators
    status_color = 'green' if routing_accuracies.mean() > 0.6 else 'red'
    axes[2,3].text(0.1, 0.3, '✅ GOOD PERFORMANCE' if routing_accuracies.mean() > 0.6 else '❌ POOR PERFORMANCE', 
                  color=status_color, fontweight='bold', transform=axes[2,3].transAxes)
    
    load_balance_status = 'green' if np.var(expert_utilization) < 0.01 else 'red'
    axes[2,3].text(0.1, 0.2, '✅ BALANCED EXPERTS' if np.var(expert_utilization) < 0.01 else '❌ IMBALANCED EXPERTS', 
                  color=load_balance_status, fontweight='bold', transform=axes[2,3].transAxes)
    
    axes[2,3].set_xlim(0, 1)
    axes[2,3].set_ylim(0, 1)
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'TRAINED_model_analysis.png'), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📊 TRAINED model analysis saved!")

def main():
    """Train model + Analyze TRAINED model"""
    args = parse_arguments()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = setup_logging()
    
    logging.info("🚀 TRAIN + ANALYZE TRAINED MODEL (NOT RANDOM WEIGHTS!)")
    logging.info("=" * 80)
    
    # Create dataloaders
    train_loaders, test_loaders, task_classes = create_task_specific_dataloaders(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        batch_size=args.batch_size
    )
    
    # Create model
    model = OptimalHippocampalMoE(
        num_experts=args.num_experts,
        classes_per_task=args.classes_per_task,
        input_channels=3
    ).to(device)
    
    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {trainable_params:,}")
    
    model.set_task_classes(task_classes)
    
    # === TRAINING (same as before) ===
    expert_results = phase1_train_experts_independently(model, train_loaders, test_loaders, device, args)
    routing_acc = phase2_train_gating_network(model, train_loaders, test_loaders, device, args)  
    phase3_joint_fine_tuning(model, train_loaders, test_loaders, device, args)
    final_results = evaluate_final_performance(model, test_loaders, task_classes, device)
    
    # === ANALYZE TRAINED MODEL ===
    logging.info("\n" + "🔬" * 80)
    logging.info("🔬 ANALYZING TRAINED MODEL (LEARNED WEIGHTS, NOT RANDOM!)")
    logging.info("🔬" * 80)
    
    trained_analysis = analyze_TRAINED_model(model, test_loaders, task_classes, device, log_dir)
    
    # Generate report
    report_text = f"""
# 🧠 TRAINED Hippocampal MoE Analysis Report

## Key Findings (From TRAINED Model):
- **Routing Accuracy**: {trained_analysis['routing_accuracy']:.1%}
- **DG Sparsity**: {trained_analysis['dg_sparsity']:.1%} (Target: 5-10%)
- **Expert Balance**: {1.0/max(trained_analysis['expert_utilization'])/min(trained_analysis['expert_utilization']):.1f}x imbalance

## Status:
{'✅ EXCELLENT ROUTING' if trained_analysis['routing_accuracy'] > 0.8 else '⚠️ POOR ROUTING' if trained_analysis['routing_accuracy'] < 0.5 else '✅ GOOD ROUTING'}
{'✅ BIOLOGICAL SPARSITY' if trained_analysis['dg_sparsity'] < 0.15 else '⚠️ HIGH SPARSITY'}
{'✅ BALANCED EXPERTS' if max(trained_analysis['expert_utilization'])/min(trained_analysis['expert_utilization']) < 3 else '⚠️ IMBALANCED EXPERTS'}

This analysis uses ACTUAL TRAINED WEIGHTS, not random initialization!
    """
    
    with open(f'{log_dir}/TRAINED_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logging.info(f"\n🎉 TRAINING + TRAINED MODEL ANALYSIS COMPLETE!")
    logging.info(f"💾 Results: {log_dir}/")
    logging.info(f"📊 TRAINED analysis: {log_dir}/trained_analysis/")
    logging.info(f"📋 Report: {log_dir}/TRAINED_analysis_report.md")

if __name__ == "__main__":
    main() 