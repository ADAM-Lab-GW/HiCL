#!/usr/bin/env python3
"""
TinyImageNet Hippocampal MoE Training V7 – DG-Gated Model
=========================================================

This version uses the DG-Gated Hippocampal MoE model from n.py
with the V7 training structure and analysis.

Key features:
1.  **DG-Gated Routing:** Uses Dentate Gyrus pattern similarity for expert selection
2.  **Standard Feature Extractor:** Uses regular LeNet-style convolutions
3.  **Enhanced Hippocampal Experts:** Custom DG and CA3 components
4.  **V7 Analysis:** Comprehensive gating deep dive analysis
"""
import torch.nn.functional as F
import os, sys, logging
import torch
import torch.nn as nn
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
import matplotlib.pyplot as plt

# Import the DG-Gated model and components from l.py (working version)
from l import (
    DGGatedHippocampalMoE,
    GridCellLayer,
    StandardFeatureExtractor,
    CustomEnhancedHippocampalExpert,
    CustomDentateGyrusExpert,
    CustomCA3PatternCompletion,
    analyze_dg_gated_model,
    create_dg_gated_visualizations,
    create_dg_deep_dive_visualizations,
    analyze_dg_deep_dive,
    phase1_train_experts_sequentially,
    phase2_contrastive_tuning,
    evaluate_dg_gated_model_standardized,
    calculate_global_contrastive_loss,
    create_balanced_loader,
    create_replay_balanced_loader,
    calculate_class_balanced_weights,
    analyze_dg_pattern_separation,
    dg_separation_diagnostic,
    diagnose_prototype_status,
    plot_prototype_stats,
    plot_gating_confidence,
    plot_cluster_purity,
    plot_per_class_accuracy,
    log_and_plot_prototype_drift,   
    calculate_prototype_regularization_loss,
    calculate_routing_confidence_penalty,
    calculate_expert_balancing_loss,
    calculate_sparsity_loss,
    calculate_distillation_loss,
    calculate_feature_distillation_loss,
    analyze_class_il_breakdown,
    analyze_task_il_breakdown,
    log_expert_utilization,
    log_active_loss_fraction,
    plot_contrastive_similarity_histogram,
    plot_prototype_distance_matrix,
    plot_tsne_dg,
    calculate_contrastive_diagnostics,
    calculate_expert_utilization_stats,
    calculate_dg_sparsity_stats,
    calculate_gating_confidence_stats,
    calculate_prototype_similarity_stats
)

# Import TinyImageNet utilities
from train_tinyimagenet_v4_deep_sparse import (
    create_tinyimagenet_tasks,
    setup_logging,
    analyze_trained_tinyimagenet_model
)

# --- V7 DEBUG IMPORTS ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Evaluation utility
from evaluation import evaluate_on_all_tasks

# === Set DG/prototype dimension at the top of the file ===
dg_dim = 512  # <--- Set this to your desired DG dimension (e.g., 512, 1024, 2048)

def debug_routing_issues(model, test_loaders, device):
    """
    Comprehensive debugging function to identify routing issues.
    Returns detailed diagnostics for fixing the routing problems.
    """
    logging.info("\n" + "🔍" * 60)
    logging.info("🔍 ROUTING ISSUE DIAGNOSTICS")
    logging.info("🔍" * 60)
    
    model.eval()
    
    # 1. Check prototype distinctness calculation
    if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None:
        
        
        prototypes = model.dg_prototypes.detach().cpu().numpy()
        proto_norms = np.linalg.norm(prototypes, axis=1)
        logging.info(f"Prototype norms: min={proto_norms.min():.4f}, max={proto_norms.max():.4f}, mean={proto_norms.mean():.4f}")
        
        # Normalize prototypes properly
        prototypes_norm = prototypes / proto_norms[:, np.newaxis]
        proto_sim = np.dot(prototypes_norm, prototypes_norm.T)
        
        # Get upper triangular (excluding diagonal)
        triu_indices = np.triu_indices(proto_sim.shape[0], k=1)
        off_diag_sims = proto_sim[triu_indices]
        
        distinctness = 1.0 - np.mean(off_diag_sims)
        logging.info(f"Off-diagonal similarities: min={off_diag_sims.min():.4f}, max={off_diag_sims.max():.4f}, mean={off_diag_sims.mean():.4f}")
        logging.info(f"Corrected Prototype Distinctness: {distinctness:.3f}")
        
        if distinctness > 1.0:
            logging.warning("⚠️ Distinctness > 1.0 indicates normalization issues!")
    
    # 2. Measure similarity gaps
    all_similarities = []
    all_correct_sims = []
    all_second_best_sims = []
    all_gaps = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                
                # Get DG outputs from all experts
                all_dg_outputs = []
                for i in range(model.num_experts):
                    dg_output, _ = model.hippocampal_experts[i](features)
                    all_dg_outputs.append(dg_output)
                
                all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [batch, experts, dg_dim]
                all_dg_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                
                # Calculate similarities
                sims = torch.einsum('bne,ne->bn', all_dg_norm, proto_norm)
                
                # Get correct and second-best similarities
                correct_sims = sims[torch.arange(sims.size(0)), task_id]
                masked_sims = sims.clone()
                masked_sims[torch.arange(sims.size(0)), task_id] = -1e9
                second_best_sims = masked_sims.max(dim=1).values
                
                gaps = correct_sims - second_best_sims
                
                all_similarities.append(sims.cpu())
                all_correct_sims.append(correct_sims.cpu())
                all_second_best_sims.append(second_best_sims.cpu())
                all_gaps.append(gaps.cpu())
    
    # Aggregate results
    all_similarities = torch.cat(all_similarities, dim=0)
    all_correct_sims = torch.cat(all_correct_sims, dim=0)
    all_second_best_sims = torch.cat(all_second_best_sims, dim=0)
    all_gaps = torch.cat(all_gaps, dim=0)
    
    # Calculate metrics
    mean_correct_sim = all_correct_sims.mean().item()
    mean_second_best_sim = all_second_best_sims.mean().item()
    mean_gap = all_gaps.mean().item()
    gap_std = all_gaps.std().item()
    
    # Calculate routing accuracy if gap is positive
    positive_gap_mask = all_gaps > 0
    routing_acc_if_gap_pos = positive_gap_mask.float().mean().item()
    
    logging.info(f"Mean correct similarity: {mean_correct_sim:.4f}")
    logging.info(f"Mean second-best similarity: {mean_second_best_sim:.4f}")
    logging.info(f"Mean gap: {mean_gap:.4f} ± {gap_std:.4f}")
    logging.info(f"Routing accuracy if gap > 0: {routing_acc_if_gap_pos:.1%}")
    
    # 3. Check oracle evaluation
    oracle_accuracies = []
    actual_accuracies = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            oracle_correct = 0
            actual_correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Oracle evaluation (force correct expert)
                oracle_outputs, _, _ = model(inputs, task_id=task_id)
                start_idx = task_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                oracle_task_outputs = oracle_outputs[:, start_idx:end_idx]
                _, oracle_pred = torch.max(oracle_task_outputs, 1)
                oracle_correct += (oracle_pred == labels).sum().item()
                
                # Actual gating evaluation
                actual_outputs, gate_logits, _ = model(inputs, task_id=None)
                _, actual_pred = torch.max(actual_outputs, 1)
                actual_correct += (actual_pred == labels).sum().item()
                
                total += labels.size(0)
            
            oracle_acc = oracle_correct / total if total > 0 else 0.0
            actual_acc = actual_correct / total if total > 0 else 0.0
            
            oracle_accuracies.append(oracle_acc)
            actual_accuracies.append(actual_acc)
            
            logging.info(f"Task {task_id}: Oracle={oracle_acc:.1%}, Actual={actual_acc:.1%}")
    
    mean_oracle = np.mean(oracle_accuracies)
    mean_actual = np.mean(actual_accuracies)
    
    logging.info(f"Mean Oracle Accuracy: {mean_oracle:.1%}")
    logging.info(f"Mean Actual Accuracy: {mean_actual:.1%}")
    
    # 4. Check expert utilization
    expert_utilization = torch.zeros(model.num_experts)
    total_samples = 0
    
    with torch.no_grad():
        for test_loader in test_loaders:
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                
                all_dg_outputs = []
                for i in range(model.num_experts):
                    dg_output, _ = model.hippocampal_experts[i](features)
                    all_dg_outputs.append(dg_output)
                
                all_dg_outputs = torch.stack(all_dg_outputs, dim=1)
                all_dg_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                
                sims = torch.einsum('bne,ne->bn', all_dg_norm, proto_norm)
                gate_probs = F.softmax(sims / model.gating_temperature, dim=1)
                
                expert_utilization += gate_probs.sum(dim=0)
                total_samples += inputs.size(0)
    
    expert_utilization = expert_utilization / total_samples
    logging.info(f"Expert utilization: {expert_utilization.numpy()}")
    
    # 5. Summary and recommendations
    logging.info("\n" + "🔍" * 60)
    logging.info("🔍 DIAGNOSTIC SUMMARY")
    logging.info("🔍" * 60)
    
    issues_found = []
    
    if mean_gap < 0.05:
        issues_found.append("❌ Similarity gap too small (< 0.05)")
    
    if routing_acc_if_gap_pos < 0.5:
        issues_found.append("❌ Even with positive gaps, routing accuracy < 50%")
    
    if mean_oracle - mean_actual > 0.3:
        issues_found.append("❌ Large gap between oracle and actual accuracy")
    
    if expert_utilization.min() < 0.02:
        issues_found.append("❌ Dead experts (utilization < 2%)")
    
    if expert_utilization.max() > 0.3:
        issues_found.append("❌ Expert hogging (utilization > 30%)")
    
    if not issues_found:
        logging.info("✅ No major issues detected")
    else:
        for issue in issues_found:
            logging.warning(issue)
    
    return {
        'mean_gap': mean_gap,
        'routing_acc_if_gap_pos': routing_acc_if_gap_pos,
        'mean_oracle': mean_oracle,
        'mean_actual': mean_actual,
        'expert_utilization': expert_utilization.numpy(),
        'issues_found': issues_found
    }

def refresh_prototypes_no_ema(model, train_loaders, device):
    """
    Recompute prototypes from full data without EMA to fix staleness.
    """
    logging.info("🔄 Refreshing prototypes without EMA...")
    
    model.eval()
    with torch.no_grad():
        for task_id, train_loader in enumerate(tqdm(train_loaders, desc="Refreshing Prototypes")):
            all_dg_outputs = []
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[task_id](features)
                all_dg_outputs.append(dg_output.cpu())
            
            # Average all DG outputs for this task (no EMA)
            if all_dg_outputs:
                prototype = torch.cat(all_dg_outputs, dim=0).mean(dim=0)
                model.dg_prototypes[task_id] = prototype.to(device)
    
    model.prototypes_computed = True
    logging.info("✅ Prototypes refreshed without EMA")

def adjust_temperature_schedule(model, current_temp, target_temp=0.7, anneal_steps=1000):
    """
    Gradually increase temperature to prevent premature winner-take-all.
    """
    if current_temp < target_temp:
        new_temp = min(target_temp, current_temp + 0.1)
        model.set_gating_temperature(new_temp)
        logging.info(f"🌡️ Increased temperature to {new_temp:.3f}")
        return new_temp
    return current_temp

def calculate_enhanced_prototype_loss(dg_outputs_all_experts, task_ids, prototypes, margin=0.1):
    """
    Enhanced prototype loss with explicit centroid pull and margin.
    """
    if not isinstance(dg_outputs_all_experts, torch.Tensor):
        dg_outputs_all_experts = torch.stack(dg_outputs_all_experts, dim=1)

    # Normalize for cosine similarity
    dg_outputs_norm = F.normalize(dg_outputs_all_experts, p=2, dim=2)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)

    # Calculate similarities
    similarities = torch.einsum('bed,ed->be', dg_outputs_norm, prototypes_norm)

    # Get correct expert similarities
    correct_expert_sims = similarities.gather(1, task_ids.unsqueeze(1)).squeeze()

    # PULL Loss: encourage correct expert to be similar to its prototype
    pull_loss = (1 - correct_expert_sims).mean()

    # MARGIN Loss: ensure correct expert is better than others by margin
    mask = torch.ones_like(similarities)
    mask.scatter_(1, task_ids.unsqueeze(1), 0)
    
    # Find best incorrect similarity for each sample
    masked_similarities = similarities * mask
    best_incorrect_sims = masked_similarities.max(dim=1).values
    
    # Margin loss: max(0, best_incorrect - correct + margin)
    margin_loss = F.relu(best_incorrect_sims - correct_expert_sims + margin).mean()

    # Combine losses
    total_loss = pull_loss + 2.0 * margin_loss
    
    return total_loss

def fix_oracle_evaluation(model, test_loaders, device):
    """
    Fix oracle evaluation to ensure it truly uses the correct expert.
    """
    logging.info("🔧 Fixing oracle evaluation...")
    
    model.eval()
    oracle_accuracies = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Force oracle routing by directly calling the expert
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, ca1_output = model.hippocampal_experts[task_id](features)
                expert_output = model.output_layers[task_id](ca1_output)
                
                # Create full output tensor
                full_outputs = torch.zeros(inputs.size(0), model.num_classes, device=device)
                start_idx = task_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                full_outputs[:, start_idx:end_idx] = expert_output
                
                # Get predictions for this task's classes
                task_outputs = full_outputs[:, start_idx:end_idx]
                _, predicted = torch.max(task_outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            acc = correct / total if total > 0 else 0.0
            oracle_accuracies.append(acc)
            logging.info(f"Task {task_id} Oracle Accuracy: {acc:.1%}")
    
    mean_oracle = np.mean(oracle_accuracies)
    logging.info(f"Mean Oracle Accuracy: {mean_oracle:.1%}")
    
    return mean_oracle

def gap_probe(model, test_loaders, device):
    """
    Critical diagnostic: measure similarity gaps between correct and second-best prototypes.
    Returns the four key numbers needed for surgical fixes.
    """
    logging.info("🔍 Running gap probe...")
    
    model.eval()
    all_correct_sims = []
    all_best_other_sims = []
    all_gaps = []
    all_routing_correct = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                
                # Get DG outputs from all experts
                all_dg_outputs = []
                for i in range(model.num_experts):
                    dg_output, _ = model.hippocampal_experts[i](features)
                    all_dg_outputs.append(dg_output)
                
                all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [batch, experts, dg_dim]
                all_dg_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                
                # Calculate similarities
                similarities = torch.einsum('bne,ne->bn', all_dg_norm, proto_norm)
                
                # Get correct and second-best similarities
                correct_sims = similarities[torch.arange(similarities.size(0)), task_id]
                masked_sims = similarities.clone()
                masked_sims[torch.arange(similarities.size(0)), task_id] = -1e9
                second_best_sims = masked_sims.max(dim=1).values
                
                gaps = correct_sims - second_best_sims
                
                # Check routing accuracy (if gap > 0, routing should be correct)
                routing_correct = (gaps > 0).float()
                
                all_correct_sims.append(correct_sims.cpu())
                all_best_other_sims.append(second_best_sims.cpu())
                all_gaps.append(gaps.cpu())
                all_routing_correct.append(routing_correct.cpu())
    
    # Aggregate results
    all_correct_sims = torch.cat(all_correct_sims, dim=0)
    all_best_other_sims = torch.cat(all_best_other_sims, dim=0)
    all_gaps = torch.cat(all_gaps, dim=0)
    all_routing_correct = torch.cat(all_routing_correct, dim=0)
    
    # Calculate key metrics
    mean_corr = all_correct_sims.mean().item()
    mean_best_other = all_best_other_sims.mean().item()
    mean_gap = all_gaps.mean().item()
    routing_upper_bound = all_routing_correct.mean().item()
    
    logging.info(f"🔍 GAP PROBE RESULTS:")
    logging.info(f"  mean_corr: {mean_corr:.4f}")
    logging.info(f"  mean_best_other: {mean_best_other:.4f}")
    logging.info(f"  mean_gap: {mean_gap:.4f}")
    logging.info(f"  routing_upper_bound: {routing_upper_bound:.1%}")
    
    return {
        'mean_corr': mean_corr,
        'mean_best_other': mean_best_other,
        'mean_gap': mean_gap,
        'routing_upper_bound': routing_upper_bound
    }

def routing_gap_loss(similarities, task_ids, margin=0.10):
    """
    Routing gap loss: ensures correct expert is better than others by margin.
    """
    b = torch.arange(task_ids.size(0), device=task_ids.device)
    corr = similarities[b, task_ids]
    mask = similarities.clone()
    mask[b, task_ids] = -1e9
    best_other, _ = mask.max(1)
    return F.relu(margin - (corr - best_other)).mean()

def adaptive_decor_weight(proto_stats):
    """
    Adaptive decorrelation weight based on current prototype separation.
    """
    if 'mean_off_diag' in proto_stats:
        m = proto_stats['mean_off_diag']
    else:
        # Compute from prototypes
        prototypes = proto_stats['prototypes']
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
        mask = torch.ones_like(similarity_matrix)
        mask.fill_diagonal_(0)
        m = (similarity_matrix * mask).sum() / mask.sum()
    
    if m < 0.10:
        return 0.0
    elif m < 0.18:
        return 0.05
    elif m < 0.25:
        return 0.1
    else:
        return 0.15

def anneal_temp(model, epoch):
    """
    Temperature annealing schedule for Phase 2.
    """
    schedule = [(0, 0.7), (3, 0.5), (6, 0.3), (9, 0.2), (12, 0.1)]
    for start, T in reversed(schedule):
        if epoch >= start:
            model.set_gating_temperature(T)
            break

def refresh_prototypes_light(model, train_loaders, device, samples_per_task=200):
    """
    Light prototype refresh with limited samples per task.
    """
    logging.info("🔄 Refreshing prototypes (light version)...")
    
    model.eval()
    with torch.no_grad():
        for t, loader in enumerate(train_loaders):
            acc = 0
            feats_accum = []
            for inputs, _ in loader:
                inputs = inputs.to(device)
                feats = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg, _ = model.hippocampal_experts[t](feats)
                feats_accum.append(dg)
                acc += len(inputs)
                if acc >= samples_per_task:
                    break
            if feats_accum:
                model.dg_prototypes[t] = torch.cat(feats_accum)[:samples_per_task].mean(0)
    
    model.prototypes_computed = True
    logging.info("✅ Prototypes refreshed (light version)")

def verify_label_expert_mapping(model):
    """
    Sanity check: verify label→expert mapping is correct.
    """
    logging.info("🔍 Verifying label→expert mapping...")
    
    for e, cls in enumerate(model.task_classes):
        for c in cls:
            expected_expert = c // model.classes_per_task
            if expected_expert != e:
                logging.error(f"❌ Mismatch: class {c} expected expert {expected_expert} got {e}")
                return False
    
    logging.info("✅ Label→expert mapping is correct")
    return True

def phase2_contrastive_tuning_surgical(*args, **kwargs):
    raise NotImplementedError('Use phase2_contrastive_tuning from l.py instead.')

def calculate_decorrelation_loss_light(prototypes):
    """
    Light decorrelation loss to prevent over-orthogonalization.
    """
    if prototypes is None:
        return torch.tensor(0.0, device=prototypes.device if prototypes is not None else 'cpu')
    
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
    
    # Only penalize high similarities (not force complete orthogonality)
    mask = torch.ones_like(similarity_matrix)
    mask.fill_diagonal_(0)
    
    # Penalize similarities above 0.1 (much more lenient)
    decorr_loss = F.relu(similarity_matrix - 0.1) * mask
    return decorr_loss.mean()

def create_balanced_loader_enhanced(task_loaders, batches_per_epoch=200):
    """
    Enhanced balanced loader with uniform mixing and warmup.
    """
    logging.info("🔄 Creating enhanced balanced loader...")
    
    # Create uniform task distribution
    num_tasks = len(task_loaders)
    uniform_dist = torch.ones(num_tasks) / num_tasks
    
    # Mix with actual task distribution for warmup
    alpha = 0.5  # Mixing parameter
    task_counts = torch.tensor([len(loader) for loader in task_loaders])
    actual_dist = task_counts / task_counts.sum()
    mixed_dist = alpha * uniform_dist + (1 - alpha) * actual_dist
    
    # Create balanced batches
    balanced_batches = []
    for _ in range(batches_per_epoch):
        # Sample task with mixed distribution
        task_id = torch.multinomial(mixed_dist, 1).item()
        
        # Get batch from that task
        try:
            inputs, labels = next(task_loaders[task_id])
            task_ids = torch.full((inputs.size(0),), task_id, dtype=torch.long)
            balanced_batches.append((inputs, labels, task_ids))
        except StopIteration:
            # Reset iterator if needed
            task_loaders[task_id] = iter(task_loaders[task_id])
            inputs, labels = next(task_loaders[task_id])
            task_ids = torch.full((inputs.size(0),), task_id, dtype=torch.long)
            balanced_batches.append((inputs, labels, task_ids))
    
    return balanced_batches

def evaluate_final_performance_dg_gated_enhanced(model, test_loaders, task_classes, device):
    """
    Enhanced final evaluation with proper oracle routing and detailed analysis.
    """
    logging.info("\n" + "="*80)
    logging.info("ENHANCED FINAL DG-GATED PERFORMANCE EVALUATION")
    logging.info("="*80)
    
    # Run comprehensive diagnostics first
    diagnostics = debug_routing_issues(model, test_loaders, device)
    
    # Fix oracle evaluation
    mean_oracle = fix_oracle_evaluation(model, test_loaders, device)
    
    # Standard evaluation
    model.eval()
    
    # Task-IL evaluation (oracle provides task_id)
    task_il_correct = 0
    task_il_total = 0
    expert_accuracies = []
    
    with torch.no_grad():
        for expert_id, test_loader in enumerate(test_loaders):
            expert_correct = 0
            expert_total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use fixed oracle routing
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, ca1_output = model.hippocampal_experts[expert_id](features)
                expert_output = model.output_layers[expert_id](ca1_output)
                
                # Create full output tensor
                full_outputs = torch.zeros(inputs.size(0), model.num_classes, device=device)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                full_outputs[:, start_idx:end_idx] = expert_output
                
                # Get predictions for this task's classes
                task_outputs = full_outputs[:, start_idx:end_idx]
                _, predicted = torch.max(task_outputs, 1)
                
                expert_correct += (predicted == labels).sum().item()
                expert_total += labels.size(0)
            
            expert_acc = (expert_correct / expert_total) * 100 if expert_total > 0 else 0.0
            expert_accuracies.append(expert_acc)
            task_il_correct += expert_correct
            task_il_total += expert_total
            
            logging.info(f"Expert {expert_id}: {expert_acc:.2f}%")
    
    task_il_accuracy = (task_il_correct / task_il_total) * 100 if task_il_total > 0 else 0.0
    
    # Class-IL evaluation (DG-gated)
    class_il_correct = 0
    class_il_total = 0
    
    with torch.no_grad():
        for test_loader in test_loaders:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use DG-gating
                outputs, _, analysis_data = model(inputs, task_id=None)
                _, predicted = torch.max(outputs, 1)
                
                class_il_correct += (predicted == labels).sum().item()
                class_il_total += labels.size(0)
    
    class_il_accuracy = (class_il_correct / class_il_total) * 100 if class_il_total > 0 else 0.0
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("FINAL PERFORMANCE SUMMARY")
    logging.info("="*80)
    logging.info(f"Task-IL Accuracy: {task_il_accuracy:.2f}%")
    logging.info(f"Class-IL Accuracy: {class_il_accuracy:.2f}%")
    logging.info(f"Oracle Accuracy: {mean_oracle:.1%}")
    logging.info(f"Routing Gap: {diagnostics['mean_gap']:.4f}")
    logging.info(f"Routing Accuracy (if gap > 0): {diagnostics['routing_acc_if_gap_pos']:.1%}")
    
    return {
        'task_il_accuracy': task_il_accuracy,
        'class_il_accuracy': class_il_accuracy,
        'oracle_accuracy': mean_oracle,
        'routing_gap': diagnostics['mean_gap'],
        'routing_accuracy': diagnostics['routing_acc_if_gap_pos'],
        'expert_utilization': diagnostics['expert_utilization'],
        'diagnostics': diagnostics
    }

def calculate_feature_distillation_loss(current_dg_output, previous_dg_outputs, current_ca1_output, previous_ca1_outputs):
    """
    Calculate feature-based distillation loss between current expert and previous experts.
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

def calculate_prototype_regularization_loss(model, inputs, labels, expert_id, device):
    """
    Enhanced prototype regularization loss to anchor DG prototypes to class centers.
    Based on l.py implementation with improvements for TinyImageNet.
    """
    if not hasattr(model, 'dg_prototypes') or model.dg_prototypes is None:
        return torch.tensor(0.0, device=device)
    
    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
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
    Enhanced routing confidence penalty to encourage more decisive gating.
    Based on l.py implementation with improvements for TinyImageNet.
    """
    # Get features and compute DG outputs from all experts
    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
    
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

def calculate_expert_balancing_loss(gate_logits, target_utilization=None):
    """
    Calculate expert balancing loss to encourage uniform expert utilization.
    Based on l.py implementation.
    """
    if gate_logits is None:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else 'cpu')
    
    # Convert to probabilities
    gate_probs = F.softmax(gate_logits, dim=1)
    
    # Calculate expert utilization
    expert_utilization = gate_probs.mean(dim=0)  # [num_experts]
    
    if target_utilization is None:
        # Target uniform utilization
        target_utilization = torch.ones_like(expert_utilization) / expert_utilization.size(0)
    
    # Calculate KL divergence from target utilization
    kl_div = F.kl_div(
        torch.log(expert_utilization + 1e-8), 
        target_utilization, 
        reduction='batchmean'
    )
    
    return kl_div

def calculate_sparsity_loss(dg_outputs):
    """
    Calculate sparsity loss to encourage sparse DG activations.
    Based on l.py implementation.
    """
    # L1 penalty on DG outputs to encourage sparsity
    sparsity_loss = torch.norm(dg_outputs, p=1, dim=1).mean()
    return sparsity_loss

class TinyImageNetFeatureExtractor(nn.Module):
    """
    Feature extractor specifically designed for TinyImageNet (64x64 images).
    """
    def __init__(self, input_channels, use_small_features=False):
        super().__init__()
        self.use_small_features = use_small_features
        
        if use_small_features:
            # Small version: 3→32→64→128
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
                GridCellLayer(32),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
                nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
            )
        else:
            # Standard version: 3→64→128→256
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
                GridCellLayer(64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
                nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
            )
    
    def forward(self, x):
        return self.net(x)

def train_experts_sequentially_tinyimagenet(model, train_loaders, test_loaders, device, args):
    """
    Advanced training function for TinyImageNet with strategies from n.py.
    """
    logger = logging.getLogger()
    logger.info("Starting advanced sequential expert training for TinyImageNet...")
    
    # Initialize class-balanced loss tracking
    class_weights = None
    if args.use_class_balanced_loss:
        logger.info("🔧 Using class-balanced loss weighting")
        # Initialize with equal weights for all TinyImageNet classes (0-199)
        all_classes = list(range(200))  # TinyImageNet has 200 classes total
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
    
    for expert_id in range(args.num_experts):
        train_loader = train_loaders[expert_id]
        test_loader = test_loaders[expert_id]
        
        logger.info(f"\nTraining Expert {expert_id}...")
        
        # Unfreeze current expert and shared layers, freeze others
        for name, p in model.named_parameters():
            is_current_expert = f"hippocampal_experts.{expert_id}" in name or f"output_layers.{expert_id}" in name
            is_shared_component = "feature_extractor" in name
            p.requires_grad = is_current_expert or is_shared_component
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.expert_epochs)
        
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        if expert_id > 0:
            logger.info(f"  Using distillation from {expert_id} previous expert(s)")
        
        # Early stopping variables exactly like l.py
        best_acc = 0
        patience_counter = 0
        patience = getattr(args, 'early_stopping_patience', 10)
        
        for epoch in range(args.expert_epochs):
            model.train()
            total_loss = 0
            total_distillation_loss = 0
            correct = 0
            total = 0
            total_ce_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}/{args.expert_epochs}")
            batch_count = 0
            max_batches = 2 if args.test_run else float('inf')
            
            for inputs, labels in progress_bar:
                if args.test_run and batch_count >= max_batches:
                    break
                batch_count += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Labels are already local (0-19 for each task)
                optimizer.zero_grad()
                
                # Add current batch to replay buffer
                model.add_to_replay_buffer(inputs.detach(), labels.detach(), expert_id)
                
                # --- ONLINE EMA PROTOTYPE UPDATE ---
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[expert_id](features)
                
                # Convert local labels to global class IDs for EMA update
                global_labels = torch.tensor([model.task_classes[expert_id][l.item()] for l in labels], 
                                           dtype=torch.long, device=device)
                model.update_class_prototype_ema(dg_output.detach(), global_labels, expert_id)
                
                # Forward pass
                outputs, _, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                # --- Classification Loss (EXACTLY from l.py) ---
                # Handle optional label smoothing (default to 0.0 if not specified)
                label_smoothing = getattr(args, 'label_smoothing', 0.0)
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                if args.use_class_balanced_loss and class_weights is not None:
                    # Get global class IDs for this batch (convert local to global)
                    global_class_ids = [model.task_classes[expert_id][l.item()] for l in labels]
                    batch_weights = torch.tensor([class_weights[class_id] for class_id in global_class_ids], 
                                               dtype=torch.float32, device=device)
                    # Apply weighted cross-entropy with label smoothing
                    log_probs = F.log_softmax(task_outputs, dim=1)
                    targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)
                    if label_smoothing > 0:
                        targets = targets * (1 - label_smoothing) + label_smoothing / args.classes_per_task
                    classification_loss = -(targets * log_probs).sum(dim=1)
                    classification_loss = (classification_loss * batch_weights).mean()
                else:
                    classification_loss = criterion(task_outputs, labels)
                
                # Initialize total loss
                loss = classification_loss
                total_ce_loss += classification_loss.item()
                
                # --- PUSH-PULL CONTRASTIVE LOSS (EXACTLY from l.py) ---
                push_pull_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply after first expert
                    # Get features and DG outputs from all experts
                    all_dg_outputs = []
                    for i in range(model.num_experts):
                        dg_output_i, _ = model.hippocampal_experts[i](features)
                        all_dg_outputs.append(dg_output_i)
                    
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
                
                # --- Distillation Loss from Previous Experts ---
                distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.distillation_coef > 0:
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
                
                # --- Feature Extractor Distillation Loss ---
                feature_distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.feature_distillation_coef > 0:
                    # Get features from current feature extractor
                    current_features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                    
                    # Get CA1 outputs from all previous experts as target features
                    previous_ca1_outputs = []
                    for prev_expert_id in range(expert_id):
                        with torch.no_grad():
                            _, prev_ca1 = model.hippocampal_experts[prev_expert_id](current_features)
                            previous_ca1_outputs.append(prev_ca1)
                    
                    if len(previous_ca1_outputs) > 0:
                        # Project current features to CA1 dimension for comparison
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
                
                # --- Prototype Regularization Loss ---
                prototype_reg_loss = torch.tensor(0.0, device=device)
                if args.use_prototype_regularization:
                    prototype_reg_loss = calculate_prototype_regularization_loss(
                        model, inputs, labels, expert_id, device
                    )
                    loss += args.prototype_reg_coef * prototype_reg_loss
                
                # --- Enhanced Routing Confidence Penalty ---
                routing_confidence_loss = torch.tensor(0.0, device=device)
                if args.use_routing_confidence_penalty:
                    routing_confidence_loss = calculate_routing_confidence_penalty(
                        model, inputs, device, expert_id
                    )
                    loss += args.routing_confidence_coef * routing_confidence_loss
                
                # --- Expert Balancing Loss ---
                expert_balancing_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply after first expert
                    # Compute gate logits for balancing
                    all_dg_outputs = []
                    for i in range(model.num_experts):
                        dg_output_i, _ = model.hippocampal_experts[i](features)
                        all_dg_outputs.append(dg_output_i)
                    
                    all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [B, num_experts, dg_dim]
                    all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                    
                    if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0:
                        prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                        gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
                        expert_balancing_loss = calculate_expert_balancing_loss(gate_logits)
                        loss += 0.1 * expert_balancing_loss  # Small weight for balancing
                
                # --- Sparsity Loss ---
                sparsity_loss = torch.tensor(0.0, device=device)
                if args.use_sparsity_loss:
                    sparsity_loss = calculate_sparsity_loss(dg_output)
                    loss += args.sparsity_coef * sparsity_loss
                
                # --- EWC Loss ---
                ewc_loss = torch.tensor(0.0, device=device)
                if args.ewc_lambda > 0 and hasattr(model, 'ewc_data') and len(model.ewc_data) > 0:
                    ewc_loss = model.calculate_ewc_loss(args.ewc_lambda)
                    loss += ewc_loss
                
                # --- Gate Supervision Loss ---
                # Compute gate_logits for the current batch (using prototypes and DG outputs)
                all_dg_outputs = []
                for i in range(model.num_experts):
                    dg_output_i, _ = model.hippocampal_experts[i](features)
                    all_dg_outputs.append(dg_output_i)
                all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [B, num_experts, dg_dim]
                all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0:
                    # --- PATCH: Normalize, sharpen, supervise, and diagnose gating ---
                    import torch.nn.functional as F
                    prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                    gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
                    # Mask out uninitialized (zero-norm) prototypes before softmax
                    invalid = (prototypes_norm.abs().sum(dim=1) == 0)  # a bool mask [num_experts]
                    invalid[expert_id] = False  # Never mask out the current expert being trained
                    gate_logits[:, invalid] = -1e9
                    # Sharpen softmax with small temperature
                    TAU = 0.01
                    gate_probs = torch.softmax(gate_logits / TAU, dim=1)
                    # Add small gating cross-entropy loss (oracle_expert is available)
                    oracle_expert = torch.full((inputs.size(0),), expert_id, dtype=torch.long, device=inputs.device)
                    L_gate = torch.nn.CrossEntropyLoss()(gate_logits, oracle_expert)
                    loss = loss + 0.01 * L_gate
                    # Diagnostics
                    sims = gate_logits.flatten()
                    print("[DIAG] Gate logits min:", sims.min().item(), "max:", sims.max().item(), "spread:", (sims.max()-sims.min()).item())
                    print("[DIAG] Sample gate_probs[0]:", gate_probs[0].detach().cpu().numpy())
                    sim_matrix = prototypes_norm @ prototypes_norm.t()
                    n_experts = prototypes_norm.shape[0]
                    off_diag = sim_matrix[~torch.eye(n_experts, dtype=bool, device=prototypes_norm.device)]
                    print("[DIAG] Proto off-diag mean:", off_diag.mean().item(), "std:", off_diag.std().item())
                    # Routing accuracy for this batch
                    chosen = gate_probs.argmax(dim=1)
                    routing_acc = (chosen == oracle_expert).float().mean().item()
                    logger.info(f"Gate CE Loss: {L_gate.item():.4f}, Routing Acc: {routing_acc:.3f}")
                
                    # --- Margin Loss on Gating Cosines ---
                    def calculate_gating_margin_loss(gate_logits, true_experts, margin=0.1):
                        batch_size, num_experts = gate_logits.shape
                        idx = torch.arange(batch_size, device=gate_logits.device)
                        correct_sim = gate_logits[idx, true_experts]            # [B]
                        correct_sim_exp = correct_sim.unsqueeze(1).expand_as(gate_logits)  # [B, E]
                        violations = F.relu(margin + gate_logits - correct_sim_exp)        # [B, E]
                        # Use out-of-place masking to zero out the correct expert
                        mask = torch.ones_like(violations)
                        mask[idx, true_experts] = 0
                        violations = violations * mask
                        return violations.mean()
                    # Add margin loss to the total loss
                    lambda_margin = 0.2
                    margin_loss = calculate_gating_margin_loss(gate_logits, oracle_expert, margin=0.1)
                    loss = loss + lambda_margin * margin_loss
                    # Log margin loss for diagnostics
                    logger.info(f"Gating margin loss: {margin_loss.item():.4f}")

                # --- Margin Loss: Increase margin from 0.1 to 0.3 ---
                # If you have a margin loss for prototype separation, increase its margin parameter
                # Example (if using calculate_enhanced_prototype_loss or similar):
                # margin_loss = F.relu(best_other_sims - correct_sims + 0.3).mean()
                # loss += margin_loss

                # --- Routing Confidence Penalty (Entropy, with epsilon) ---
                entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1).mean()
                loss -= 0.5 * entropy  # Subtract to encourage confident gating

                # --- Expert Balancing Loss (KL to uniform, with epsilon) ---
                util = gate_probs.mean(dim=0)
                uniform = torch.ones_like(util) / util.size(0)
                kl = F.kl_div((util + 1e-8).log(), uniform, reduction='batchmean')
                loss += 0.01 * kl  # Keep at 0.01 for stability

                # --- NaN checks for diagnostics ---
                if torch.isnan(gate_logits).any():
                    print("Warning: NaN in gate_logits!")
                if torch.isnan(gate_probs).any():
                    print("Warning: NaN in gate_probs!")
                if torch.isnan(loss):
                    print("Warning: NaN in loss!")

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_distillation_loss += distillation_loss.item()
                _, predicted = task_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Build postfix exactly like l.py
                postfix = {
                    'Loss': f'{loss.item():.4f}',
                    'CE': f'{classification_loss.item():.4f}',
                    'Distill': f'{distillation_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                
                # Add conditional losses exactly like l.py
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
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _ = model(inputs, task_id=expert_id)
                    start_idx = expert_id * model.classes_per_task
                    end_idx = start_idx + model.classes_per_task
                    task_outputs = outputs[:, start_idx:end_idx]
                    _, predicted = task_outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Best accuracy tracking exactly like l.py
            acc = 100. * val_correct / val_total
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            
            # Log distillation info exactly like l.py
            if expert_id > 0 and args.distillation_coef > 0:
                logger.info(f"  Epoch {epoch+1} | Test Acc: {100.*val_correct/val_total:.2f}% | Distill Loss: {total_distillation_loss/len(train_loader):.4f}")
            else:
                logger.info(f"  Epoch {epoch+1} | Test Acc: {100.*val_correct/val_total:.2f}%")
        
        # === EWC Step 1: Compute and Store Importance (AFTER training is done) ===
        # Use enhanced Fisher computation for better robustness
        fisher_matrix = model.compute_fisher_importance_enhanced(train_loader, device, num_samples=500, num_forward_passes=3)
        star_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Analyze Fisher quality
        fisher_analysis = model.analyze_fisher_quality(fisher_matrix)
        
        model.ewc_data.append({'fisher': fisher_matrix, 'star_params': star_params})
        logger.info(f"🔧 EWC: Computed Fisher importance for expert {expert_id} ({len(fisher_matrix)} parameter groups, {fisher_analysis.get('total_parameters', 0):,} total parameters)")
        # =======================================================================
        
        # --- Update prototype for this expert after training (streaming mean, memory-efficient) ---
        model.eval()
        proto_sum = None
        count = 0
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[expert_id](features)
                dg_output_cpu = dg_output.detach().cpu()
                if proto_sum is None:
                    proto_sum = dg_output_cpu.sum(dim=0)
                else:
                    proto_sum += dg_output_cpu.sum(dim=0)
                count += dg_output_cpu.size(0)
        if count > 0:
            prototype = proto_sum / count
            model.dg_prototypes[expert_id] = prototype.to(device)
        model.prototypes_computed = True  # Allow gating to use updated prototypes

        # --- FREEZE EXPERT PROTOTYPES AFTER TRAINING ---
        model.freeze_expert_prototypes(expert_id)
        model.trained_experts += 1
        # Update DG prototypes from EMA (optional, if you want to keep EMA logic)
        model.update_dg_prototypes_from_ema()
        
        logger.info(f"✅ Expert {expert_id} training complete.")
    

class TinyImageNetDGGatedMoE(DGGatedHippocampalMoE):
    """TinyImageNet version using the EXACT same model as l.py but with TinyImageNet feature extractor."""

    def __init__(self, num_experts, classes_per_task, input_channels=3, target_sparsity=0.05, memory_size=200, use_small_features=False):
        # Call parent with correct parameters (EXACT same as l.py)
        super().__init__(num_experts, classes_per_task, input_channels, target_sparsity, memory_size, use_small_features)
        
        # ONLY change: Replace the feature extractor with TinyImageNet-specific one
        self.feature_extractor = TinyImageNetFeatureExtractor(input_channels, use_small_features=use_small_features)
        
        # Recalculate feature dimension for TinyImageNet (64x64 images)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 64, 64)  # TinyImageNet size
            dummy_output = self.feature_extractor(dummy_input)
            feature_extractor_output_dim = dummy_output.numel()
        
        # Re-create experts with correct input dimension (EXACT same as l.py otherwise)
        self.hippocampal_experts = nn.ModuleList([
            CustomEnhancedHippocampalExpert(
                input_dim=feature_extractor_output_dim,
                dg_dim=dg_dim,  # <--- Use dg_dim variable here
                ca3_dim=256,
                target_sparsity=target_sparsity,
                dropout_rate=0.1
            ) for _ in range(num_experts)
        ])
        
        # Update feature_to_ca1 projection for feature distillation
        self.feature_to_ca1 = nn.Linear(feature_extractor_output_dim, 128)
        
        # Fix: Initialize EMA tracking for full TinyImageNet class space (200 classes)
        self.num_classes = 200  # TinyImageNet has 200 classes total
        
        logging.info(f"🔧 TinyImageNet DG-Gated MoE: EXACT same model as l.py but with TinyImageNet feature extractor")
        logging.info(f"🔧 Feature dimension: {feature_extractor_output_dim} (64x64 images)")
        logging.info(f"🔧 DG/prototype dimension: {dg_dim}")
        logging.info(f"🔧 All other components: IDENTICAL to l.py")

    # Use the exact same forward method from l.py - no need to duplicate!
    pass

    # Use the exact same methods from l.py - no need to duplicate!
    pass

def evaluate_dg_gated_model_standard(model, test_loaders, num_tasks, classes_per_task, device):
    """
    Custom evaluation function for DG-gated models that handles the fact that
    only one expert's output is available at a time.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMING FINAL EVALUATION ON ALL TASKS (DG-GATED)")
    logger.info("="*80)

    model.eval()
    
    # --- Task-IL Evaluation (Oracle) ---
    task_il_correct = 0
    task_il_total = 0
    expert_accuracies = []

    logger.info("Evaluating Task-IL Performance (with Oracle Task-ID)...")
    
    with torch.no_grad():
        for task_id in range(num_tasks):
            test_loader = test_loaders[task_id]
            expert_correct = 0
            expert_total = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # For Task-IL, we use oracle routing - directly use the correct expert
                # This bypasses the DG-gating mechanism
                outputs, _, _ = model(inputs, task_id=task_id)  # Use oracle task_id
                
                # Get the expert's output for this task
                start_idx = task_id * classes_per_task
                end_idx = start_idx + classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]

                _, predicted = torch.max(task_outputs, 1)
                expert_correct += (predicted == labels).sum().item()
                expert_total += labels.size(0)

            expert_acc = (expert_correct / expert_total) * 100 if expert_total > 0 else 0.0
            expert_accuracies.append(expert_acc)
            task_il_correct += expert_correct
            task_il_total += expert_total
            logger.info(f"  - Accuracy for Task {task_id+1}: {expert_acc:.2f}%")

    task_il_accuracy = (task_il_correct / task_il_total) * 100 if task_il_total > 0 else 0.0
    
    # --- Class-IL & Routing Evaluation ---
    class_il_correct = 0
    class_il_total = 0
    routing_correct = 0
    routing_total = 0

    logger.info("\nEvaluating Class-IL & Routing Performance (no Oracle)...")
    with torch.no_grad():
        for task_id in range(num_tasks):
            test_loader = test_loaders[task_id]
            for inputs, local_labels in tqdm(test_loader, desc=f"Testing Task {task_id+1}", leave=False):
                inputs, local_labels = inputs.to(device), local_labels.to(device)
                
                # Convert local labels (e.g., 0-19) to global labels
                global_labels = local_labels + task_id * classes_per_task

                # Use the DG-gated forward pass (no oracle)
                outputs, _, analysis_data = model(inputs)
                
                # Ensure all tensors have the same batch size
                batch_size = outputs.size(0)
                if batch_size != global_labels.size(0):
                    # Truncate global_labels to match outputs batch size
                    global_labels = global_labels[:batch_size]
                    logger.warning(f"⚠️  Task {task_id+1}: Truncated global_labels from {global_labels.size(0)} to {batch_size}")

                # 1. Class-IL Accuracy
                _, predicted_cls = torch.max(outputs, 1)
                class_il_correct += (predicted_cls == global_labels).sum().item()
                class_il_total += global_labels.size(0)

                # 2. Routing Accuracy
                chosen_experts = analysis_data.get('chosen_experts', torch.zeros(batch_size, dtype=torch.long, device=device))
                if chosen_experts.size(0) != batch_size:
                    chosen_experts = chosen_experts[:batch_size]
                routing_correct += (chosen_experts == task_id).sum().item()
                routing_total += batch_size

    class_il_accuracy = (class_il_correct / class_il_total) * 100 if class_il_total > 0 else 0.0
    routing_accuracy = (routing_correct / routing_total) * 100 if routing_total > 0 else 0.0

    # --- Summary ---
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL PERFORMANCE SUMMARY (DG-GATED)")
    logger.info("="*80)
    logger.info(f"  - Expert Accuracies (Task-IL): {[f'{acc:.1f}%' for acc in expert_accuracies]}")
    logger.info(f"  - Average Task-IL Accuracy: {task_il_accuracy:.2f}%")
    logger.info(f"  - Class-IL Accuracy: {class_il_accuracy:.2f}%")
    logger.info(f"  - Routing Accuracy: {routing_accuracy:.2f}%")
    logger.info(f"  - Forgetting Gap (Task-IL vs Class-IL): {task_il_accuracy - class_il_accuracy:.2f}%")

    results = {
        'expert_accuracies': expert_accuracies,
        'task_il_accuracy': task_il_accuracy,
        'class_il_accuracy': class_il_accuracy,
        'routing_accuracy': routing_accuracy,
        'task_class_gap': task_il_accuracy - class_il_accuracy
    }

    return results

def evaluate_final_performance_dg_gated_tinyimagenet(model, test_loaders, task_classes, device):
    """
    TinyImageNet-specific evaluation function that handles local task labels correctly.
    """
    logging.info("=" * 80)
    logging.info("FINAL DG-GATED PERFORMANCE EVALUATION (TinyImageNet)")
    logging.info("=" * 80)
    
    model.eval()
    
    # Task-IL evaluation (oracle routing)
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
            task_class_list = task_classes[expert_id]  # Global class IDs for this task
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Labels are already local (0-19), no conversion needed
                local_labels = labels
                
                outputs, _, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                loss = F.cross_entropy(task_outputs, local_labels, reduction='none')
                probs = F.softmax(task_outputs, dim=1)
                conf = probs.max(dim=1)[0]
                _, predicted = torch.max(task_outputs, 1)
                
                # Convert local labels back to global for tracking
                global_labels = torch.tensor([task_class_list[l.item()] for l in local_labels], 
                                           dtype=torch.long, device=device)
                
                all_true_taskil.extend(global_labels.cpu().numpy())
                all_pred_taskil.extend((predicted + start_idx).cpu().numpy())
                all_expert_taskil.extend([expert_id] * len(labels))
                all_loss_taskil.extend(loss.cpu().numpy())
                all_conf_taskil.extend(conf.cpu().numpy())
                
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
                original_strategy = model.gating_strategy
                model.gating_strategy = 'hard'
                outputs, _, analysis_data = model(inputs)
                model.gating_strategy = original_strategy
            else:
                outputs, _, analysis_data = model(inputs)
            
            # Convert local labels to global for Class-IL evaluation
            # We need to determine which task this sample belongs to
            # For simplicity, we'll use the task that contains the most similar prototype
            features = model.feature_extractor(inputs).view(inputs.size(0), -1)
            
            # Get DG outputs and find best matching expert
            best_expert = torch.zeros(inputs.size(0), dtype=torch.long, device=device)
            global_labels = torch.zeros(inputs.size(0), dtype=torch.long, device=device)
            
            for i in range(inputs.size(0)):
                # Find which task this sample belongs to by checking all task classes
                sample_label = labels[i].item()
                sample_task = None
                for task_id, task_class_list in enumerate(task_classes):
                    if sample_label < len(task_class_list):  # Local label within this task
                        sample_task = task_id
                        global_labels[i] = task_class_list[sample_label]
                        break
                
                if sample_task is not None:
                    best_expert[i] = sample_task
            
            loss = F.cross_entropy(outputs, global_labels, reduction='none')
            probs = F.softmax(outputs, dim=1)
            conf = probs.max(dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            
            chosen_experts = analysis_data['chosen_experts'].cpu().numpy() if analysis_data.get('chosen_experts') is not None else np.full(len(labels), -1)
            
            all_true_classil.extend(global_labels.cpu().numpy())
            all_pred_classil.extend(predicted.cpu().numpy())
            all_expert_classil.extend(chosen_experts)
            all_loss_classil.extend(loss.cpu().numpy())
            all_conf_classil.extend(conf.cpu().numpy())
            
            class_il_correct += (predicted == global_labels).sum().item()
            class_il_total += global_labels.size(0)
    
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
    
    # Log results
    logging.info("=== PER-CLASS CONFUSION MATRIX (TASK-IL) ===\n" + str(cm_taskil))
    logging.info("=== PER-CLASS CONFUSION MATRIX (CLASS-IL) ===\n" + str(cm_classil))
    logging.info("=== PER-CLASS STATS (TASK-IL) ===\n" + str(stats_taskil))
    logging.info("=== PER-CLASS STATS (CLASS-IL) ===\n" + str(stats_classil))
    logging.info("=== ROUTING COUNTS (CLASS-IL) ===\n" + str(routing_counts))
    
    logging.info(f"\nFINAL RESULTS:")
    logging.info(f"  - Task-IL Accuracy (Oracle): {task_il_accuracy:.2f}%")
    logging.info(f"  - Class-IL Accuracy (DG-Gated): {class_il_accuracy:.2f}%")
    
    return {
        'task_il_accuracy': task_il_accuracy, 
        'class_il_accuracy': class_il_accuracy
    }

def calculate_feature_distillation_loss(current_dg_output, previous_dg_outputs, current_ca1_output, previous_ca1_outputs):
    """
    Calculate feature-based distillation loss between current expert and previous experts.
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

def calculate_prototype_regularization_loss(model, inputs, labels, expert_id, device):
    """
    Enhanced prototype regularization loss to anchor DG prototypes to class centers.
    Based on l.py implementation with improvements for TinyImageNet.
    """
    if not hasattr(model, 'dg_prototypes') or model.dg_prototypes is None:
        return torch.tensor(0.0, device=device)
    
    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
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
    Enhanced routing confidence penalty to encourage more decisive gating.
    Based on l.py implementation with improvements for TinyImageNet.
    """
    # Get features and compute DG outputs from all experts
    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
    
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

def calculate_expert_balancing_loss(gate_logits, target_utilization=None):
    """
    Calculate expert balancing loss to encourage uniform expert utilization.
    Based on l.py implementation.
    """
    if gate_logits is None:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else 'cpu')
    
    # Convert to probabilities
    gate_probs = F.softmax(gate_logits, dim=1)
    
    # Calculate expert utilization
    expert_utilization = gate_probs.mean(dim=0)  # [num_experts]
    
    if target_utilization is None:
        # Target uniform utilization
        target_utilization = torch.ones_like(expert_utilization) / expert_utilization.size(0)
    
    # Calculate KL divergence from target utilization
    kl_div = F.kl_div(
        torch.log(expert_utilization + 1e-8), 
        target_utilization, 
        reduction='batchmean'
    )
    
    return kl_div

def calculate_sparsity_loss(dg_outputs):
    """
    Calculate sparsity loss to encourage sparse DG activations.
    Based on l.py implementation.
    """
    # L1 penalty on DG outputs to encourage sparsity
    sparsity_loss = torch.norm(dg_outputs, p=1, dim=1).mean()
    return sparsity_loss

def train_experts_sequentially_tinyimagenet(model, train_loaders, test_loaders, device, args):
    """
    Advanced training function for TinyImageNet with strategies from n.py.
    """
    logger = logging.getLogger()
    logger.info("Starting advanced sequential expert training for TinyImageNet...")
    
    # Initialize class-balanced loss tracking
    class_weights = None
    if args.use_class_balanced_loss:
        logger.info("🔧 Using class-balanced loss weighting")
        # Initialize with equal weights for all TinyImageNet classes (0-199)
        all_classes = list(range(200))  # TinyImageNet has 200 classes total
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
    
    for expert_id in range(args.num_experts):
        train_loader = train_loaders[expert_id]
        test_loader = test_loaders[expert_id]
        
        logger.info(f"\nTraining Expert {expert_id}...")
        
        # Unfreeze current expert and shared layers, freeze others
        for name, p in model.named_parameters():
            is_current_expert = f"hippocampal_experts.{expert_id}" in name or f"output_layers.{expert_id}" in name
            is_shared_component = "feature_extractor" in name
            p.requires_grad = is_current_expert or is_shared_component
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.expert_epochs)
        
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        if expert_id > 0:
            logger.info(f"  Using distillation from {expert_id} previous expert(s)")
        
        # Early stopping variables exactly like l.py
        best_acc = 0
        patience_counter = 0
        patience = getattr(args, 'early_stopping_patience', 10)
        
        for epoch in range(args.expert_epochs):
            model.train()
            total_loss = 0
            total_distillation_loss = 0
            correct = 0
            total = 0
            total_ce_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Expert {expert_id} Epoch {epoch+1}/{args.expert_epochs}")
            batch_count = 0
            max_batches = 2 if args.test_run else float('inf')
            
            for inputs, labels in progress_bar:
                if args.test_run and batch_count >= max_batches:
                    break
                batch_count += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Labels are already local (0-19 for each task)
                optimizer.zero_grad()
                
                # Add current batch to replay buffer
                model.add_to_replay_buffer(inputs.detach(), labels.detach(), expert_id)
                
                # --- ONLINE EMA PROTOTYPE UPDATE ---
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[expert_id](features)
                
                # Convert local labels to global class IDs for EMA update
                global_labels = torch.tensor([model.task_classes[expert_id][l.item()] for l in labels], 
                                           dtype=torch.long, device=device)
                model.update_class_prototype_ema(dg_output.detach(), global_labels, expert_id)
                
                # Forward pass
                outputs, _, _ = model(inputs, task_id=expert_id)
                start_idx = expert_id * model.classes_per_task
                end_idx = start_idx + model.classes_per_task
                task_outputs = outputs[:, start_idx:end_idx]
                
                # --- Classification Loss (EXACTLY from l.py) ---
                # Handle optional label smoothing (default to 0.0 if not specified)
                label_smoothing = getattr(args, 'label_smoothing', 0.0)
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                if args.use_class_balanced_loss and class_weights is not None:
                    # Get global class IDs for this batch (convert local to global)
                    global_class_ids = [model.task_classes[expert_id][l.item()] for l in labels]
                    batch_weights = torch.tensor([class_weights[class_id] for class_id in global_class_ids], 
                                               dtype=torch.float32, device=device)
                    # Apply weighted cross-entropy with label smoothing
                    log_probs = F.log_softmax(task_outputs, dim=1)
                    targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)
                    if label_smoothing > 0:
                        targets = targets * (1 - label_smoothing) + label_smoothing / args.classes_per_task
                    classification_loss = -(targets * log_probs).sum(dim=1)
                    classification_loss = (classification_loss * batch_weights).mean()
                else:
                    classification_loss = criterion(task_outputs, labels)
                
                # Initialize total loss
                loss = classification_loss
                total_ce_loss += classification_loss.item()
                
                # --- PUSH-PULL CONTRASTIVE LOSS (EXACTLY from l.py) ---
                push_pull_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply after first expert
                    # Get features and DG outputs from all experts
                    all_dg_outputs = []
                    for i in range(model.num_experts):
                        dg_output_i, _ = model.hippocampal_experts[i](features)
                        all_dg_outputs.append(dg_output_i)
                    
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
                
                # --- Distillation Loss from Previous Experts ---
                distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.distillation_coef > 0:
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
                
                # --- Feature Extractor Distillation Loss ---
                feature_distillation_loss = torch.tensor(0.0, device=device)
                if expert_id > 0 and args.feature_distillation_coef > 0:
                    # Get features from current feature extractor
                    current_features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                    
                    # Get CA1 outputs from all previous experts as target features
                    previous_ca1_outputs = []
                    for prev_expert_id in range(expert_id):
                        with torch.no_grad():
                            _, prev_ca1 = model.hippocampal_experts[prev_expert_id](current_features)
                            previous_ca1_outputs.append(prev_ca1)
                    
                    if len(previous_ca1_outputs) > 0:
                        # Project current features to CA1 dimension for comparison
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
                
                # --- Prototype Regularization Loss ---
                prototype_reg_loss = torch.tensor(0.0, device=device)
                if args.use_prototype_regularization:
                    prototype_reg_loss = calculate_prototype_regularization_loss(
                        model, inputs, labels, expert_id, device
                    )
                    loss += args.prototype_reg_coef * prototype_reg_loss
                
                # --- Enhanced Routing Confidence Penalty ---
                routing_confidence_loss = torch.tensor(0.0, device=device)
                if args.use_routing_confidence_penalty:
                    routing_confidence_loss = calculate_routing_confidence_penalty(
                        model, inputs, device, expert_id
                    )
                    loss += args.routing_confidence_coef * routing_confidence_loss
                
                # --- Expert Balancing Loss ---
                expert_balancing_loss = torch.tensor(0.0, device=device)
                if expert_id > 0:  # Only apply after first expert
                    # Compute gate logits for balancing
                    all_dg_outputs = []
                    for i in range(model.num_experts):
                        dg_output_i, _ = model.hippocampal_experts[i](features)
                        all_dg_outputs.append(dg_output_i)
                    
                    all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [B, num_experts, dg_dim]
                    all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                    
                    if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0:
                        prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                        gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
                        expert_balancing_loss = calculate_expert_balancing_loss(gate_logits)
                        loss += 0.1 * expert_balancing_loss  # Small weight for balancing
                
                # --- Sparsity Loss ---
                sparsity_loss = torch.tensor(0.0, device=device)
                if args.use_sparsity_loss:
                    sparsity_loss = calculate_sparsity_loss(dg_output)
                    loss += args.sparsity_coef * sparsity_loss
                
                # --- EWC Loss ---
                ewc_loss = torch.tensor(0.0, device=device)
                if args.ewc_lambda > 0 and hasattr(model, 'ewc_data') and len(model.ewc_data) > 0:
                    ewc_loss = model.calculate_ewc_loss(args.ewc_lambda)
                    loss += ewc_loss
                
                # --- Gate Supervision Loss ---
                # Compute gate_logits for the current batch (using prototypes and DG outputs)
                all_dg_outputs = []
                for i in range(model.num_experts):
                    dg_output_i, _ = model.hippocampal_experts[i](features)
                    all_dg_outputs.append(dg_output_i)
                all_dg_outputs = torch.stack(all_dg_outputs, dim=1)  # [B, num_experts, dg_dim]
                all_dg_outputs_norm = F.normalize(all_dg_outputs, p=2, dim=2)
                if hasattr(model, 'dg_prototypes') and model.dg_prototypes.numel() > 0:
                    # --- PATCH: Normalize, sharpen, supervise, and diagnose gating ---
                    prototypes_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(device)
                    gate_logits = torch.einsum('bne,ne->bn', all_dg_outputs_norm, prototypes_norm)
                    # Mask out uninitialized (zero-norm) prototypes before softmax
                    invalid = (prototypes_norm.abs().sum(dim=1) == 0)  # a bool mask [num_experts]
                    invalid[expert_id] = False  # Never mask out the current expert being trained
                    gate_logits[:, invalid] = -1e9
                    # Sharpen softmax with small temperature
                    TAU = 0.01
                    gate_probs = torch.softmax(gate_logits / TAU, dim=1)
                    # Add small gating cross-entropy loss (oracle_expert is available)
                    oracle_expert = torch.full((inputs.size(0),), expert_id, dtype=torch.long, device=inputs.device)
                    L_gate = torch.nn.CrossEntropyLoss()(gate_logits, oracle_expert)
                    loss = loss + 0.01 * L_gate
                    # Diagnostics
                    sims = gate_logits.flatten()
                    print("[DIAG] Gate logits min:", sims.min().item(), "max:", sims.max().item(), "spread:", (sims.max()-sims.min()).item())
                    print("[DIAG] Sample gate_probs[0]:", gate_probs[0].detach().cpu().numpy())
                    sim_matrix = prototypes_norm @ prototypes_norm.t()
                    n_experts = prototypes_norm.shape[0]
                    off_diag = sim_matrix[~torch.eye(n_experts, dtype=bool, device=prototypes_norm.device)]
                    print("[DIAG] Proto off-diag mean:", off_diag.mean().item(), "std:", off_diag.std().item())
                    # Routing accuracy for this batch
                    chosen = gate_probs.argmax(dim=1)
                    routing_acc = (chosen == oracle_expert).float().mean().item()
                    logger.info(f"Gate CE Loss: {L_gate.item():.4f}, Routing Acc: {routing_acc:.3f}")
                
                    # --- Margin Loss on Gating Cosines ---
                    def calculate_gating_margin_loss(gate_logits, true_experts, margin=0.1):
                        batch_size, num_experts = gate_logits.shape
                        idx = torch.arange(batch_size, device=gate_logits.device)
                        correct_sim = gate_logits[idx, true_experts]            # [B]
                        correct_sim_exp = correct_sim.unsqueeze(1).expand_as(gate_logits)  # [B, E]
                        violations = F.relu(margin + gate_logits - correct_sim_exp)        # [B, E]
                        # Use out-of-place masking to zero out the correct expert
                        mask = torch.ones_like(violations)
                        mask[idx, true_experts] = 0
                        violations = violations * mask
                        return violations.mean()
                    # Add margin loss to the total loss
                    lambda_margin = 0.2
                    margin_loss = calculate_gating_margin_loss(gate_logits, oracle_expert, margin=0.1)
                    loss = loss + lambda_margin * margin_loss
                    # Log margin loss for diagnostics
                    logger.info(f"Gating margin loss: {margin_loss.item():.4f}")

                # --- Margin Loss: Increase margin from 0.1 to 0.3 ---
                # If you have a margin loss for prototype separation, increase its margin parameter
                # Example (if using calculate_enhanced_prototype_loss or similar):
                # margin_loss = F.relu(best_other_sims - correct_sims + 0.3).mean()
                # loss += margin_loss

                # --- Routing Confidence Penalty (Entropy, with epsilon) ---
                entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1).mean()
                loss -= 0.5 * entropy  # Subtract to encourage confident gating

                # --- Expert Balancing Loss (KL to uniform, with epsilon) ---
                util = gate_probs.mean(dim=0)
                uniform = torch.ones_like(util) / util.size(0)
                kl = F.kl_div((util + 1e-8).log(), uniform, reduction='batchmean')
                loss += 0.01 * kl  # Keep at 0.01 for stability

                # --- NaN checks for diagnostics ---
                if torch.isnan(gate_logits).any():
                    print("Warning: NaN in gate_logits!")
                if torch.isnan(gate_probs).any():
                    print("Warning: NaN in gate_probs!")
                if torch.isnan(loss):
                    print("Warning: NaN in loss!")

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_distillation_loss += distillation_loss.item()
                _, predicted = task_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Build postfix exactly like l.py
                postfix = {
                    'Loss': f'{loss.item():.4f}',
                    'CE': f'{classification_loss.item():.4f}',
                    'Distill': f'{distillation_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                
                # Add conditional losses exactly like l.py
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
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _ = model(inputs, task_id=expert_id)
                    start_idx = expert_id * model.classes_per_task
                    end_idx = start_idx + model.classes_per_task
                    task_outputs = outputs[:, start_idx:end_idx]
                    _, predicted = task_outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Best accuracy tracking exactly like l.py
            acc = 100. * val_correct / val_total
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            
            # Log distillation info exactly like l.py
            if expert_id > 0 and args.distillation_coef > 0:
                logger.info(f"  Epoch {epoch+1} | Test Acc: {100.*val_correct/val_total:.2f}% | Distill Loss: {total_distillation_loss/len(train_loader):.4f}")
            else:
                logger.info(f"  Epoch {epoch+1} | Test Acc: {100.*val_correct/val_total:.2f}%")
        
        # === EWC Step 1: Compute and Store Importance (AFTER training is done) ===
        # Use enhanced Fisher computation for better robustness
        fisher_matrix = model.compute_fisher_importance_enhanced(train_loader, device, num_samples=500, num_forward_passes=3)
        star_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Analyze Fisher quality
        fisher_analysis = model.analyze_fisher_quality(fisher_matrix)
        
        model.ewc_data.append({'fisher': fisher_matrix, 'star_params': star_params})
        logger.info(f"🔧 EWC: Computed Fisher importance for expert {expert_id} ({len(fisher_matrix)} parameter groups, {fisher_analysis.get('total_parameters', 0):,} total parameters)")
        # =======================================================================
        
        # --- Update prototype for this expert after training (streaming mean, memory-efficient) ---
        model.eval()
        proto_sum = None
        count = 0
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[expert_id](features)
                dg_output_cpu = dg_output.detach().cpu()
                if proto_sum is None:
                    proto_sum = dg_output_cpu.sum(dim=0)
                else:
                    proto_sum += dg_output_cpu.sum(dim=0)
                count += dg_output_cpu.size(0)
        if count > 0:
            prototype = proto_sum / count
            model.dg_prototypes[expert_id] = prototype.to(device)
        model.prototypes_computed = True  # Allow gating to use updated prototypes

        # --- FREEZE EXPERT PROTOTYPES AFTER TRAINING ---
        model.freeze_expert_prototypes(expert_id)
        model.trained_experts += 1
        # Update DG prototypes from EMA (optional, if you want to keep EMA logic)
        model.update_dg_prototypes_from_ema()
        
        logger.info(f"✅ Expert {expert_id} training complete.")
    
    # Compute DG prototypes after all experts are trained
    logger.info("Computing DG prototypes for all experts...")
    model.compute_dg_prototypes(train_loaders, device)
    model.trained_experts = args.num_experts  # Mark all experts as trained
    logger.info("✅ DG prototypes computed successfully.")

def build_arg_parser():
    p = ArgumentParser(description='TinyImageNet Hippocampal MoE V7 – DG-Gated Model')
    p.add_argument('--data_path', type=str, default='./data/tiny-imagenet-200')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--classes_per_task', type=int, default=20)
    p.add_argument('--router_epochs', type=int, default=20)
    p.add_argument('--expert_epochs', type=int, default=20)
    p.add_argument('--joint_epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--router_lr', type=float, default=5e-4)
    p.add_argument('--expert_lr', type=float, default=5e-4)
    p.add_argument('--joint_lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--deep_sparsity_levels', type=float, nargs='+', default=[0.5, 0.35, 0.25])
    p.add_argument('--gating_loss_coef', type=float, default=2.0)
    p.add_argument('--balance_loss_coef', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--test_run', action='store_true')
    
    # Additional arguments needed for DG-gated model
    p.add_argument('--dg_sparsity', type=float, default=0.05)
    p.add_argument('--dropout_rate', type=float, default=0.1)
    p.add_argument('--memory_size', type=int, default=200)
    p.add_argument('--gating_temperature', type=float, default=0.1, help='Temperature for softmax gating (balanced gating)')
    p.add_argument('--gating_strategy', type=str, default='soft', choices=['soft', 'hard', 'top2', 'soft_hard'])
    p.add_argument('--distillation_coef', type=float, default=0.1)
    p.add_argument('--feature_distillation_coef', type=float, default=0.05)
    p.add_argument('--replay_loss_coef', type=float, default=0.1)
    p.add_argument('--contrastive_epochs', type=int, default=15)
    p.add_argument('--contrastive_lr', type=float, default=1e-4)
    p.add_argument('--contrastive_margin', type=float, default=0.6)
    p.add_argument('--enable_phase2', action='store_true', help='Enable Phase 2 contrastive fine-tuning with replay')
    p.add_argument('--use_class_balanced_loss', action='store_true', help='Enable class-balanced loss weighting')
    p.add_argument('--class_balance_epsilon', type=float, default=0.1, help='Small constant to prevent division by zero in class weights')
    p.add_argument('--class_balance_smoothing', type=float, default=0.1, help='Smoothing factor for class weights to prevent instability')
    p.add_argument('--use_routing_confidence_penalty', action='store_true', help='Enable routing confidence penalty to encourage decisive gating')
    p.add_argument('--routing_confidence_coef', type=float, default=0.05, help='Coefficient for routing confidence penalty')
    p.add_argument('--sparsity_coef', type=float, default=0.01, help='Coefficient for sparsity loss')
    p.add_argument('--use_sparsity_loss', action='store_true', help='Use sparsity loss to encourage sparse DG activations')
    p.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for cross-entropy loss (from l.py)')
    p.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience (from l.py)')
    p.add_argument('--use_prototype_regularization', action='store_true', help='Enable prototype regularization to anchor DG prototypes to class centers')
    p.add_argument('--prototype_reg_coef', type=float, default=0.1, help='Coefficient for prototype regularization loss')
    p.add_argument('--ewc_lambda', type=float, default=1000.0, help='EWC regularization strength to prevent forgetting')
    p.add_argument('--use_small_features', action='store_true', help='Use small feature extractor (128 channels instead of 256) for 48% FLOP savings')
    p.add_argument('--save_dir', type=str, default=None)
    
    return p

def main():
    args = build_arg_parser().parse_args()
    # Ensure compatibility with V4 trainer which expects args.num_experts
    args.num_experts = args.num_tasks

    # Setup
    log_dir = setup_logging()
    logger = logging.getLogger()
    logger.info('🚀 TinyImageNet V7 DG-Gated Model Training')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if args.test_run:
        logger.info('TEST-RUN mode: epochs set to 1 and 2 batches per epoch for training only.')
        args.router_epochs = args.expert_epochs = args.joint_epochs = 1
        args.test_run = True  # Pass this flag to training functions

    # Data
    train_loaders, test_loaders, task_classes = create_tinyimagenet_tasks(
        data_dir=args.data_path,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        batch_size=args.batch_size
    )

    # Model
    model = TinyImageNetDGGatedMoE(
        num_experts=args.num_tasks,
        classes_per_task=args.classes_per_task,
        input_channels=3,  # TinyImageNet has 3 channels
        target_sparsity=args.dg_sparsity,
        memory_size=args.memory_size,
        use_small_features=args.use_small_features
    ).to(device)
    logger.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Set task classes for the model
    model.set_task_classes(task_classes)
    
    # Initialize prototype tracking early to avoid IndexError
    logger.info("🔧 Initializing prototype tracking for TinyImageNet...")
    # dg_dim = 512  # DG dimension from the model (REMOVE THIS LINE)
    model.initialize_prototype_tracking(dg_dim, 200, device)  # 200 classes for TinyImageNet
    logger.info("✅ Prototype tracking initialized")
    
    # Set gating parameters
    model.set_gating_temperature(args.gating_temperature)
    model.set_gating_strategy(args.gating_strategy)

    # Training phases - using the DG-gated training functions from n.py
    logger.info("Starting DG-Gated training phases...")
    
    # Phase 1: Train experts sequentially
    logger.info(f"Phase 1: Training experts sequentially with lr={args.expert_lr}")
    args.learning_rate = args.expert_lr
    train_experts_sequentially_tinyimagenet(model, train_loaders, test_loaders, device, args)


    # DG Separation Diagnostic
    logger.info("Running DG Separation Diagnostic after Phase 1...")
    all_dg = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            for inputs, _ in loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[task_id](features)
                all_dg.append(dg_output.cpu().numpy())
                all_labels.append(np.full(dg_output.shape[0], task_id))
    all_dg = np.concatenate(all_dg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    separation_metrics = dg_separation_diagnostic(all_dg, all_labels, logger=logger)

    # Phase 2: Joint Contrastive Fine-Tuning (optional)
    if args.enable_phase2:
        logger.info("🚀 Starting Phase 2: Joint Contrastive Fine-Tuning with Replay")
        phase2_contrastive_tuning(model, train_loaders, device, args, log_dir)
        
        # Refresh prototypes after Phase 2
        logger.info("🔄 Refreshing prototypes after Phase 2...")
        refresh_prototypes_light(model, train_loaders, device)
        logger.info("✅ Final prototype refresh after Phase 2 complete")
        proto_phase2 = model.dg_prototypes.cpu().numpy().copy()
    else:
        logger.info("⏭️ Skipping Phase 2 (use --enable_phase2 to enable)")

    # Evaluation
    logger.info("Running final evaluation...")
    analyze_dg_gated_model(model, test_loaders, device, os.path.join(log_dir, 'analysis'))

    # --- Final standardized evaluation (Task-IL, Class-IL, Routing) ---
    # For DG-gated models, we need custom evaluation since standard evaluation expects all experts' outputs
    # but DG-gating only provides one expert's output
    eval_results = evaluate_dg_gated_model_standard(model, test_loaders, args.num_tasks, args.classes_per_task, device)
    
    # DEBUG: Log detailed routing analysis
    logger.info("="*60)
    logger.info("🔍 DEBUG: ROUTING ANALYSIS")
    logger.info("="*60)
    logger.info(f"Task-IL Accuracy: {eval_results['task_il_accuracy']:.2f}% (Oracle routing)")
    logger.info(f"Class-IL Accuracy: {eval_results['class_il_accuracy']:.2f}% (DG-gated routing)")
    logger.info(f"Routing Accuracy: {eval_results.get('routing_accuracy', 0.0):.2f}% (Expert selection)")
    
    # Calculate expected Class-IL based on routing and expert performance
    routing_acc = eval_results.get('routing_accuracy', 0.0)
    if routing_acc is None:
        logger.warning("⚠️ Routing accuracy is None! This is a critical error.")
        routing_acc = 0.0
    
    task_il_acc = eval_results['task_il_accuracy']
    expected_class_il = (routing_acc / 100.0) * (task_il_acc / 100.0) * 100.0
    logger.info(f"Expected Class-IL: {expected_class_il:.2f}% (routing_acc × task_il_acc)")
    logger.info(f"Actual vs Expected gap: {eval_results['class_il_accuracy'] - expected_class_il:.2f}%")
    
    if abs(eval_results['class_il_accuracy'] - expected_class_il) > 5.0:
        logger.warning("⚠️  Large gap between actual and expected Class-IL! This suggests an evaluation bug.")
    
    logger.info("="*60)
    all_dg = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            for inputs, _ in loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[task_id](features)
                all_dg.append(dg_output.cpu().numpy())
                all_labels.append(np.full(dg_output.shape[0], task_id))
    all_dg = np.concatenate(all_dg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    dg_separation_diagnostic(all_dg, all_labels, logger=logger)


    # --- V7 GATING DEEP DIVE ---
    # analyze_gating_deep_dive(model, test_loaders, device, os.path.join(log_dir, 'analysis'))
    
    # --- V7 DG-GATED VISUALIZATIONS ---
    logger.info("\n" + "📊" * 60)
    logger.info("📊 GENERATING DG-GATED VISUALIZATIONS")
    logger.info("📊" * 60)
    
    # Create analysis directory if it doesn't exist
    analysis_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate DG-Gated visualizations
    try:
        # Collect data for visualizations
        all_gate_logits = []
        all_dg_outputs = []
        all_ca1_outputs = []
        all_task_labels = []
        
        model.eval()
        with torch.no_grad():
            for task_id, test_loader in enumerate(test_loaders):
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    
                    # Get outputs from all experts
                    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                    _, gate_logits, _ = model.forward(inputs)
                    
                    # Get DG and CA1 outputs from all experts
                    all_expert_dg = []
                    all_expert_ca1 = []
                    for expert_id in range(model.num_experts):
                        dg_output, ca1_output = model.hippocampal_experts[expert_id](features)
                        all_expert_dg.append(dg_output)
                        all_expert_ca1.append(ca1_output)
                    
                    # Use the correct expert's outputs for visualization
                    correct_expert = task_id
                    dg_output = all_expert_dg[correct_expert]
                    ca1_output = all_expert_ca1[correct_expert]
                    
                    all_gate_logits.append(gate_logits.cpu())
                    all_dg_outputs.append(dg_output.cpu())
                    all_ca1_outputs.append(ca1_output.cpu())
                    all_task_labels.append(torch.full((inputs.size(0),), task_id))
        
        # Concatenate all data
        gate_logits_cat = torch.cat(all_gate_logits, dim=0).numpy()
        dg_outputs_cat = torch.cat(all_dg_outputs, dim=0).numpy()
        ca1_outputs_cat = torch.cat(all_ca1_outputs, dim=0).numpy()
        task_labels_cat = torch.cat(all_task_labels, dim=0).numpy()
        
        # Calculate routing matrix
        predicted_experts = np.argmax(gate_logits_cat, axis=1)
        num_tasks = len(test_loaders)
        routing_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(len(task_labels_cat)):
            true_task = task_labels_cat[i]
            pred_expert = predicted_experts[i]
            routing_matrix[true_task, pred_expert] += 1
        
        # Normalize routing matrix
        row_sums = routing_matrix.sum(axis=1, keepdims=True)
        routing_matrix = routing_matrix / row_sums
        
        # Calculate expert utilization
        expert_utilization = np.bincount(predicted_experts, minlength=num_tasks) / len(predicted_experts)
        
        # Get DG prototypes
        dg_prototypes = model.dg_prototypes.detach().cpu().numpy() if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None else np.zeros((num_tasks, dg_outputs_cat.shape[1]))
        
        # Create DG-Gated visualizations
        create_dg_gated_visualizations(
            gate_logits_cat, dg_outputs_cat, ca1_outputs_cat, task_labels_cat,
            routing_matrix, expert_utilization, dg_prototypes, analysis_dir
        )
        
        # Create DG Deep-Dive analysis
        analyze_dg_deep_dive(model, test_loaders, device, analysis_dir)
        
        # Run DG Pattern Separation Diagnostic
        logger.info("\n" + "🔬" * 60)
        logger.info("🔬 RUNNING DG PATTERN SEPARATION DIAGNOSTIC")
        logger.info("🔬" * 60)
        
        
        # Store metrics for potential use in training decisions
        if hasattr(model, 'separation_metrics'):
            model.separation_metrics = separation_metrics
        else:
            model.separation_metrics = separation_metrics
        
        logger.info("✅ DG-Gated visualizations generated successfully!")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate DG-Gated visualizations: {e}")
        logger.warning("Continuing with training completion...")

    logger.info('✅ V7 DG-Gated training complete.')

    # === MANUAL METRIC CHECK: Debug Class-IL, Routing, Task-IL ===
    logger.info("\n=== MANUAL METRIC CHECK: Debugging Class-IL, Routing, Task-IL ===")
    model.eval()
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                B = inputs.size(0)
                # Get all experts' logits
                logits_all = []
                for expert_id in range(model.num_experts):
                    features = model.feature_extractor(inputs).view(B, -1)
                    _, ca1 = model.hippocampal_experts[expert_id](features)
                    logits_all.append(model.output_layers[expert_id](ca1))
                logits_all = torch.stack(logits_all, dim=1)  # [B, num_experts, num_classes]
                # Get gating probabilities
                _, gate_logits, _ = model.forward(inputs)
                gate_probs = torch.softmax(gate_logits, dim=1)  # [B, num_experts]
                chosen = gate_probs.argmax(dim=1)  # [B]
                # Routed logits
                routed_logits = logits_all[range(B), chosen]  # [B, num_classes]
                class_pred = routed_logits.argmax(dim=1)
                # Oracle (Task-IL)
                oracle_expert = torch.full((B,), task_id, dtype=torch.long, device=inputs.device)
                task_logits = logits_all[range(B), oracle_expert]  # [B, 20]
                logger.info(f"[Manual Check] Task {task_id}: task_logits shape={task_logits.shape}")
                task_logits_masked = task_logits  # Already [B, 20]
                task_labels_local = labels  # Already local (0-19)
                task_pred = task_logits_masked.argmax(dim=1)
                # Metrics
                routing_acc = (chosen == oracle_expert).float().mean().item()
                task_il_acc = (task_pred == task_labels_local).float().mean().item()

                # === ORACLE GATE DIAGNOSTIC ===
                features = model.feature_extractor(inputs).view(B, -1)
                dg = F.normalize(model.hippocampal_experts[task_id](features)[0], dim=1)
                protos = F.normalize(model.dg_prototypes, dim=1)
                cosines = dg @ protos.T  # [B, num_experts]
                oracle_gate = cosines.argmax(-1)
                routing_acc_oracle = (oracle_gate == task_id).float().mean().item()
                logger.info(f"[DIAG] Oracle routing acc: {routing_acc_oracle:.3f}")
                logger.info(f"[DIAG] Learned gate routing acc: {routing_acc:.3f}")

                # === GATING DISTRIBUTION PLOTS ===
                import matplotlib.pyplot as plt
                # Plot histogram of all gate logits
                plt.figure()
                plt.hist(gate_logits.cpu().numpy().flatten(), bins=30)
                plt.title('Gate Logits Distribution')
                plt.xlabel('Logit Value')
                plt.ylabel('Count')
                plt.show()
                # Plot histogram of top-1 gate probabilities
                top1_probs = gate_probs.max(dim=1).values.cpu().numpy()
                plt.figure()
                plt.hist(top1_probs, bins=30)
                plt.title('Top-1 Gate Probability Distribution')
                plt.xlabel('Top-1 Probability')
                plt.ylabel('Count')
                plt.show()
                break  # Only do one batch per task for diagnostics

    # After Phase 1 prototype computation and DG diagnostic:
    debug_routing_and_prototypes(model, test_loaders, logger=logger, num_samples=8)

def analyze_gating_deep_dive(model, test_loaders, device, save_dir):
    """
    Generates a comprehensive 10-plot analysis of the gating pathway to diagnose feature collapse.
    Adapted for DG-Gated model.
    """
    # Get the logger
    logger = logging.getLogger()
    
    logger.info("\n" + "="*80)
    logger.info("🔬 V7 DG-GATED GATING DEEP DIVE ANALYSIS")
    logger.info("="*80)
    model.eval()

    # --- 1. Extract features, labels, and gate outputs for all test data ---
    all_features = []
    all_gate_logits = []
    all_task_labels = []
    similarity_profiles = []  # <-- Ensure this is defined
    
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            for inputs, _ in loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                _, gate_logits, _ = model.forward(inputs)
                all_features.append(features.cpu())
                all_gate_logits.append(gate_logits.cpu())
                all_task_labels.append(torch.full((inputs.size(0),), task_id))
                # Optionally, collect similarity profiles here if needed
                # similarity_profiles.append(...)

    features_cat = torch.cat(all_features, dim=0).numpy()
    gate_logits_cat = torch.cat(all_gate_logits, dim=0).numpy()
    labels_cat = torch.cat(all_task_labels, dim=0).numpy()
    gate_probs_cat = F.softmax(torch.from_numpy(gate_logits_cat), dim=1).numpy()
    predicted_experts = np.argmax(gate_logits_cat, axis=1)

    # If similarity_profiles is needed for later plots, compute it here
    # For now, create a dummy similarity_profiles if not used elsewhere
    if not similarity_profiles:
        similarity_profiles = [np.zeros((1, model.num_experts)) for _ in range(model.num_experts)]

    # --- 2. Create the 10-plot figure ---
    fig, axes = plt.subplots(5, 2, figsize=(20, 45))
    fig.suptitle('V7 DG-Gated Gating Pathway Deep Dive Analysis', fontsize=24, y=0.95)
    
    # Plot 1: t-SNE of Raw Features
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    features_2d = tsne.fit_transform(features_cat)
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels_cat, palette='tab10', ax=axes[0, 0], legend='full')
    axes[0, 0].set_title('1. t-SNE of Raw Gating Features', fontsize=14)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Plot 2: Inter- vs. Intra-Task Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(features_cat)
    intra_task_sims, inter_task_sims = [], []
    for i in range(len(labels_cat)):
        for j in range(i + 1, len(labels_cat)):
            if labels_cat[i] == labels_cat[j]:
                intra_task_sims.append(similarities[i, j])
            else:
                inter_task_sims.append(similarities[i, j])
    sns.histplot(intra_task_sims, color='blue', label='Intra-Task (Same)', ax=axes[0, 1], stat='density', kde=True)
    sns.histplot(inter_task_sims, color='red', label='Inter-Task (Different)', ax=axes[0, 1], stat='density', kde=True)
    axes[0, 1].set_title('2. Feature Similarity Distribution', fontsize=14)
    axes[0, 1].legend()

    # Plot 3: Feature Similarity Heatmap
    num_tasks = len(test_loaders)
    sim_matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            features_i = features_cat[labels_cat == i]
            features_j = features_cat[labels_cat == j]
            if len(features_i) > 0 and len(features_j) > 0:
                sim_matrix[i, j] = cosine_similarity(features_i, features_j).mean()
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis", ax=axes[1, 0])
    axes[1, 0].set_title('3. Avg. Feature Similarity Between Tasks', fontsize=14)
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('Task')

    # Plot 4: Feature Norm Distribution
    feature_norms = np.linalg.norm(features_cat, axis=1)
    for i in range(num_tasks):
        task_mask = labels_cat == i
        if np.any(task_mask):
            sns.kdeplot(feature_norms[task_mask], label=f'Task {i}', ax=axes[1, 1], fill=True)
    axes[1, 1].set_title('4. Feature Vector L2 Norm Distribution', fontsize=14)
    axes[1, 1].legend()

    # Plot 5: Routing Confusion Matrix
    cm = confusion_matrix(labels_cat, predicted_experts)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2, 0])
    axes[2, 0].set_title('5. DG-Gated Routing Confusion Matrix', fontsize=14)
    axes[2, 0].set_xlabel('Predicted Expert')
    axes[2, 0].set_ylabel('True Task')

    # Plot 6: Expert Utilization
    utilization = np.bincount(predicted_experts, minlength=num_tasks) / len(predicted_experts)
    sns.barplot(x=list(range(num_tasks)), y=utilization, ax=axes[2, 1])
    axes[2, 1].axhline(1.0 / num_tasks, ls='--', color='red', label='Ideal Uniform')
    axes[2, 1].set_title('6. DG-Gated Expert Utilization', fontsize=14)
    axes[2, 1].set_xlabel('Expert ID')
    axes[2, 1].set_ylabel('Fraction of Times Chosen')
    axes[2, 1].legend()

    # Plot 7: Gating Confidence (Entropy)
    gate_entropy = entropy(gate_probs_cat, axis=1)
    sns.histplot(gate_entropy, ax=axes[3, 0], kde=True)
    axes[3, 0].axvline(np.log(num_tasks), ls='--', color='red', label='Max Entropy (Total Confusion)')
    axes[3, 0].set_title('7. DG-Gated Gating Confidence (Entropy)', fontsize=14)
    axes[3, 0].set_xlabel('Entropy of Gating Probabilities')
    axes[3, 0].legend()

    # Plot 8: Gating Logits for Correct vs. Incorrect Experts
    correct_logits, incorrect_logits = [], []
    for i in range(len(labels_cat)):
        true_task = labels_cat[i]
        for expert_id in range(num_tasks):
            if expert_id == true_task:
                correct_logits.append(gate_logits_cat[i, expert_id])
            else:
                incorrect_logits.append(gate_logits_cat[i, expert_id])
    sns.boxplot(data=[correct_logits, incorrect_logits], ax=axes[3, 1])
    axes[3, 1].set_xticklabels(['Correct Expert', 'Incorrect Experts'])
    axes[3, 1].set_title('8. DG-Gated Logit Value Distribution', fontsize=14)
    axes[3, 1].set_ylabel('Logit Value')
    
    # Plot 9: "Oracle" Router Performance
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features_cat, labels_cat, test_size=0.3, random_state=42)
    oracle_clf = LogisticRegression(max_iter=1000, solver='liblinear')
    oracle_clf.fit(X_train, y_train)
    oracle_acc = oracle_clf.score(X_test, y_test)
    axes[4, 0].bar(['Actual DG-Gating', 'Oracle Gating'], [np.mean(predicted_experts == labels_cat), oracle_acc])
    axes[4, 0].set_title('9. Oracle vs. Actual DG-Gating Accuracy', fontsize=14)
    axes[4, 0].set_ylabel('Accuracy')
    axes[4, 0].set_ylim(0, 1)

    # Plot 10: Correlation of Feature Norm and Routing Accuracy
    task_avg_norm = [np.linalg.norm(features_cat[labels_cat == i], axis=1).mean() for i in range(num_tasks)]
    task_routing_acc = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(num_tasks)]
    sns.regplot(x=task_avg_norm, y=task_routing_acc, ax=axes[4, 1])
    axes[4, 1].set_title('10. Feature Norm vs. DG-Gating Accuracy', fontsize=14)
    axes[4, 1].set_xlabel('Average Feature L2 Norm for Task')
    axes[4, 1].set_ylabel('Per-Task DG-Gating Accuracy')

    # --- 3. Save the figure ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'dg_gating_deep_dive.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"✅ DG-Gated Gating Deep Dive analysis saved to: {save_path}")

# Use imported create_dg_gated_visualizations from l.py instead of duplicate

# Use imported create_dg_deep_dive_visualizations from l.py instead of duplicate
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Gating Decision Profile - Box Plot
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
            sparse_activation_outputs[expert_id].append(output.detach().cpu())
        return hook
    
    # Register hooks on SparseActivation layers
    hooks = []
    for expert_id in range(model.num_experts):
        # Find the SparseActivation layer in the DG expert
        # Commented out due to potential attribute access issues
        # for name, module in model.hippocampal_experts[expert_id].dg.named_modules():
        #     if hasattr(module, 'sparsity') or 'sparse' in name.lower():
        #         hook = module.register_forward_hook(get_sparse_activation_hook(expert_id))
        #         hooks.append(hook)
        #         break
        pass  # Skip hook registration for now
    
    # --- Data Collection ---
    # Store DG outputs for each task's data, processed by the correct expert
    dg_outputs_per_task = [[] for _ in range(model.num_experts)]
    # Store similarity scores of each task's data against all prototypes
    similarity_profiles = [[] for _ in range(model.num_experts)]

    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in tqdm(test_loader, desc=f"Deep Analyzing Task {task_id}"):
                inputs = inputs.to(device)
                
                # Get DG outputs from all experts
                all_dg_outputs = []
                for expert_id in range(model.num_experts):
                    features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                    dg_output, _ = model.hippocampal_experts[expert_id](features)
                    all_dg_outputs.append(dg_output)
                    
                    # Store for the correct expert
                    if expert_id == task_id:
                        dg_outputs_per_task[expert_id].append(dg_output.cpu())
                
                # Calculate similarity to all prototypes
                if hasattr(model, 'dg_prototypes') and model.dg_prototypes is not None:
                    prototypes = model.dg_prototypes.detach()
                    similarities = F.cosine_similarity(
                        torch.stack(all_dg_outputs, dim=1),  # (batch, experts, dg_dim)
                        prototypes.unsqueeze(0),  # (1, experts, dg_dim)
                        dim=2
                    )  # (batch, experts)
                    similarity_profiles[task_id].append(similarities.cpu())

    # --- Calculate Statistics ---
    # 1. TRUE Sparsity (from SparseActivation layer)
    expert_sparsity = []
    for expert_id in range(model.num_experts):
        if sparse_activation_outputs[expert_id]:
            all_sparse = torch.cat(sparse_activation_outputs[expert_id], dim=0)
            sparsity = (all_sparse != 0).float().mean().item()
            expert_sparsity.append(sparsity)
        else:
            expert_sparsity.append(0.0)
    
    # 2. Final DG Output Sparsity
    final_dg_sparsity = []
    for expert_id in range(model.num_experts):
        if dg_outputs_per_task[expert_id]:
            all_dg = torch.cat(dg_outputs_per_task[expert_id], dim=0)
            sparsity = (all_dg != 0).float().mean().item()
            final_dg_sparsity.append(sparsity)
        else:
            final_dg_sparsity.append(0.0)
    
    # 3. Pattern Separation Analysis
    separation_data = []
    for expert_id in range(model.num_experts):
        if dg_outputs_per_task[expert_id]:
            dg_outputs = torch.cat(dg_outputs_per_task[expert_id], dim=0)
            
            # Convert to binary patterns (active/inactive)
            binary_patterns = (dg_outputs != 0).float()
            
            # Calculate Jaccard similarity between patterns
            n_samples = binary_patterns.shape[0]
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    pattern_i = binary_patterns[i]
                    pattern_j = binary_patterns[j]
                    
                    intersection = (pattern_i * pattern_j).sum()
                    union = (pattern_i + pattern_j).clamp(0, 1).sum()
                    
                    if union > 0:
                        jaccard = intersection / union
                        separation_data.append({
                            'expert': f'Expert {expert_id}',
                            'similarity': jaccard.item(),
                            'type': 'Intra-Task'
                        })
    
    # 4. Prepare similarity profiles for visualization
    if similarity_profiles[0]:  # Check if we have data
        similarity_profiles = [torch.cat(profs, dim=0).numpy() for profs in similarity_profiles]
    else:
        similarity_profiles = [np.zeros((1, model.num_experts)) for _ in range(model.num_experts)]
    
    # --- Create Visualizations ---
    create_dg_deep_dive_visualizations(expert_sparsity, final_dg_sparsity, separation_data, similarity_profiles, save_dir)
    
    # --- Cleanup ---
    for hook in hooks:
        hook.remove()
    
    logging.info(f"✅ DG Deep-Dive Analysis completed!")

def diagnose_prototype_status(model):
    """
    Diagnostic function to check the status of DG prototypes and identify gating issues.
    """
    logging.info("\n" + "🔍" * 50)
    logging.info("🔍 DG PROTOTYPE STATUS DIAGNOSTIC")
    logging.info("🔍" * 50)
    
    # Check if prototypes exist
    if not hasattr(model, 'dg_prototypes'):
        logging.error("❌ CRITICAL: Model has no 'dg_prototypes' attribute!")
        logging.error("   This means prototypes were never initialized.")
        return False
    
    if model.dg_prototypes is None:
        logging.error("❌ CRITICAL: 'dg_prototypes' is None!")
        logging.error("   Prototypes were initialized but set to None.")
        return False
    
    # Check prototype shape and basic properties
    proto_shape = model.dg_prototypes.shape
    logging.info(f"📊 Prototype shape: {proto_shape}")
    
    if len(proto_shape) != 2:
        logging.error(f"❌ CRITICAL: Expected 2D prototypes, got shape {proto_shape}")
        return False
    
    num_experts, dg_dim = proto_shape
    logging.info(f"📊 Number of experts: {num_experts}")
    logging.info(f"📊 DG dimension: {dg_dim}")
    
    # Check for zero prototypes
    zero_protos = (model.dg_prototypes == 0).all(dim=1)
    num_zero_protos = zero_protos.sum().item()
    logging.info(f"📊 Zero prototypes: {num_zero_protos}/{num_experts}")
    
    if num_zero_protos == num_experts:
        logging.error("❌ CRITICAL: ALL prototypes are zero!")
        logging.error("   This will cause uniform gating probabilities.")
        return False
    
    # Check prototype variance
    proto_std = model.dg_prototypes.std().item()
    logging.info(f"📊 Overall prototype std: {proto_std:.6f}")
    
    if proto_std < 1e-6:
        logging.error("❌ CRITICAL: Prototype standard deviation is too low!")
        logging.error(f"   Std: {proto_std:.6f} < 1e-6 threshold")
        logging.error("   This will trigger the fallback to uniform gating.")
        return False
    
    # Check individual expert prototypes
    logging.info("\n" + "📊" * 30)
    logging.info("📊 INDIVIDUAL EXPERT PROTOTYPE ANALYSIS")
    logging.info("📊" * 30)
    
    for expert_id in range(num_experts):
        proto = model.dg_prototypes[expert_id]
        proto_norm = torch.norm(proto).item()
        proto_std_expert = proto.std().item()
        proto_mean = proto.mean().item()
        
        logging.info(f"Expert {expert_id}: norm={proto_norm:.4f}, "
                    f"std={proto_std_expert:.6f}, mean={proto_mean:.6f}")
        
        if proto_norm < 1e-6:
            logging.warning(f"⚠️ Expert {expert_id}: Prototype has near-zero norm!")
        if proto_std_expert < 1e-6:
            logging.warning(f"⚠️ Expert {expert_id}: Prototype has near-zero std!")
    
    # Check prototype distinctness
    if num_experts > 1:
        proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1)
        proto_sim = torch.mm(proto_norm, proto_norm.T)
        
        # Get off-diagonal similarities
        mask = ~torch.eye(num_experts, dtype=bool, device=proto_sim.device)
        off_diag_sims = proto_sim[mask]
        
        max_sim = off_diag_sims.max().item()
        mean_sim = off_diag_sims.mean().item()
        min_sim = off_diag_sims.min().item()
        
        logging.info(f"\n📊 Prototype Similarity Analysis:")
        logging.info(f"   Max off-diagonal similarity: {max_sim:.4f}")
        logging.info(f"   Mean off-diagonal similarity: {mean_sim:.4f}")
        logging.info(f"   Min off-diagonal similarity: {min_sim:.4f}")
        
        if max_sim > 0.9:
            logging.warning("⚠️ WARNING: Very high prototype similarity detected!")
            logging.warning("   This indicates prototype collapse.")
        elif max_sim > 0.7:
            logging.warning("⚠️ WARNING: High prototype similarity detected!")
            logging.warning("   Prototypes may not be sufficiently distinct.")
    
    # Check gating temperature
    if hasattr(model, 'gating_temperature'):
        temp = model.gating_temperature
        logging.info(f"\n📊 Gating temperature: {temp:.4f}")
        
        if temp > 10.0:
            logging.warning("⚠️ WARNING: Very high gating temperature!")
            logging.warning("   This will make gating probabilities very uniform.")
        elif temp < 0.1:
            logging.warning("⚠️ WARNING: Very low gating temperature!")
            logging.warning("   This will make gating very sharp (one-hot).")
    else:
        logging.warning("⚠️ WARNING: No gating_temperature attribute found!")
    
    # Check if prototypes were computed
    if hasattr(model, 'prototypes_computed'):
        logging.info(f"📊 Prototypes computed flag: {model.prototypes_computed}")
    else:
        logging.warning("⚠️ WARNING: No 'prototypes_computed' flag found!")
    
    logging.info("🔍" * 50)
    return True

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
    
    # First, check prototype status
    diagnose_prototype_status(model)
    
    model.eval()
    
    # Collect DG outputs and true expert labels
    all_dg_outputs = []
    all_true_experts = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            for inputs, _ in tqdm(test_loader, desc=f"Collecting DG outputs for Task {task_id}"):
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                
                # Get DG outputs from all experts
                batch_dg_outputs = []
                for expert_id in range(model.num_experts):
                    dg_output, _ = model.hippocampal_experts[expert_id](features)
                    batch_dg_outputs.append(dg_output)
                
                # Stack outputs from all experts: [batch_size, num_experts, dg_dim]
                batch_dg_outputs = torch.stack(batch_dg_outputs, dim=1)
                
                all_dg_outputs.append(batch_dg_outputs.cpu())
                all_true_experts.append(torch.full((inputs.size(0),), task_id))
    
    # Concatenate all batches
    dg_outputs = torch.cat(all_dg_outputs, dim=0)  # [N, num_experts, dg_dim]
    true_expert = torch.cat(all_true_experts, dim=0)  # [N]
    
    N, num_experts, dg_dim = dg_outputs.shape
    
    logging.info(f"📊 Collected {N} samples across {num_experts} experts")
    logging.info(f"📊 DG embedding dimension: {dg_dim}")
    
    # 1. Flatten and assign each sample to the highest-scoring expert
    normed = torch.nn.functional.normalize(dg_outputs.view(N, -1), p=2, dim=1)
    embeddings = normed.cpu().numpy()
    labels = true_expert.cpu().numpy()
    
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
    
    # 3. Compute intra/inter distances manually
    def pairwise_dists(x):
        sq = np.sum(x**2, axis=1, keepdims=True)
        return np.sqrt(sq + sq.T - 2*x.dot(x.T))
    
    try:
        dists = pairwise_dists(embeddings)
        
        intra = []
        inter = []
        for i in range(num_experts):
            mask_i = (labels == i)
            if np.sum(mask_i) > 1:  # Need at least 2 samples per cluster
                intra += list(dists[mask_i][:, mask_i].ravel())
                inter += list(dists[mask_i][:, ~mask_i].ravel())
        
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
            elif separation_ratio > 1.2:
                logging.info("⚠️ WEAK: Limited DG pattern separation")
            else:
                logging.warning("❌ POOR: Very weak DG pattern separation - possible collapse!")
        
    except Exception as e:
        mean_intra = mean_inter = float('nan')
        logging.warning(f"⚠️ Failed to compute distance metrics: {e}")
    
    # 4. Additional diagnostics: Expert-wise analysis
    logging.info("\n" + "📊" * 40)
    logging.info("📊 EXPERT-WISE DG SEPARATION ANALYSIS")
    logging.info("📊" * 40)
    
    for expert_id in range(num_experts):
        expert_mask = (labels == expert_id)
        n_samples = np.sum(expert_mask)
        
        if n_samples > 0:
            expert_embeddings = embeddings[expert_mask]
            
            # Compute expert-specific metrics
            if n_samples > 1:
                # Intra-cluster distance for this expert
                expert_dists = pairwise_dists(expert_embeddings)
                expert_intra = np.mean(expert_dists[np.triu_indices_from(expert_dists, k=1)])
                
                # Distance to other experts' centroids
                other_centroids = []
                for other_id in range(num_experts):
                    if other_id != expert_id:
                        other_mask = (labels == other_id)
                        if np.sum(other_mask) > 0:
                            other_centroid = np.mean(embeddings[other_mask], axis=0)
                            other_centroids.append(other_centroid)
                
                if other_centroids:
                    other_centroids = np.array(other_centroids)
                    expert_centroid = np.mean(expert_embeddings, axis=0)
                    expert_inter = np.mean(np.linalg.norm(other_centroids - expert_centroid, axis=1))
                    
                    logging.info(f"Expert {expert_id}: {n_samples} samples, "
                               f"Intra: {expert_intra:.4f}, Inter: {expert_inter:.4f}, "
                               f"Ratio: {expert_inter/expert_intra:.4f}")
                else:
                    logging.info(f"Expert {expert_id}: {n_samples} samples, Intra: {expert_intra:.4f}")
            else:
                logging.info(f"Expert {expert_id}: {n_samples} sample (insufficient for analysis)")
        else:
            logging.warning(f"⚠️ Expert {expert_id}: No samples!")
    
    # 5. Create visualization if save_dir is provided
    if save_dir is not None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('🔬 DG Pattern Separation Diagnostic', fontsize=16, fontweight='bold')
            
            # Plot 1: Silhouette Score Distribution
            if not np.isnan(sil_score):
                axes[0,0].bar(['Silhouette Score'], [sil_score], color='skyblue')
                axes[0,0].set_ylim(-1, 1)
                axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[0,0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Good')
                axes[0,0].set_title('Silhouette Score')
                axes[0,0].set_ylabel('Score')
                axes[0,0].legend()
                axes[0,0].text(0, sil_score + 0.05, f'{sil_score:.3f}', ha='center', va='bottom')
            
            # Plot 2: Davies-Bouldin Index
            if not np.isnan(db_index):
                axes[0,1].bar(['Davies-Bouldin'], [db_index], color='lightcoral')
                axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
                axes[0,1].set_title('Davies-Bouldin Index (Lower is Better)')
                axes[0,1].set_ylabel('Index')
                axes[0,1].legend()
                axes[0,1].text(0, db_index + 0.1, f'{db_index:.3f}', ha='center', va='bottom')
            
            # Plot 3: Distance Distribution
            if not np.isnan(mean_intra) and not np.isnan(mean_inter):
                axes[1,0].bar(['Intra-Cluster', 'Inter-Cluster'], [mean_intra, mean_inter], 
                             color=['lightgreen', 'orange'])
                axes[1,0].set_title('Mean Distances')
                axes[1,0].set_ylabel('Distance')
                axes[1,0].text(0, mean_intra + 0.01, f'{mean_intra:.3f}', ha='center', va='bottom')
                axes[1,0].text(1, mean_inter + 0.01, f'{mean_inter:.3f}', ha='center', va='bottom')
            
            # Plot 4: t-SNE of DG embeddings colored by expert
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, N//4))
                embeddings_2d = tsne.fit_transform(embeddings)
                
                scatter = axes[1,1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                          c=labels, cmap='tab10', alpha=0.6, s=10)
                axes[1,1].set_title('DG Embeddings (t-SNE)')
                axes[1,1].set_xlabel('t-SNE 1')
                axes[1,1].set_ylabel('t-SNE 2')
                
                # Add legend
                legend1 = axes[1,1].legend(*scatter.legend_elements(),
                                         title="Expert", loc="upper right")
                axes[1,1].add_artist(legend1)
                
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f't-SNE failed:\n{e}', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('DG Embeddings (t-SNE) - Failed')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'DG_Pattern_Separation_Diagnostic.png'), 
                       dpi=200, bbox_inches='tight')
            plt.close()
            
            logging.info(f"📊 DG Pattern Separation Diagnostic plot saved!")
            
        except Exception as e:
            logging.warning(f"⚠️ Failed to create diagnostic plot: {e}")
    
    # 6. Summary and recommendations
    logging.info("\n" + "💡" * 40)
    logging.info("💡 DG PATTERN SEPARATION SUMMARY")
    logging.info("💡" * 40)
    
    if not np.isnan(sil_score):
        if sil_score > 0.5:
            logging.info("✅ Silhouette Score: EXCELLENT separation")
        elif sil_score > 0.2:
            logging.info("✅ Silhouette Score: GOOD separation")
        elif sil_score > 0:
            logging.info("⚠️ Silhouette Score: WEAK separation")
        else:
            logging.warning("❌ Silhouette Score: POOR separation (negative)")
    
    if not np.isnan(db_index):
        if db_index < 0.5:
            logging.info("✅ Davies-Bouldin: EXCELLENT cluster separation")
        elif db_index < 1.0:
            logging.info("✅ Davies-Bouldin: GOOD cluster separation")
        elif db_index < 2.0:
            logging.info("⚠️ Davies-Bouldin: MODERATE cluster separation")
        else:
            logging.warning("❌ Davies-Bouldin: POOR cluster separation")
    
    if not np.isnan(mean_intra) and not np.isnan(mean_inter):
        separation_ratio = mean_inter / mean_intra
        if separation_ratio > 2.0:
            logging.info("🎉 Distance Ratio: EXCELLENT pattern separation!")
        elif separation_ratio > 1.5:
            logging.info("✅ Distance Ratio: GOOD pattern separation")
        elif separation_ratio > 1.2:
            logging.info("⚠️ Distance Ratio: WEAK pattern separation")
        else:
            logging.warning("❌ Distance Ratio: POOR pattern separation - possible collapse!")
    
    return {
        'silhouette_score': sil_score,
        'davies_bouldin_index': db_index,
        'mean_intra_distance': mean_intra,
        'mean_inter_distance': mean_inter,
        'separation_ratio': mean_inter / mean_intra if not np.isnan(mean_intra) and not np.isnan(mean_inter) else float('nan')
    }

def compute_dg_prototypes(model, train_loaders, device):
    """
    Computes the prototype DG pattern for each expert by averaging the DG
    output over all training samples for that expert's task.
    EXACT COPY from l.py
    """
    logging.info("🧠 Computing all DG Prototypes post-Phase 1...")
    model.eval()
    with torch.no_grad():
        for task_id, train_loader in enumerate(tqdm(train_loaders, desc="Computing Prototypes")):
            all_dg_outputs = []
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                dg_output, _ = model.hippocampal_experts[task_id](features)
                all_dg_outputs.append(dg_output.cpu())
            
            # Average all DG outputs for this task
            with torch.no_grad():
                model.dg_prototypes[task_id] = torch.cat(all_dg_outputs, dim=0).mean(dim=0)
    
    model.prototypes_computed = True
    logging.info("✅ All expert prototypes computed and stored.")

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
        logger.info(f"  Silhouette Score: {sil:.3f} (≳ 0.5 is good)")
        logger.info(f"  Davies–Bouldin Index: {db:.3f} (< 1 is good)")
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

# === DIAGNOSTICS SECTION ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def plot_prototype_stats(prototypes, phase, save_dir, logger=None):
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

# === INTEGRATE INTO MAIN ===
# After Phase 1 prototype computation and DG diagnostic:
    # Save prototype stats
    plot_prototype_stats(model.dg_prototypes.cpu().numpy(), phase='phase1', save_dir=log_dir, logger=logger)
    # If you have gate_logits from evaluation, plot gating confidence
    # plot_gating_confidence(gate_logits, phase='phase1', save_dir=log_dir, logger=logger)
    # If you have cluster assignments, plot cluster purity
    # plot_cluster_purity(all_labels, predicted_experts, phase='phase1', save_dir=log_dir, logger=logger)
    # If you have per-class accuracy, plot it
    # plot_per_class_accuracy(all_true, all_pred, n_classes, phase='phase1', save_dir=log_dir, logger=logger)
# Repeat after Phase 2 as well.

def log_and_plot_prototype_drift(proto_before, proto_after, phase_from, phase_to, save_dir, logger=None):
    """
    Logs and plots prototype drift between two phases.
    """
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

def debug_routing_and_prototypes(model, test_loaders, logger=None, num_samples=8):
    logger = logger or logging.getLogger()
    logger.info("\n=== DEBUG: ROUTING AND PROTOTYPE MATRICES ===")
    # 1. Prototype matrix
    protos = model.dg_prototypes.detach().cpu().numpy()
    logger.info(f"Prototype matrix shape: {protos.shape}")
    logger.info(f"Prototype stats: min={protos.min():.4f}, max={protos.max():.4f}, mean={protos.mean():.4f}, std={protos.std():.4f}")
    logger.info(f"First 2 prototypes:\n{protos[:2]}")
    # 2. Pairwise cosine similarity matrix
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    protos_norm = protos / (norms + 1e-8)
    sim_matrix = np.dot(protos_norm, protos_norm.T)
    off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    logger.info(f"Prototype cosine similarity: min={off_diag.min():.4f}, max={off_diag.max():.4f}, mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    logger.info(f"Sample of similarity matrix:\n{sim_matrix[:4,:4]}")
    # 3. Routing debug for a small batch
    model.eval()
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            for inputs, labels in loader:
                inputs, labels = inputs[:num_samples].to(model.dg_prototypes.device), labels[:num_samples].to(model.dg_prototypes.device)
                features = model.feature_extractor(inputs).view(inputs.size(0), -1)
                # DG outputs for all experts
                all_dg = []
                for expert_id in range(model.num_experts):
                    dg_out, _ = model.hippocampal_experts[expert_id](features)
                    all_dg.append(dg_out)
                all_dg = torch.stack(all_dg, dim=1)  # [B, num_experts, dg_dim]
                # Cosine similarities
                all_dg_norm = F.normalize(all_dg, p=2, dim=2)
                proto_norm = F.normalize(model.dg_prototypes, p=2, dim=1).to(inputs.device)
                sims = torch.einsum('bne,ne->bn', all_dg_norm, proto_norm)
                logger.info(f"Cosine similarity matrix (batch):\n{sims.cpu().numpy()}")
                # Gating logits and softmax
                _, gate_logits, _ = model.forward(inputs)
                gate_probs = torch.softmax(gate_logits, dim=1)
                chosen = gate_probs.argmax(dim=1)
                logger.info(f"Gating logits (batch):\n{gate_logits.cpu().numpy()}")
                logger.info(f"Gating softmax (batch):\n{gate_probs.cpu().numpy()}")
                logger.info(f"Predicted experts: {chosen.cpu().numpy()}")
                logger.info(f"Oracle experts (task_id): {np.full(inputs.size(0), task_id)}")
                routing_acc = (chosen.cpu().numpy() == task_id).mean()
                logger.info(f"Routing accuracy for this batch: {routing_acc:.3f}")
                logger.info(f"Labels: {labels.cpu().numpy()}")
                return  # Only do one batch

if __name__ == '__main__':
    main() 