import torch
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

def evaluate_on_all_tasks(model, test_loaders, num_tasks, classes_per_task, device):
    """
    Comprehensive final performance evaluation for a continual learning model.

    Calculates:
    1. Task-IL Accuracy: Performance on each task given an oracle task ID.
    2. Class-IL Accuracy: Performance across all classes without task ID.
    3. Routing Accuracy: How well a model routes inputs to the correct expert (if applicable).

    Args:
        model (torch.nn.Module): The model to evaluate. It's assumed to be in `eval` mode.
                                 The model MUST have a method `forward_eval(inputs)` which
                                 returns a tuple `(all_class_logits, gate_logits)`.
                                 `gate_logits` can be `None` if the model has no router.
        test_loaders (list): A list of DataLoaders for each task's test set.
        num_tasks (int): The total number of tasks.
        classes_per_task (int): The number of classes in each task.
        device (torch.device): The device to run evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing all calculated performance metrics.
    """
    logger.info("\n" + "="*80)
    logger.info("PERFORMING FINAL EVALUATION ON ALL TASKS")
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

                # Use the model's evaluation-specific forward pass
                # For Task-IL, we still get all logits but slice them based on the oracle task_id
                all_outputs, _ = model.forward_eval(inputs)

                # Slice the outputs to get the logits for the current task
                start_idx = task_id * classes_per_task
                end_idx = start_idx + classes_per_task
                task_outputs = all_outputs[:, start_idx:end_idx]

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
    has_gating_info = False

    logger.info("\nEvaluating Class-IL & Routing Performance (no Oracle)...")
    with torch.no_grad():
        for task_id in range(num_tasks):
            test_loader = test_loaders[task_id]
            for inputs, local_labels in tqdm(test_loader, desc=f"Testing Task {task_id+1}", leave=False):
                inputs, local_labels = inputs.to(device), local_labels.to(device)
                
                # Convert local labels (e.g., 0-19) to global labels
                global_labels = local_labels + task_id * classes_per_task

                # Use the single evaluation forward pass
                outputs, gate_logits = model.forward_eval(inputs)

                # 1. Class-IL Accuracy
                _, predicted_cls = torch.max(outputs, 1)
                class_il_correct += (predicted_cls == global_labels).sum().item()
                class_il_total += global_labels.size(0)

                # 2. Routing Accuracy (if applicable)
                if gate_logits is not None:
                    has_gating_info = True
                    _, predicted_expert = torch.max(gate_logits, 1)
                    routing_correct += (predicted_expert == task_id).sum().item()
                    routing_total += inputs.size(0)

                    # DEBUG: Log a few samples from each task's evaluation
                    if local_labels.size(0) > 4 and np.random.rand() < 0.01: # Log ~1% of batches
                        logger.debug(f"\n--- EVAL DEBUG: Task {task_id+1} ---")
                        for i in range(4): # Log first 4 samples
                            true_global = global_labels[i].item()
                            pred_global = predicted_cls[i].item()
                            is_correct = "✅" if true_global == pred_global else "❌"
                            true_expert = task_id
                            pred_expert = predicted_expert[i].item()
                            gate_probs = F.softmax(gate_logits[i], dim=0)
                            
                            logger.debug(f"  Sample {i}: {is_correct} Pred: {pred_global}, True: {true_global} | "
                                         f"Gate Pred: {pred_expert}, True: {true_expert} | "
                                         f"Gate Probs: {np.round(gate_probs.cpu().numpy(), 2)}")
                        logger.debug("--- EVAL DEBUG END ---\n")
                    
                    # DEBUG: Log routing statistics for this task
                    task_routing_correct = (predicted_expert == task_id).sum().item()
                    task_routing_total = inputs.size(0)
                    task_routing_acc = (task_routing_correct / task_routing_total) * 100 if task_routing_total > 0 else 0.0
                    logger.debug(f"Task {task_id+1} routing: {task_routing_correct}/{task_routing_total} = {task_routing_acc:.1f}%")

    class_il_accuracy = (class_il_correct / class_il_total) * 100 if class_il_total > 0 else 0.0
    routing_accuracy = (routing_correct / routing_total) * 100 if routing_total > 0 else 0.0

    # --- Summary ---
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"  - Expert Accuracies (Task-IL): {[f'{acc:.1f}%' for acc in expert_accuracies]}")
    logger.info(f"  - Average Task-IL Accuracy: {task_il_accuracy:.2f}%")
    logger.info(f"  - Class-IL Accuracy: {class_il_accuracy:.2f}%")
    if has_gating_info:
        logger.info(f"  - Routing Accuracy: {routing_accuracy:.2f}%")
    logger.info(f"  - Forgetting Gap (Task-IL vs Class-IL): {task_il_accuracy - class_il_accuracy:.2f}%")

    results = {
        'expert_accuracies': expert_accuracies,
        'task_il_accuracy': task_il_accuracy,
        'class_il_accuracy': class_il_accuracy,
        'task_class_gap': task_il_accuracy - class_il_accuracy
    }
    if has_gating_info:
        results['routing_accuracy'] = routing_accuracy

    return results
