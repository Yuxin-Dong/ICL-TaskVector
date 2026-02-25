# This must be first
import argparse
import copy

import numpy as np
import torch
from torch import nn
from dotenv import load_dotenv
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE

from core.attention_saliency import LlamaAttentionerManager

load_dotenv(".env")

import sys
import os
import pickle
import time
from typing import Optional
import matplotlib.pyplot as plt

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits, tokenize_datasets
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, run_multi_task_vector, get_task_hiddens
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: int, num_train: int, num_valid: int, num_tvs: int, multiple_dataset: bool) -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}_{num_train}_{num_valid}_{num_tvs}{'_m' if multiple_dataset else ''}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, experiment_id: int, num_train: int, num_valid: int, num_tvs: int, multiple_dataset: bool) -> None:
    seed_everything(experiment_id)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=50, num_train=num_valid, num_valid=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_train=num_train, num_valid=num_valid)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_train=num_train, num_valid=num_valid)
    icl_predictions = run_icl(model, tokenizer, task, test_datasets)

    test_multi_datasets = [test_datasets]
    dev_multi_datasets = [dev_datasets]
    for _ in range(num_tvs - 1):
        test_multi_datasets.append(task.create_datasets(num_datasets=num_test_datasets, num_train=num_train, num_valid=num_valid, same_test=test_datasets))
        dev_multi_datasets.append(task.create_datasets(num_datasets=num_dev_datasets, num_train=num_train, num_valid=num_valid, same_test=dev_datasets))

    multi_tv_predictions, multi_tv_dev_accuracy_by_layer, multi_task_hiddens = run_multi_task_vector(
        model,
        tokenizer,
        task,
        test_multi_datasets,
        dev_multi_datasets,
        multiple_dataset
    )

    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )

    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    accuracies["multi_tv_dev_by_layer"] = multi_tv_dev_accuracy_by_layer
    accuracies["multi_tv"] = calculate_accuracy_on_datasets(task, multi_tv_predictions, test_datasets)

    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens[0].mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, tv_ordered_tokens_by_layer


def run_main_experiment(
    model_type: str,
    model_variant: str,
    num_train: int,
    num_valid: int,
    num_tvs: int,
    multiple_dataset: bool,
    experiment_id: int,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id, num_train, num_valid, num_tvs, multiple_dataset)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # if os.path.exists(results_file):
    #     with open(results_file, "rb") as f:
    #         results = pickle.load(f)
    # else:
    results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, experiment_id, num_train, num_valid, num_tvs, multiple_dataset)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        print(f"Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")
        print()
        print(f"Multi Task Vector Accuracy: {accuracies['multi_tv']:.2f}")
        print(f"Multi Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["multi_tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")
        print()
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_train": num_train,
            "num_valid": num_valid,
            "num_tvs": num_tvs,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "multi_tv_accuracy": accuracies["multi_tv"],
            "multi_tv_dev_accruacy_by_layer": accuracies["multi_tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def analyze_task_vector_weights(
    model_type: str,
    model_variant: str,
    num_train: int,
) -> None:
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    n_heads = model.config.num_attention_heads
    seed_everything(42)

    task = get_task_by_name(tokenizer=tokenizer, task_name="algorithmic_to_upper")
    datasets = [task.create_datasets(num_datasets=100, num_examples=num_train)]
    for i in range(num_train):
        new_datasets = copy.deepcopy(datasets[0])
        for dataset in new_datasets:
            new_input = task.sample_inputs(1, dataset.train_inputs + [dataset.test_input])[0]

            new_output = task.calc_test_output(new_input)
            dataset.train_outputs[i] = new_output

        datasets.append(new_datasets)

    task_hiddens = get_task_hiddens(model, tokenizer, task, datasets)[:, :, 13, :]
    n, k, d = task_hiddens.shape
    d_head = d // n_heads

    for i in range(n_heads):
        task_diff = (task_hiddens[:, 1:, i*d_head:(i+1)*d_head] - task_hiddens[:, 0, i*d_head:(i+1)*d_head].unsqueeze(1)).norm(dim=2)
        task_weights = task_diff.mean(dim=0)
        task_weights = task_weights.cpu().numpy()

    pickle.dump(task_weights, open('task_weights.pkl', 'wb'))


def plot_bijection_results(args):
    results_file = get_results_file_path(args.model_type, args.model_variant, "", args.num_train, 0, 1, False)
    with open(results_file, "rb") as f:
        acc = pickle.load(f)
        for i, task_name in enumerate(TASKS_TO_EVALUATE):
            print(f'{task_name} {acc[task_name]["icl_accuracy"]:.2f} & {acc[task_name]["tv_accuracy"]:.2f}')


def collect_multi_tv_results(model_type, model_variant, num_train):
    print(model_type, model_variant)
    results = np.zeros([len(TASKS_TO_EVALUATE), 5, 21])
    for num_tvs in range(1, 6, 1):
        num_te = num_tvs - 1
        for seed in range(5):
            results_file = get_results_file_path(model_type, model_variant, f"{seed}", num_train, num_te, num_tvs, False)
            with open(results_file, "rb") as f:
                acc = pickle.load(f)
                for i, task_name in enumerate(TASKS_TO_EVALUATE):
                    results[i, seed, 4 * num_tvs - 3] = acc[task_name]["baseline_accuracy"]
                    results[i, seed, 4 * num_tvs - 2] = np.max(list(acc[task_name]["tv_dev_accruacy_by_layer"].values()))
                    results[i, seed, 4 * num_tvs - 1] = np.max(list(acc[task_name]["multi_tv_dev_accruacy_by_layer"].values()))

            results_file = get_results_file_path(model_type, model_variant, f"{seed}", num_train, num_te, num_tvs, num_tvs > 1)
            with open(results_file, "rb") as f:
                acc = pickle.load(f)
                for i, task_name in enumerate(TASKS_TO_EVALUATE):
                    results[i, seed, 4 * num_tvs] = np.max(list(acc[task_name]["multi_tv_dev_accruacy_by_layer"].values()))

            if num_tvs == 1:
                for i, task_name in enumerate(TASKS_TO_EVALUATE):
                    results[i, seed, 0] = acc[task_name]["icl_accuracy"]

    # acc_mean = results.mean(axis=(0,1))
    # print(acc_mean)
    # print(acc_mean[1::4].mean(), acc_mean[2::4].mean(), acc_mean[3::4].mean(), acc_mean[4::4].mean())

    tasks = [(0, 4, 'Knowledge'), (4, 10, 'Algorithmic'), (10, 16, 'Translation'), (16, 25, 'Linguistic'), (25, 34, 'Bijection'), (0, 34, 'Average')]

    def get_max(num_tvs):
        max_acc = np.zeros((3, 6))
        for index, (start, end, task) in enumerate(tasks):
            accs = results[start:end].mean(axis=0) * 100
            acc_mean = np.mean(accs, axis=0)

            max_acc[0, index] = acc_mean[4 * num_tvs - 3]
            max_acc[1, index] = acc_mean[4 * num_tvs - 2]
            if num_tvs > 1:
                max_acc[2, index] = acc_mean[4 * num_tvs - 0]
        return max_acc.max(axis=0)

    def print_line(num_tvs, method, max_acc):
        for index, (start, end, task) in enumerate(tasks):
            accs = results[start:end].mean(axis=0) * 100
            acc_mean = np.mean(accs, axis=0)
            acc_std = np.std(accs, axis=0)

            def print_val(col):
                if acc_mean[col] >= max_acc[index]:
                    print(f'& \\textbf{{{acc_mean[col]:.2f}}} \\scriptsize{{$\\pm$ {acc_std[col]:.2f}}} ', end='')
                else:
                    print(f'& {acc_mean[col]:.2f} \\scriptsize{{$\\pm$ {acc_std[col]:.2f}}} ', end='')

            if method == 0:
                print_val(4 * num_tvs - 3)
            elif method == 1:
                print_val(4 * num_tvs - 2)
            else:
                print_val(4 * num_tvs)
        print('\\\\')

    for num_tvs in range(1, 6):
        if num_tvs > 1:
            print(f'\\multirow{{3}}{{*}}{{${num_tvs-1}$-shot}} ', end='')
        else:
            print(f'\\multirow{{2}}{{*}}{{$0$-shot}} ', end='')
        max_acc = get_max(num_tvs)
        print('& Baseline ', end='')
        print_line(num_tvs, 0, max_acc)
        print('& TaskV ', end='')
        print_line(num_tvs, 1, max_acc)
        if num_tvs > 1:
            print('& TaskV-M ', end='')
            print_line(num_tvs, 2, max_acc)

        if num_tvs < 5:
            print('\\midrule')

    print()


def plot_bipartite_weight_matrix(W, task, l, node_labels=None, max_linewidth=5):
    n = W.shape[0]
    assert W.shape[0] == W.shape[1], "Weight matrix must be square."

    # Normalize weights for linewidth scaling
    norm = np.abs(W) / np.max(np.abs(W))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # Node positions
    left_nodes = [(0, i) for i in range(n)]
    right_nodes = [(1, i) for i in range(n)]

    # Draw nodes
    for i, (x, y) in enumerate(left_nodes):
        ax.plot(x, y, 'o', color='skyblue', ms=2)
        if node_labels:
            ax.text(x - 0.1, y, node_labels[i], ha='right', va='center', size=4)
    for j, (x, y) in enumerate(right_nodes):
        ax.plot(x, y, 'o', color='lightcoral', ms=2)
        if node_labels:
            ax.text(x + 0.1, y, node_labels[j], ha='left', va='center', size=4)

    # Draw edges with thickness proportional to weight
    for i in range(n):
        for j in range(n):
            weight = W[i, j]
            if weight != 0:
                linewidth = norm[i, j] * max_linewidth
                ax.plot([0, 1], [i, j], color='gray', linewidth=linewidth, alpha=norm[i, j])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1, n)
    os.makedirs(os.path.join('attention', task), exist_ok=True)
    plt.savefig(os.path.join('attention', task, f'{l}.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def get_saliency(args):
    model, tokenizer = load_model_and_tokenizer(args.model_type, args.model_variant)
    seed_everything(42)
    manager = LlamaAttentionerManager(model)

    num_examples = 10
    labels = ['BOS']
    for i in range(num_examples):
        labels += ['Exp', ':', 'x', '->', 'y', '\\n']
    labels += ['Exp', ':', 'x', '->']

    task = get_task_by_name(tokenizer=tokenizer, task_name='algorithmic_prev_letter')
    datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)
    inputs = tokenize_datasets(tokenizer, datasets).to(model.device)
    target_ids = []
    for dataset in datasets:
        target_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + dataset.test_output))[0])
    target_ids = torch.tensor(target_ids).to(model.device)

    # Enable gradients on input
    manager.zero_grad()
    for param in model.parameters():
        param.grad = None
    outputs = model(**inputs)
    logits = outputs.logits  # [1, T, V]
    next_token_logits = logits[:, -1, :]  # [1, V]

    # Cross entropy loss with target token
    loss = F.cross_entropy(next_token_logits, target_ids)
    loss.backward()

    # Extract gradients and attention
    saliency = manager.grad()
    weights = []
    for map in saliency:
        weights.append(map.mean(dim=0).cpu().numpy())
    pickle.dump(weights, open('saliency.pkl', 'wb'))


def main():
    parser = argparse.ArgumentParser(description="Task Vector")
    parser.add_argument('-t', '--model_type', type=str, help='Type of the model')
    parser.add_argument('-v', '--model_variant', type=str, help='Variant of the model')
    parser.add_argument('-id', '--experiment_id', type=int, default=10, help='ID of experiment')
    parser.add_argument('-tr', '--num_train', type=int, default=10, help='Number of training samples')
    parser.add_argument('-va', '--num_valid', type=int, default=0, help='Number of validation samples')
    parser.add_argument('-tv', '--num_tvs', type=int, default=1, help='Number of TVs')
    parser.add_argument('-m', '--multiple_dataset', action='store_true', help='Use multiple datasets')

    parser.add_argument('-p', '--plot_results', action='store_true', help='Plot results')
    parser.add_argument('-a', '--plot_attention', action='store_true', help='Plot attention')
    parser.add_argument('-w', '--weight_analysis', action='store_true', help='Analyze task vector weights')
    args = parser.parse_args()

    if args.weight_analysis:
        analyze_task_vector_weights(args.model_type, args.model_variant, num_train=args.num_train)
        return

    if args.plot_results:
        # plot_bijection_results(args)
        for model_type, model_variant in MODELS_TO_EVALUATE:
            collect_multi_tv_results(model_type, model_variant, args.num_train)
        return

    if args.plot_attention:
        get_saliency(args)
        return

    if args.model_type and args.model_variant:
        run_main_experiment(args.model_type, args.model_variant, experiment_id=args.experiment_id, num_train=args.num_train,
            num_valid=args.num_valid, num_tvs=args.num_tvs, multiple_dataset=args.multiple_dataset)
    else:
        # No arguments provided; run all models
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_main_experiment(model_type, model_variant, experiment_id=args.experiment_id, num_train=args.num_train,
                num_valid=args.num_valid, num_tvs=args.num_tvs, multiple_dataset=args.multiple_dataset)


if __name__ == "__main__":
    main()
