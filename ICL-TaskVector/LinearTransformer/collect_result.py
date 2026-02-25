import argparse
import os
import pickle
import time

import numpy as np
import seaborn
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from weight_sum import optimize_upper_triangular

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=4, suppress=True)

class Transformer_F(nn.Module):
    def __init__(self, L, d, dp):
        super(Transformer_F, self).__init__()
        self.register_parameter('A', nn.Parameter(torch.zeros(L, d, d)))
        self.register_parameter('B', nn.Parameter(torch.zeros(L, d, d)))
        self.register_parameter('C', nn.Parameter(torch.zeros(L, d, d)))

        if dp > 0:
            self.register_parameter('D', nn.Parameter(torch.zeros(L, dp, dp)))
        else:
            self.D = [None] * L

        self.L = L
        self.d = d
        self.dp = dp

def generate_data(mode, N, d, B, rankone=False):
    if rankone:
        a = torch.FloatTensor(B, d, 1).normal_(0, 1).to(device)
        W = a @ torch.FloatTensor(B, 1, d).normal_(0, 1).to(device)
        X = torch.FloatTensor(B, N+1, 1).normal_(0, 1).to(device) @ a.transpose(1, 2)
    else:
        W = torch.FloatTensor(B, d, d).normal_(0, 1).to(device)
        X = torch.FloatTensor(B, N+1, d).normal_(0, 1).to(device)

    Y = torch.matmul(X, W)

    Y_test = Y[:, -1, :].clone()
    Y[:, -1, :] = 0

    if mode == 'single':
        Z = torch.cat([X, Y], dim=2)

    elif mode == 'pair':
        Z = torch.zeros(B, (N+1)*2, d*2).to(device)

        Z[:, ::2, :d] = X
        Z[:, 1::2, d:] = Y

    else:
        Z = torch.zeros(B, (N+1)*3, d*2).to(device)

        Z[:, ::3, :d] = X
        Z[:, 2::3, d:] = Y

    return Z, Y_test

def attention(A, B, C, D, Z, mode, N, d):
    X = Z[:, :, :d]
    Y = Z[:, :, d:]

    Attn = torch.einsum('BNi, BMi -> BNM', (X, X))
    Attn = Attn * C[0, 0]

    if mode != 'single':
        Attn += D.unsqueeze(0)

    Attn[:, :, -1] = 0

    VX = X * A[0, 0]
    VY = Y * B[0, 0]

    X = torch.einsum('BNM, BML -> BNL', (Attn, VX))
    Y = torch.einsum('BNM, BML -> BNL', (Attn, VY))

    return torch.cat([X, Y], dim=2) / N

def scaled_loss(model, Z, Y_test, mode, d, N):
    for i in range(model.L):
        residues = attention(model.A[i], model.B[i], model.C[i], model.D[i], Z, mode, N, model.d)
        Z = Z + residues

    predict = Z[:, -1, model.d:model.d*2]
    predict_norm = predict / predict.norm(dim=1, keepdim=True) * np.sqrt(d)
    Y_test_norm = Y_test / Y_test.norm(dim=1, keepdim=True) * np.sqrt(d)
    return ((predict_norm + Y_test_norm) ** 2).mean()

def in_context_loss(model, Z, Y_test, mode, d, N):
    for i in range(model.L):
        residues = attention(model.A[i], model.B[i], model.C[i], model.D[i], Z, mode, N, model.d)
        Z = Z + residues

    diff = Z[:, -1, model.d:model.d*2] + Y_test
    loss = (diff ** 2).mean()
    return loss

def task_vector_loss(model, Z, Y_test, mode, d, N):
    Z, Z_zero = Z.clone(), Z.clone()
    Z_zero[:, :N*3, :] = 0

    Z[:, N*3:, :] = 0
    tvs = attention(model.A[0], model.B[0], model.C[0], model.D[0], Z, 'triple', N, model.d)[:, N*3+1, :]
    tvs /= tvs.norm(dim=1).mean() # simulate layer normalization
    Z_zero[:, N*3+1, :] = tvs

    for i in range(model.L):
        residues = attention(model.A[i], model.B[i], model.C[i], model.D[i], Z_zero, 'triple', N, model.d)
        Z_zero = Z_zero + residues

    predict = Z_zero[:, -1, model.d:model.d*2]
    predict_norm = predict / predict.norm(dim=1, keepdim=True) * np.sqrt(d)
    Y_test_norm = Y_test / Y_test.norm(dim=1, keepdim=True) * np.sqrt(d)
    return ((predict_norm + Y_test_norm) ** 2).mean()

def get_path(L, N, d, mode, seed):
    log_dir = f'results/{L}_{N}_{d}_{mode}_{seed}/'
    return log_dir, os.path.join(log_dir, 'model.pt')

def get_loss(L, N, d, mode, seed, model, Z, Y_test, loss_func):
    log_dir, model_savepath = get_path(L, N, d, mode, seed)
    log_modelparam = torch.load(model_savepath)

    log_losses = []
    for i in range(1, 6):
        with torch.no_grad():
            for param, saved in zip(model.parameters(), log_modelparam[-i]):
                param.copy_(saved)
            log_losses.append(loss_func(model, Z, Y_test, mode, d, N).item())

    return np.mean(log_losses)

def set_default_configs(plt, seaborn=None):
    # assuming in the figsize height = 5
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['font.family'] = 'Calibri'
    # Make sure no Type 3 fonts are used. Such fonts are not accepted by some conferences/journals.
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    if seaborn is not None:
        seaborn.set_style("whitegrid")

def collect_icl_result():
    if os.path.exists('icl_results.pkl'):
        data = pickle.load(open('icl_results.pkl', 'rb'))
    else:
        data = np.zeros((3, 3, 6))
        for mode in ['single', 'pair', 'triple']:
            for N in range(5, 31, 5):
                if mode == 'single':
                    dp = 0
                    Ls = [1, 2, 3]
                    m = 0
                elif mode == 'pair':
                    dp = (N+1) * 2
                    Ls = [2, 3]
                    m = 1
                else:
                    dp = (N+1) * 3
                    Ls = [2, 3]
                    m = 2

                np.random.seed(99)
                torch.manual_seed(99)
                Z, Y_test = generate_data(mode, N, 4, 1000)

                for L in Ls:
                    model = Transformer_F(L, 4, dp).to(device)
                    model.eval()

                    losses = []
                    for seed in range(40):
                        try:
                            losses.append(get_loss(L, N, 4, mode, seed, model, Z, Y_test, in_context_loss))
                        except:
                            pass
                    data[m, L-1, N//5-1] = np.min(losses)

        pickle.dump(data, open('icl_results.pkl', 'wb'))

    return data

def collect_icl_N1_result():
    if os.path.exists('icln1_results.pkl'):
        data = pickle.load(open('icln1_results.pkl', 'rb'))
    else:
        data = np.zeros(2)
        N = 1
        dp = (N+1) * 3
        Ls = [2, 3]

        np.random.seed(99)
        torch.manual_seed(99)
        Z, Y_test = generate_data('triple', N, 4, 1000, rankone=True)

        for L in Ls:
            model = Transformer_F(L, 4, dp).to(device)
            model.eval()

            losses = []
            for seed in range(40):
                try:
                    losses.append(get_loss(L, N, 4, 'triple', seed, model, Z, Y_test, scaled_loss))
                except:
                    pass
            data[L-2] = np.min(losses)
            print(data[L-2])

        pickle.dump(data, open('icln1_results.pkl', 'wb'))

    return data

def collect_tv_result():
    if os.path.exists('tv_results.pkl'):
        data = pickle.load(open('tv_results.pkl', 'rb'))
    else:
        data = np.zeros((2, 4))
        for N in range(5, 21, 5):
            dp = (N+1) * 3
            Ls = [2, 3]

            np.random.seed(99)
            torch.manual_seed(99)
            Z, Y_test = generate_data('triple', N, 4, 1000, rankone=True)

            for L in Ls:
                model = Transformer_F(L, 4, dp).to(device)
                model.eval()

                losses = []
                for seed in range(40):
                    try:
                        losses.append(get_loss(L, N, 4, 'triple', seed, model, Z, Y_test, task_vector_loss))
                    except:
                        pass
                data[L-2, N//5-1] = np.min(losses)
                print(data[L-2, N//5-1])

        pickle.dump(data, open('tv_results.pkl', 'wb'))

    return data

def plot_icl_results(results, L):
    fig, ax = plt.subplots(figsize=(4.5, 3))

    xs = np.arange(5, 31, 5)
    bar_width = 10.0 / (results.size // len(xs) + 2)  # adjust spacing

    if L == 2:
        labels = [
            'S ($L=1$)',
            'P ($L=2$)',
            'T ($L=2$)',
            'S ($L=2$)',
        ]
        configs = [(0, 0), (1, 1), (2, 1), (0, 1)]
        ax.set_ylim(bottom=1e-2, top=10)
    else:
        labels = [
            'S ($L=2$)',
            'P ($L=3$)',
            'T ($L=3$)',
            'S ($L=3$)',
        ]
        configs = [(0, 1), (1, 2), (2, 2), (0, 2)]
        ax.set_ylim(bottom=1e-5, top=10)

    n_configs = len(configs)

    for i, (r, c) in enumerate(configs):
        x_pos = xs + (i - n_configs / 2 + 0.5) * bar_width
        ax.bar(x_pos, results[r, c], width=bar_width, label=labels[i])

    ax.set_xlabel('$n$')
    ax.set_ylabel('ICL Risk')
    ax.set_yscale('log')
    ax.legend(ncol=2)
    ax.set_xticks(xs)
    # ax.set_xticklabels(xs)

    fig.savefig(f'figures/icl_results_{L}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

def plot_tv_results(icln1_results, tv_results):
    fig, ax = plt.subplots(figsize=(2.5, 3))

    xs = np.array([2, 3])
    bar_width = 8.0 / (80 // len(xs) + 2)  # adjust spacing

    labels = [
        'ICL ($n=1$)',
        'Task Vector',
    ]
    configs = [icln1_results, tv_results]

    n_configs = len(configs)

    for i, results in enumerate(configs):
        x_pos = xs + (i - n_configs / 2 + 0.5) * bar_width
        ax.bar(x_pos, results, width=bar_width, label=labels[i])

    ax.set_xlabel('$L$')
    ax.set_ylabel('ICL Risk')
    ax.set_ylim(top=0.45)
    ax.set_xlim(left=1.5, right=3.5)
    ax.legend()
    ax.set_xticks(xs)
    # ax.set_xticklabels(xs)

    fig.savefig(f'figures/tv_results.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

def plot_tv_weights():
    np.random.seed(24)
    tv_weights = pickle.load(open('task_weights.pkl', 'rb'))
    sim_weights = np.abs(optimize_upper_triangular(10, 0.2)[:,-1])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(np.arange(len(tv_weights))+1, tv_weights, width=0.5)

    ax.set_xlabel('$i$')
    ax.set_ylabel(r'$\|z_{\mathrm{tv}}^{-i} - z_{\mathrm{tv}}\|$')
    ax.set_ylim(bottom=1.2)
    ax.set_xticks([1, 5, 10])

    ax2 = ax.twinx()
    lc = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    ax2.plot(np.arange(len(tv_weights))+1, sim_weights, color=lc, lw=2)
    ax2.set_ylabel(r'$\beta_i$', rotation=-90, labelpad=20)

    fig.savefig(f'figures/tv_weights.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

def plot_bipartite_weight_matrix(W, l, labels, highlights, max_linewidth=3):
    n = W.shape[0]

    # Normalize weights for linewidth scaling
    norm = np.abs(W) / np.max(np.abs(W))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')

    # Node positions
    left_nodes = [(i, 1) for i in range(n)]
    right_nodes = [(i, 0) for i in range(n)]

    colors = ['#e34a33', '#3182bd']

    # Draw nodes
    for i, (x, y) in enumerate(left_nodes):
        ax.plot(x, y, 'o', color=colors[0], ms=10)
        ax.text(x, y + 0.1, labels[i], ha='center', va='bottom', size=12)
    for j, (x, y) in enumerate(right_nodes):
        ax.plot(x, y, 'o', color=colors[1], ms=10)
        ax.text(x, y - 0.1, labels[j], ha='center', va='top', size=12)

    # Draw edges with thickness proportional to weight
    for i in range(n):
        for j in range(n):
            weight = W[i, j]
            if weight != 0:
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] if (i, j) in highlights else 'gray'
                norm[i, j] = min(norm[i, j], 0.7) / 0.7
                linewidth = norm[i, j] * max_linewidth
                ax.plot([i, j], [1, 0], color=color, linewidth=linewidth, alpha=norm[i, j])

    ax.set_xlim(-1, n)
    ax.set_ylim(-0.3, 1.3)
    plt.savefig(f'figures/saliency_{l}.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_saliency():
    saliency = pickle.load(open('saliency.pkl', 'rb'))

    num_examples = 3
    labels = []
    indices = []
    for i in range(1, num_examples+1):
        labels += [f'$x_{i}$', r'$\to$', f'$y_{i}$']
        indices += [(10-num_examples+i)*6-3, (10-num_examples+i)*6-2, (10-num_examples+i)*6-1]
    labels += [r'$x_{\mathrm{test}}$', r'$\to$']
    indices += [-2, -1]

    plot_bipartite_weight_matrix(saliency[10][indices, :][:, indices], 10, labels,
        [(2, 0), (2, 1), (2, 2), (5, 3), (5, 4), (5, 5), (8, 6), (8, 7), (8, 8)])
    plot_bipartite_weight_matrix(saliency[12][indices, :][:, indices], 12, labels,
        [(10, 2), (10, 5), (10, 8)])

if __name__ == '__main__':
    set_default_configs(plt, seaborn)

    # icl_results = collect_icl_result()
    # plot_icl_results(icl_results, 2)
    # plot_icl_results(icl_results, 3)

    # icln1_results = collect_icl_N1_result()
    # tv_results = collect_tv_result()
    # tv_results = tv_results.mean(axis=1)
    # print(icln1_results, tv_results)
    # plot_tv_results(icln1_results, tv_results)

    # plot_tv_weights()

    plot_saliency()
