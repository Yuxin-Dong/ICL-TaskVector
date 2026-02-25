import argparse
import os
import pickle
import time

import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# Create custom colormap
colors = ['#e34a33', '#fdbb84', '#ffffff', '#9ecae1', '#3182bd'][::-1]
custom_div_cmap = LinearSegmentedColormap.from_list('BlOr', colors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--clip_r", type=float, default=1000, help="clip gradient maximum norm")
parser.add_argument("--var", type=float, default=0.1, help="initializations scale of transformer parameter")
parser.add_argument("--wd", type=float, default=0.00000001, help="weight decay")
parser.add_argument("--B", type=int, default=1000, help="batch size")
parser.add_argument("--max_iters", type=int, default=20001, help="number of iterations")
parser.add_argument("--stride", type=int, default=100, help="log stride")
parser.add_argument("--seed", type=int, default=1, help="random seed")

parser.add_argument("-L", "--n_layer", type=int, default=2, help="number of layers")
parser.add_argument("--N", type=int, default=10, help="number of in-context samples")
parser.add_argument("--d", type=int, default=4, help="dimension of inputs")
parser.add_argument("-m", "--mode", type=str, default='single', help="data processing mode")
parser.add_argument("--n_head", type=int, default=1, help="number of heads")

parser.add_argument("-t", "--test", action='store_true', help="test only")
parser.add_argument("-i", "--inseparable", action='store_true', help="inseparable x and y")
parser.add_argument("-p", "--dropout", action='store_true', help="add token-wise dropout")
args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2)

def get_path(args):
    mod = f'{"i" if args.inseparable else ""}{"p" if args.dropout else ""}'
    log_dir = f'results/{args.n_layer}_{args.N}_{args.d}_{args.mode}_{args.seed}{f"_{mod}" if mod else ""}/'
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(args, open(os.path.join(log_dir, 'args.pkl'), 'wb'))

    return log_dir, os.path.join(log_dir, 'model.pt')

log_dir, model_savepath = get_path(args)

if args.mode == 'single':
    dp = 0
elif args.mode == 'pair':
    dp = (args.N+1) * 2
elif args.mode == 'triple':
    dp = (args.N+1) * 3


class Transformer_F(nn.Module):
    def __init__(self, L, d, dp, var):
        super(Transformer_F, self).__init__()
        self.register_parameter('A', nn.Parameter(torch.zeros(L, d, d)))
        self.register_parameter('B', nn.Parameter(torch.zeros(L, d, d)))
        self.register_parameter('C', nn.Parameter(torch.zeros(L, d, d)))
        with torch.no_grad():
            self.A.normal_(0, var)
            self.B.normal_(0, var)
            self.C.normal_(0, var)

        if dp > 0:
            self.register_parameter('D', nn.Parameter(torch.zeros(L, dp, dp)))
            with torch.no_grad():
                self.D.normal_(0, var)
        else:
            self.D = [None] * L

        self.L = L


torch.manual_seed(args.seed)
model = Transformer_F(args.n_layer, args.d, dp, args.var)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.05)

# set seed and initialize initial training batch
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# a convenience function for taking a step and clipping
def clip_and_step(params, optimizer, clip_r = None):
    acc_norm_p=0
    for param in params:
        grad = param.grad
        if clip_r is not None:
            norm_p = grad.norm().item()
            if norm_p > clip_r:
                grad.mul_(clip_r/norm_p)
            acc_norm_p += norm_p

    optimizer.step()
    return acc_norm_p

def generate_data(mode, N, d, B):
    W = torch.FloatTensor(B, d, d).normal_(0, 1).to(device)

    X = torch.FloatTensor(B, N+1, d).normal_(0, 1).to(device)
    Y = torch.einsum('bij, bnj -> bni', (W, X))

    Y_test = Y[:, -1, :].clone()
    Y[:, -1, :] = 0

    if mode == 'single':
        Z = torch.cat([X, Y], dim=2)

    elif mode == 'pair':
        Z = torch.zeros(B, (N+1)*2, d*2).to(device)

        if args.inseparable:
            Z[:, ::2, d:] = X
        else:
            Z[:, ::2, :d] = X
        Z[:, 1::2, d:] = Y

    elif mode == 'triple':
        Z = torch.zeros(B, (N+1)*3, d*2).to(device)

        if args.inseparable:
            Z[:, ::3, d:] = X
        else:
            Z[:, ::3, :d] = X
        Z[:, 2::3, d:] = Y

    return Z, Y_test


def draw_mat(t, mat, title):
    if len(mat.shape) > 2:
        mat = mat.squeeze()
    mat = mat.cpu()

    maxv = max(-mat.min(), mat.max())
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    norm = TwoSlopeNorm(vmin=-maxv, vcenter=0, vmax=maxv)
    # Create a heatmap using imshow
    im = ax.imshow(mat, cmap=custom_div_cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(im, cax=cax)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(log_dir, f'{t}_{title}.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def rand_pm1(shape):
    return (2 * torch.randint(0, 2, shape) - 1).float().to(device)


def attention(A, B, C, D, Z, i=0, t=0):
    if args.dropout:
        d1 = 2 if args.mode == 'pair' else 3
        Q = torch.bernoulli(torch.full((Z.shape[0], Z.shape[1]-d1), 0.8)).unsqueeze(2).float().to(device)
        Z = torch.concat([Z[:, :-d1, :] * Q, Z[:, -d1:, :]], dim=1)
    X = Z[:, :, :args.d]
    Y = Z[:, :, args.d:]

    # Attn = torch.einsum('BNi, ij, BMj -> BNM', (X, C, X))
    Attn = torch.einsum('BNi, BMi -> BNM', (X, X))
    Attn = Attn * C[0, 0]

    # Randomly flip P to simulate symmetry
    if args.mode != 'single':
        if args.mode == 'pair':
            P = rand_pm1((Z.shape[0], args.N+1))
            P = P.repeat_interleave(2, dim=1)
        elif args.mode == 'triple':
            P = torch.zeros(Z.shape[0], (args.N+1)*3).float().to(device)
            U = rand_pm1((Z.shape[0], args.N+1))
            P[:, ::3] = U[:, :]
            P[:, 1::3] = 1
            P[:, 2::3] = U[:, :]
        Attn += P.unsqueeze(2) * D.unsqueeze(0) * P.unsqueeze(1)

    # Apply mask M
    Attn[:, :, -1] = 0

    # VX = torch.einsum('ij, BNj -> BNi', (A, X))
    # VY = torch.einsum('ij, BNj -> BNi', (B, Y))
    VX = X * A[0, 0]
    VY = Y * B[0, 0]

    X = torch.einsum('BNM, BML -> BNL', (Attn, VX))
    Y = torch.einsum('BNM, BML -> BNL', (Attn, VY))

    if t > 0:
        draw_mat(t, Attn, 'A{}'.format(i))

    return torch.cat([X, Y], dim=2) / args.N


# evaluate the loss of model, given data (Z,y)
def in_context_loss(model, Z, Y_test, d, t=0):
    for i in range(args.n_layer):
        residues = attention(model.A[i], model.B[i], model.C[i], model.D[i], Z, i=i, t=t)
        Z = Z + residues

        if t > 0:
            draw_mat(t, residues, 'R{}'.format(i))
            draw_mat(t, Z, 'Z{}'.format(i))

    diff = Z[:, -1, d:d*2] + Y_test
    loss = (diff ** 2).mean()
    return loss


def draw_result(iter):
    ####################################
    # display the parameter matrices
    ####################################
    model_test = Transformer_F(args.n_layer, args.d, dp, args.var).to(device)

    np.random.seed(99)
    torch.manual_seed(99)
    Z, Y_test = generate_data(args.mode, args.N, args.d, 1)
    t = len(log_modelparam) - 1
    with torch.no_grad():
        for param, saved in zip(model_test.parameters(), log_modelparam[t]):
            param.copy_(saved)

        draw_mat(t, Z, 'Z')
        if iter == 0:
            return

        in_context_loss(model_test, Z, Y_test, args.d, t=t)

    for l in range(args.n_layer):
        draw_mat(t, torch.block_diag(log_modelparam[t][0][l], log_modelparam[t][1][l]), 'V{}'.format(l))

    for l in range(args.n_layer):
        if args.mode != 'single':
            draw_mat(t, torch.block_diag(log_modelparam[t][2][l], torch.zeros(args.d, args.d).to(device),
                log_modelparam[t][3][l]), 'Q{}'.format(l))
            # draw_mat(t, log_modelparam[t][3][l].T, 'D{}'.format(l))

            # D = log_modelparam[-1][3]
            # A = D[0, 1::3, ::3]
            # draw_mat(0, A.T @ A, 'ATA')
        else:
            draw_mat(t, torch.block_diag(log_modelparam[t][2][l], torch.zeros(args.d, args.d).to(device)), 'Q{}'.format(l))

    ####################################
    # compute test loss
    ####################################
    np.random.seed(99)
    torch.manual_seed(99)
    Z, Y_test = generate_data(args.mode, args.N, args.d, args.B)

    log_losses = []
    for i in range(len(log_modelparam)):
        with torch.no_grad():
            for param, saved in zip(model_test.parameters(), log_modelparam[i]):
                param.copy_(saved)
            log_losses.append(in_context_loss(model_test, Z, Y_test, args.d).item())

    ####################################
    # plot the test loss with error bars
    ####################################
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.plot(np.arange(len(log_losses)) * args.stride, log_losses, color='red', lw=3)
    ax.set_xlabel('Iteration', fontsize=40)
    ax.set_ylabel('ICL Test Loss', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30, width=3, length=10)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=3, length=5)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'loss.png'), dpi=100)
    plt.close(fig)

    return log_losses

if not args.test:
    log_modelparam = []

    for t in range(args.max_iters):
        start = time.time()
        # generate a new batch of training set
        Z, Y_test = generate_data(args.mode, args.N, args.d, args.B)
        base_loss = in_context_loss(model, Z, Y_test, args.d)

        l1_norm = torch.stack([p.abs().sum() for p in model.parameters()]).sum()
        loss = base_loss + (args.wd if t >= 2000 else 0) * l1_norm

        # compute gradient, take step
        loss.backward()
        norms = clip_and_step(model.parameters(), optimizer, clip_r=args.clip_r)
        optimizer.zero_grad()
        end = time.time()

        if t % 100 == 0:
            # save model parameters
            log_modelparam.append([p.detach().clone() for p in model.parameters()])
            print('iter {} | loss: {}  reg: {}  time: {}  gradnorm: {}'.format(t, base_loss.item(), l1_norm.item(), end - start, norms))

        if t % 2500 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
            draw_result(t)
            torch.save(log_modelparam, model_savepath)

log_modelparam = torch.load(model_savepath)

print(draw_result(-1)[-1])

