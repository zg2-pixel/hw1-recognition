# q2 t-SNE plot: python q1_q2_classification/tsne_q2.py --checkpoint checkpoint-model-epoch50.pth
# (run from hw1-recognition after train_q2.py)
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from utils import ARGS
from voc_dataset import VOCDataset
from train_q2 import ResNet  # noqa: F401  (needed to unpickle the checkpoint)


def extract_features(model, loader, device):
    feats, labels, weights = [], [], []
    # hook on avgpool -> 512-d feature before fc
    captured = {}

    def hook(_module, _inp, out):
        captured['feat'] = torch.flatten(out, 1)

    handle = model.resnet.avgpool.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for data, target, wgt in loader:
            model(data.to(device))
            feats.append(captured['feat'].cpu())
            labels.append(target)
            weights.append(wgt)
    handle.remove()
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(weights).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint-model-epoch50.pth')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--out', type=str, default='q2_tsne.png')
    parser.add_argument('--seed', type=int, default=0)
    cli = parser.parse_args()

    args = ARGS(use_cuda=True, inp_size=224)
    device = args.device
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    model = torch.load(cli.checkpoint, map_location='cpu', weights_only=False).to(device)

    test_set = VOCDataset('test', args.inp_size)
    idx = np.random.choice(len(test_set), size=cli.num_images, replace=False)
    loader = DataLoader(Subset(test_set, idx.tolist()), batch_size=100,
                        shuffle=False, num_workers=4)

    feats, labels, _ = extract_features(model, loader, device)
    print('features:', feats.shape)

    emb = TSNE(n_components=2, perplexity=30, init='pca', random_state=cli.seed).fit_transform(feats)

    # multi-label image -> mean of its class colours
    class_names = VOCDataset.CLASS_NAMES
    palette = plt.get_cmap('tab20')(np.linspace(0, 1, len(class_names)))[:, :3]
    counts = labels.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1
    colors = (labels @ palette) / counts

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, alpha=0.85, linewidths=0)
    ax.set_title(f't-SNE of fine-tuned ResNet-18 features ({cli.num_images} PASCAL VOC test images)')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    handles = [Line2D([0], [0], marker='o', linestyle='', color=palette[i], label=class_names[i], markersize=7)
               for i in range(len(class_names))]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False,
              title='GT class (multi-label = mean colour)')
    fig.tight_layout()
    fig.savefig(cli.out, dpi=200, bbox_inches='tight')
    print('saved', os.path.abspath(cli.out))
