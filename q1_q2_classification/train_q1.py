import argparse
import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    # --no_aug for the comparison plots
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.5)
    cli = parser.parse_args()
    if cli.log_dir is None:
        cli.log_dir = 'runs/q1_no_aug' if cli.no_aug else 'runs/q1_aug'

    ##################################################################
    # TODO: Create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of
    # 224x224 for the rest of the questions.
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    ##################################################################
    args = ARGS(
        epochs=cli.epochs,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=cli.lr,
        batch_size=cli.batch_size,
        step_size=cli.step_size,
        gamma=cli.gamma,
        augment=not cli.no_aug,
        log_dir=cli.log_dir,
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    print(args)

    # initializes the model
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3)
    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
