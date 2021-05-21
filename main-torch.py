from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from vime_utils import EarlyStopping
from vime_trainer import train_unsup, test_unsup
from data_loader import uci_datasets
from vime_model import VIME_Encoder

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch VIME")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--unsup_train_file_path",
        type=str,
        default="./data/uci_income_train_unlabel.csv",
        help="file path to load unlabeled train data",
    )
    parser.add_argument(
        "--label_train_file_path",
        type=str,
        default="./data/uci_income_train_label.csv",
        help="file path to load labeld train data",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        default="./data/uci_income_test.csv",
        help="file path to load test data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, metavar="N", help="hidden dimensions",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        metavar="Alpha",
        help="ratio between reconstruction loss and mask loss",
    )
    parser.add_argument(
        "--pm", type=float, default=0.2, help="Ratio of masked data",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.75,
        metavar="M",
        help="Learning rate step gamma",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=9,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:5" if use_cuda else "cpu")

    # train_loader, valid_loader, test_loader = mnist_datasets(args)
    train_loader, valid_loader, test_loader = uci_datasets(args)
    input_dim = 105
    model = VIME_Encoder(input_dim, args.hidden_dim).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)
    early_stopping = EarlyStopping(verbose=True, path="./save_model/uci_encoder.pt")
    train_loss = []
    valid_loss = []
    for epoch in range(1, args.epochs + 1):
        loss = train_unsup(args, model, device, train_loader, optimizer, epoch)
        train_loss.append(loss)
        val_loss = test_unsup(args, model, device, valid_loader)
        valid_loss.append(val_loss)
        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
