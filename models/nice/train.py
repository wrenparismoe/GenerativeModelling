import argparse

import torch
import numpy as np
from .data import load_data
from .loss import (
    GaussianNICECriterion,
    LogisticNICECriterion,
    StandardLogisticDistribution,
)
from .model import NICE
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import set_seed
import os

import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Training loop
def train(args, model, train_dataset, test_dataset):
    if args.model_path is not None:
        assert os.path.exists(args.model_path), "Error: Model path does not exist"
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

    # Create the optimizer (AdaM)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay,
        # betas=(args.B1, args.B2),
        # eps=args.eps,
    )

    if args.prior == "gaussian":
        nice_loss_fn = GaussianNICECriterion(average=False)
    elif args.prior == "logistic":
        # nice_loss_fn = LogisticNICECriterion(average=False)
        distribution = StandardLogisticDistribution(device=device)

        def nice_loss_fn(y, log_jacobian):
            log_likelihood = distribution.log_pdf(y) + log_jacobian
            return -log_likelihood.sum()

    else:
        raise ValueError("Invalid loss function")

    num_steps = 0
    loss = 0
    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} of {args.epochs}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        for batch, labels in (
            pbar := tqdm(train_loader, desc="Training", postfix={"Loss": loss})
        ):
            y, log_jacobian = model(batch.to(device))
            loss = nice_loss_fn(y, log_jacobian)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1
            pbar.set_postfix({"Loss": loss.item()})

            if args.wandb:
                wandb.log({"train_loss": loss.item()}, step=num_steps)
        del train_loader

        if args.save_model and (epoch) % args.save_epoch == 0:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _args = f"{args.dataset}_{args.num_layers}_{args.hidden_dim}_{args.prior}_epoch{epoch+1}_{_device}.pt"
            torch.save(model.state_dict(), os.path.join(args.save_path, _args))
            print(f"Model saved >>> {_args}")

        # val_results = validate(args, model, test_dataset, nice_loss_fn)
        # print values of val_results with a precision of 4 decimal places
        # print({k: f"{v:.4f}" for k, v in val_results.items()})
        # if args.wandb:
        # wandb.log(val_results, step=epoch)


num_val_steps = 0


def validate(args, model, dataset, loss_fn):
    """Perform validation on a dataset."""
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # set model to eval mode (turns batch norm training off)
    model.eval()
    global num_val_steps

    loss = 0
    validation_losses = []
    with torch.no_grad():
        for batch, _ in (
            pbar := tqdm(val_loader, desc="Validation", postfix={"Loss": loss})
        ):
            y, log_jacobian = model(batch.to(device))
            loss = loss_fn(y, log_jacobian)
            validation_losses.append(loss.item())
            pbar.set_postfix({"Loss": loss.item()})
            num_val_steps += 1

            if args.wandb:
                wandb.log({"val_loss": loss.item()}, step=num_val_steps)
    del val_loader
    model.train()

    results = {
        "val_loss_min": np.amin(validation_losses),
        "val_loss_med": np.median(validation_losses),
        "val_loss_mean": np.mean(validation_losses),
        "val_loss_max": np.amax(validation_losses),
    }

    return results


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--learning_rate", dest="lr", type=float, default=0.0002)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--beta1", dest="B1", type=float, default=0.9)
    parser.add_argument("--beta2", dest="B2", type=float, default=0.999)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--epsilon", dest="eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonlinear_layers", dest="num_layers", type=int, default=5)
    parser.add_argument(
        "--nonlinear_hidden_dim", dest="hidden_dim", type=int, default=1000
    )
    parser.add_argument(
        "--prior", type=str, choices=("gaussian", "logistic"), default="logistic"
    )
    # args for save_model, save_epoch, save_path, and model_path
    parser.add_argument(
        "--no_save_model",
        dest="save_model",
        action="store_false",
        help="Do not save the model. If unspecified, 'save_model'=True",
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=9,
        help="Number of saves between epochs. Default: 10",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model. Default: None",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./saved_models/nice",
        help="Path to save model. Default: './saved_models/nice'",
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seed(args)

    if args.wandb:
        # Initialize wandb
        wandb.init(
            project="GenerativeModelling",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataset": args.dataset,
                "learning_rate": args.lr,
                "decay": args.decay,
                "beta1": args.B2,
                "beta2": args.B2,
                "lambda": args.lam,
                "epsilon": args.eps,
                "nonlinear_layers": args.num_layers,
                "nonlinear_hidden_dim": args.hidden_dim,
                "prior": args.prior,
            },
        )

    # load data using load_data function from data.py
    train_dataset, test_dataset = load_data(args.dataset)
    print(
        f"Loaded {args.dataset} dataset with {len(train_dataset)} training samples and {len(test_dataset)} test samples."
    )

    input_dim = None
    if args.dataset == "mnist":
        input_dim = 28 * 28
    else:
        raise NotImplementedError(
            f"Dataset {args.dataset} not implemented yet. Please choose from ['mnist']"
        )

    # Print device to run model on
    print(f"Running model on {device}")

    # Create the model
    model = NICE(
        input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers
    ).to(device)

    train(args, model, train_dataset, test_dataset)
