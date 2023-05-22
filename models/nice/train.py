import argparse

import torch
import numpy as np
from data import load_data
from loss import GaussianNICECriterion
from model import NICE
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed

import wandb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Training loop
def train(args, model, train_dataset, test_dataset):
    # Construct DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the optimizer (AdaM)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay, betas=(args.B1, args.B2),
        eps=args.eps
    )

    nice_loss_fn = GaussianNICECriterion(average=True)

    num_steps = 0
    loss = None
    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} of {args.epochs}")
        for batch, labels in (pbar := tqdm(train_loader, desc="Training", postfix={"Loss": loss})):
            optimizer.zero_grad()
            outputs = model(batch.to(device))
            loss = nice_loss_fn(outputs, model.scaling_diag)
            loss.backward()
            optimizer.step()
            num_steps += 1
            pbar.set_postfix({"Loss": loss.item()})
            
            if args.wandb:
                wandb.log({"train_loss": loss.item()}, step=num_steps)
        
        val_results = validate(args, model, test_loader, nice_loss_fn)
        print(val_results)

        if args.wandb:
            wandb.log(val_results, step=num_steps)

num_val_steps = 0
def validate(args, model, dataloader, loss_fn):
    """Perform validation on a dataset."""
    # set model to eval mode (turns batch norm training off)
    model.eval()
    global num_val_steps

    loss = None
    validation_losses = []
    with torch.no_grad():
        for inputs,_ in (pbar := tqdm(dataloader, desc="Validation", postfix={"Loss": loss})):
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, model.scaling_diag)
            validation_losses.append(loss.item())
            pbar.set_postfix({"Loss": loss.item()})
            num_val_steps += 1

            if args.wandb:
                wandb.log({"val_loss": loss.item()}, step=num_val_steps)
        
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--learning_rate", dest="lr", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=1e-6)
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
        "--prior", type=str, choices=("gaussian", "logistic"), default="gaussian"
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
