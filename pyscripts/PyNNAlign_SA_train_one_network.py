import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import wandb
import torch.nn as nn

#Resolve project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

#Ensure project root is in Python path for absolute imports
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from src.models import NNAlign_MA
from src.datasets import NNAlign_MA_Dataset
from src.datasets_utils import Collator_SA_Blosum_ClassII, load_blosum, load_pseudoseqs
from src.trainers import NNAlign_MA_trainer
from src.utils import plot_training_curves


def args_parser():

    parser = argparse.ArgumentParser(description="Train one NNAlign_MA-like model.")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size (number of samples per training step).")
    parser.add_argument("-sa", "--burn_in_sa", type=int, default=20, help="Number of initial burn-in epochs using single-allele (SA) data only.")
    parser.add_argument("-tr", "--training_file", type=str, help="Path to the training data file.")
    parser.add_argument("-bl", "--blosum_file", type=str, help="Path to the blosum file.")
    parser.add_argument("-ps", "--pseudoseqs_file", type=str, help="Path to the pseudoseqs file.")
    parser.add_argument("-syn", "--synapse_file", type=str, help="Path to save the model weights.")
    parser.add_argument("-nh", "--n_hidden", type=int, default=56, help="Number of hidden neurons.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025, help="Learning rate.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("-e", "--num_epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("-val", "--validation_file", type=str, help="Path to the validation data file.")
    parser.add_argument("-tc", "--training_curves", type=str, help="Path to save the training curves figure.")
    parser.add_argument("-wb", "--wandb_name", type=str, help="Name of this run to be logged in W&B.")
    parser.add_argument("-w", "--wandb_dir", type=str, help="The name of the directory to save wandb generated files.")
    parser.add_argument("-a","--activation", choices=["relu","tanh"], default="tanh")
    parser.add_argument("-c","--criterion",  choices=["bce","mse"],   default="mse")
    parser.add_argument("-o","--optimizer",  choices=["adam","adamw","sgd"], default="sgd")

    return parser.parse_args()


def main():

    args = args_parser()

    data_file = args.training_file
    val_file = args.validation_file
    blosum_file = args.blosum_file
    pseudoseqs_file = args.pseudoseqs_file
    syn_path = args.synapse_file
    batch_size = args.batch_size
    SA_burn_in = args.burn_in_sa #number of single-allele burn-in epochs


    run = wandb.init(project="PyNNAlign_MA",
                     name=args.wandb_name,
                     config=vars(args),
                     dir=args.wandb_dir)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True

    else:
        device = torch.device("cpu")
        pin_memory = False

    print(f"[INFO] Starting script. Target device: {device}")

    #Load preprocessing resources
    print("[1/4] Loading BLOSUM resources...")
    blosum_matrix, aa_to_idx = load_blosum(blosum_file=blosum_file)
    
    print(f"[2/4] Loading pseudosequences from {pseudoseqs_file}...")
    pseudoseqs_dict = load_pseudoseqs(aa_to_idx=aa_to_idx,
                                      blosum_matrix=blosum_matrix, 
                                      pseudoseqs_file=pseudoseqs_file)
    
    #Initialize training dataset
    dataset_class = NNAlign_MA_Dataset
    print(f"[3/4] Loading full dataset into RAM from {data_file}...")
    dataset_sa = dataset_class(file_path=data_file)

    #Initialize validation dataset
    dataset_val = dataset_class(file_path=val_file)

    #Initialize collator for batch construction
    collator = Collator_SA_Blosum_ClassII(blosum_matrix=blosum_matrix, aa_to_idx=aa_to_idx, pseudoseqs_dict=pseudoseqs_dict)

    #Initialize dataloaders
    loader_sa = DataLoader(dataset_sa, 
                           batch_size=batch_size, 
                           shuffle=True,
                           num_workers=4, 
                           collate_fn=collator, 
                           pin_memory=pin_memory,
                           persistent_workers=True)
    
    loader_val = DataLoader(dataset_val, 
                           batch_size=batch_size * 5, 
                           shuffle=False,
                           num_workers=4, 
                           collate_fn=collator, 
                           pin_memory=pin_memory,
                           persistent_workers=True)
    
    ACTIVATION_FACTORY = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh()}
    
    activation = ACTIVATION_FACTORY[args.activation]
    
    print(f"[MODEL] Initializing NNAlign_SA architecture...")
    if args.criterion == "bce":
        logits = True
    else:
        logits = False

    model = NNAlign_MA(n_hidden=args.n_hidden,
                       activation = activation,
                       logits=logits)

    CRITERION_FACTORY = {
        "bce": nn.BCEWithLogitsLoss(),
        "mse": nn.MSELoss()}

    criterion = CRITERION_FACTORY[args.criterion]

    lr = args.learning_rate
    wd = args.weight_decay

    OPTIMIZER_FACTORY = {
        "adam": lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
        "adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
        "sgd": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd)}

    optimizer = OPTIMIZER_FACTORY[args.optimizer](
        model.parameters(),
        lr,
        wd
    )

    trainer = NNAlign_MA_trainer(model=model, 
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=device,
                                 SA_burn_in=SA_burn_in,
                                 loader_ma=loader_sa,
                                 loader_sa=loader_sa,
                                 loader_val=loader_val,
                                 logger=run)
    
    #Train model and save learned weights
    trainer.train(num_epochs=args.num_epochs)
    print(f"[DONE] Saving weights to {syn_path}")

    trainer.save(syn_path=syn_path)

    #Plot and save training curves
    plot_training_curves(trainer=trainer, save_path=args.training_curves)

    wandb.finish()

if __name__ == "__main__":
    main()