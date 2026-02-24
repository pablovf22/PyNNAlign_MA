import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Resolve project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

# Ensure project root is in Python path for absolute imports
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from src.models import NNAlign_MA
from src.datasets import NNAlign_MA_Dataset, NNAlign_MA_IterableDataset, NNAlign_MA_OffsetDataset
from src.datasets_utils import CollatorClassII, load_blosum, load_allelist, load_pseudoseqs
from src.trainers import NNAlign_MA_trainer


def args_parser():

    parser = argparse.ArgumentParser(description="Train one NNAlign_MA-like model.")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size (number of samples per training step).")
    parser.add_argument("-sa", "--burn_in_sa", type=int, default=20, help="Number of initial burn-in epochs using single-allele (SA) data only.")
    parser.add_argument("-tr", "--training_file", type=str, help="Path to the training data file.")
    parser.add_argument("-a", "--allelelist_file", type=str, help="Path to the allelelist file.")
    parser.add_argument("-bl", "--blosum_file", type=str, help="Path to the blosum file.")
    parser.add_argument("-ps", "--pseudoseqs_file", type=str, help="Path to the pseudoseqs file.")
    parser.add_argument("-syn", "--synapse_file", type=str, help="Path to save the model weights.")
    parser.add_argument("-nh", "--n_hidden", type=int, default=66, help="Number of hidden neurons.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("-e", "--num_epochs", type=int, default=300, help="Number of epochs.")

    return parser.parse_args()
    
    
def main():

    args = args_parser()

    data_file = args.training_file
    allelelist_file = args.allelelist_file
    blosum_file = args.blosum_file
    pseudoseqs_file = args.pseudoseqs_file
    syn_path = args.synapse_file

    batch_size = args.batch_size
    SA_burn_in = args.burn_in_sa  # number of single-allele burn-in epochs
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True

    else:
        device = torch.device("cpu")
        pin_memory = False

    print(f"[INFO] Starting script. Target device: {device}")

    # Load preprocessing resources
    print("[1/4] Loading BLOSUM and Allele resources...")
    blosum_matrix, aa_to_idx = load_blosum(blosum_file=blosum_file)
    allele_dict = load_allelist(allelelist_file=allelelist_file)
    print(f"[2/4] Loading pseudosequences from {pseudoseqs_file}...")
    pseudoseqs_dict = load_pseudoseqs(aa_to_idx=aa_to_idx,
                                      blosum_matrix=blosum_matrix, 
                                      pseudoseqs_file=pseudoseqs_file)

    # Initialize collator for batch construction
    collator = CollatorClassII(blosum_matrix=blosum_matrix, aa_to_idx=aa_to_idx, pseudoseqs_dict=pseudoseqs_dict, allele_dict=allele_dict)
    dataset_class = NNAlign_MA_Dataset
    print(f"[3/4] Loading full dataset into RAM from {data_file}...")
    dataset_ma = dataset_class(file_path=data_file)
    print("[4/4] Filtering Single-Allele (SA) indices for burn-in...")
    sa_idx = [
        i for i, (pep, lab, cell_line) in enumerate(dataset_ma.dataset)
        if len(allele_dict[cell_line]) == 1
    ]

    sa_dataset = torch.utils.data.Subset(dataset_ma, sa_idx)
    print(f"[STATUS] Dataset ready. Total samples: {len(dataset_ma)} | SA samples: {len(sa_idx)}")

    def make_loader(epoch):

        if epoch == 0:
            print(f"--- Phase: SA Burn-in (Epochs 1-{SA_burn_in}) ---")
        elif epoch == SA_burn_in:
            print(f"--- Phase: Full Multi-Allele (Epochs {SA_burn_in+1}+) ---")

        # Create epoch-dependent dataset (SA burn-in phase)
        dataset = sa_dataset if epoch < SA_burn_in else dataset_ma

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=collator,
            pin_memory=pin_memory,
            shuffle=True
        )

        return loader


    # Initialize model, loss and optimizer
    print(f"[MODEL] Initializing NNAlign_MA architecture...")
    model = NNAlign_MA(n_hidden=arg.n_hidden)

    criterion = torch.nn.BCEWithLogitsLoss()
    lr = args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = NNAlign_MA_trainer(model=model, 
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=device,
                                 make_loader=make_loader)
    
    # Train model and save learned weights
    print(f"[TRAIN] Beginning training for {SA_burn_in} SA epochs + remaining MA epochs...")
    trainer.train(num_epochs=args.num_epochs)
    print(f"[DONE] Saving weights to {syn_path}")
    trainer.save(syn_path=syn_path)

    
if __name__ == "__main__":
    main()