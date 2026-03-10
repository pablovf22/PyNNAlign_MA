import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

#Resolve project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

#Ensure project root is in Python path for absolute imports
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from src.models import NNAlign_MA, NNAlign_MA_Extra_Features
from src.datasets import NNAlign_MA_Dataset
from src.datasets_utils import Collator_SA_Blosum_ClassII_Inference, Collator_SA_Blosum_ClassII_Extra_Features_Inference, load_blosum, load_pseudoseqs, load_blosum_freq_rownorm


def args_parser():

    parser = argparse.ArgumentParser(description="Inference with one NNAlign_MA-like model.")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size (number of samples per inference step).")
    parser.add_argument("-f", "--data_file", type=str, help="Path to the inference data file.")
    parser.add_argument("-bl", "--blosum_file", type=str, help="Path to the blosum file.")
    parser.add_argument("-ps", "--pseudoseqs_file", type=str, help="Path to the pseudoseqs file.")
    parser.add_argument("-syn", "--synapse_file", type=str, help="Path to the file with model weights.")
    parser.add_argument("-a","--activation", choices=["relu","tanh", "sig"], default="tanh")
    parser.add_argument("-p", "--pred_file", type=str, help="Path to save the predictions.")
    parser.add_argument("-ft", "--extra_features", action="store_true", help="Enable extra peptide-context features (PFR composition, peptide length and PFR length encodings).")
    parser.add_argument("-blf", "--blosum_freq_file", type=str, help="Path to the blosum file freq rownorm.")
    parser.add_argument("-pl", "--peptide_lengths", type=int, nargs=2, default=[12, 19], metavar=("MIN", "MAX"), help="Allowed peptide length range for length encoding.")

    return parser.parse_args()


def main():

    args = args_parser()

    print("\n[INFO] Starting inference script")

    #Setting available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    print(f"[INFO] Using device: {device}")

    #Loading preprocessing resouces
    print("[INFO] Loading BLOSUM matrix...")
    blosum_matrix, aa_to_idx = load_blosum(blosum_file=args.blosum_file)

    print("[INFO] Loading pseudosequences...")
    pseudoseqs_dict = load_pseudoseqs(
        aa_to_idx=aa_to_idx,
        blosum_matrix=blosum_matrix, 
        pseudoseqs_file=args.pseudoseqs_file
    )
    
    #Initialize inference dataset
    print("[INFO] Loading dataset...")
    dataset = NNAlign_MA_Dataset(file_path=args.data_file, min_length=args.peptide_lengths[0])
    print(f"[INFO] Dataset size: {len(dataset)} peptides")

    #Initialize collator for batch construction
    if args.extra_features:
        blosum_matrix_freq, aa_to_idx_freq = load_blosum_freq_rownorm(blosum_file=args.blosum_freq_file)
        collator = Collator_SA_Blosum_ClassII_Extra_Features_Inference(blosum_matrix=blosum_matrix, aa_to_idx=aa_to_idx, pseudoseqs_dict=pseudoseqs_dict, blosum_matrix_freq=blosum_matrix_freq, aa_to_idx_freq=aa_to_idx_freq, min_length=args.peptide_lengths[0], max_length=args.peptide_lengths[1])
    else:
        collator = Collator_SA_Blosum_ClassII_Inference(blosum_matrix=blosum_matrix, aa_to_idx=aa_to_idx, pseudoseqs_dict=pseudoseqs_dict)

    #Initialize dataloader
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=pin_memory
    )

    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Number of batches: {len(loader)}")

    #Activation funtion options
    ACTIVATION_FACTORY = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sig": nn.Sigmoid()
    }
    
    activation = ACTIVATION_FACTORY[args.activation]

    #Initialize model
    print("[INFO] Loading model checkpoint...")
    checkpoint = torch.load(args.synapse_file, map_location=device)

    if args.extra_features:
        model = NNAlign_MA_Extra_Features(n_hidden=checkpoint["n_hidden"],
                                          activation=activation)
    else:
        model = NNAlign_MA(n_hidden=checkpoint["n_hidden"],
                           activation = activation)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully")
    print("[INFO] Starting inference...\n")

    processed = 0

    with torch.no_grad():

        with open(args.pred_file, "w") as outfile:

            for batch_idx, batch in enumerate(loader):

                #move batch tensors to device
                X, y, pep_idx = batch[0]
                X = X.to(device, non_blocking=True)
                pep_idx = pep_idx.to(device, non_blocking=True)

                pep_list, comb_list = batch[1:]

                z_max, idx_max = model.inference(X, pep_idx)

                y = y.tolist()
                z_max = z_max.cpu().tolist()
                idx_max = idx_max.cpu().tolist()

                comb_max_list = [comb_list[i] for i in idx_max]

                for i in range(len(pep_list)):
                    print(
                        f"{pep_list[i]}\t{comb_max_list[i][0]}\t{y[i]}\t{z_max[i]}\t{comb_max_list[i][1]}",
                        file=outfile
                    )

                processed += len(pep_list)

                #Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"[INFO] Batch {batch_idx+1}/{len(loader)} processed | peptides processed: {processed}")

    print("\n[INFO] Inference completed")
    print(f"[INFO] Total peptides processed: {processed}")
    print(f"[INFO] Predictions saved to: {args.pred_file}")


if __name__ == "__main__":
    main()