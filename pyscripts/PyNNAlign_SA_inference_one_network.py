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

from src.models import NNAlign_MA
from src.datasets import NNAlign_MA_Dataset
from src.datasets_utils import Collator_SA_Blosum_ClassII_Inference, load_blosum, load_pseudoseqs


def args_parser():

    parser = argparse.ArgumentParser(description="Inference with one NNAlign_MA-like model.")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size (number of samples per inference step).")
    parser.add_argument("-tr", "--data_file", type=str, help="Path to the inference data file.")
    parser.add_argument("-bl", "--blosum_file", type=str, help="Path to the blosum file.")
    parser.add_argument("-ps", "--pseudoseqs_file", type=str, help="Path to the pseudoseqs file.")
    parser.add_argument("-syn", "--synapse_file", type=str, help="Path to the file with model weights.")
    parser.add_argument("-nh", "--n_hidden", type=int, default=56, help="Number of hidden neurons.")
    parser.add_argument("-a","--activation", choices=["relu","tanh", "sig"], default="tanh")
    parser.add_argument("-p", "--pred_file", type=str, help="Path to save the predictions.")

    return parser.parse_args()


def main():

    args = args_parser()

    #Setting available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    #Loading preprocessing resouces
    blosum_matrix, aa_to_idx = load_blosum(blosum_file=args.blosum_file)

    pseudoseqs_dict = load_pseudoseqs(aa_to_idx=aa_to_idx,
                                      blosum_matrix=blosum_matrix, 
                                      pseudoseqs_file=args.pseudoseqs_file)
    
    #Initialize inference dataset
    dataset = NNAlign_MA_Dataset(file_path=args.data_file) 

    #Initialize collator for batch construction
    collator = Collator_SA_Blosum_ClassII_Inference(blosum_matrix=blosum_matrix, aa_to_idx=aa_to_idx, pseudoseqs_dict=pseudoseqs_dict)

    #Initialize dataloader
    loader = DataLoader(dataset, 
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collator,
                        pin_memory=pin_memory)
    
    #Activation funtion options
    ACTIVATION_FACTORY = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sig": nn.Sigmoid()}
    
    #Define the activation funtion
    activation = ACTIVATION_FACTORY[args.activation]

    #Initialize model
    model = NNAlign_MA(n_hidden=args.n_hidden,
                       activation = activation)
    
    model.load_state_dict(torch.load(args.synapse_file, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():

        with open(args.pred_file, "w") as outfile:

            for batch in loader:

                #move batch tensors to device
                X, y, pep_idx = [tensor.to(device, non_blocking=True) for tensor in batch[0]]
                pep_list, comb_list = batch[1:]

                z_max, idx_max = model.inference(X, pep_idx)

                y = y.cpu().tolist()
                z_max = z_max.cpu().tolist()
                idx_max = idx_max.cpu().tolist()

                comb_max_list = [comb_list[i] for i in idx_max]

                for i in range(len(pep_list)):
                    print(f"{pep_list[i]}\t{comb_max_list[i][0]}\t{y[i]}\t{z_max[i]}\t{comb_max_list[i][1]}", file=outfile)


if __name__ == "__main__":
    main()