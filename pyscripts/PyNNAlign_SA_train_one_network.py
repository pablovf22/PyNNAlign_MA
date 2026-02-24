import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

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