import torch
import torch.nn as nn
from torch_scatter import scatter_max

class NNAlign_MA(nn.Module):

    """
    NNAlign-style neural network for pMHC interaction modeling with
    hard max pooling over candidate peptide windows and allele representations.

    All valid (window, allele) combinations are scored independently by a shared
    feed-forward network. For each peptide, the highest-scoring combination is
    selected via hard max pooling and only its score contributes to the loss. The
    model outputs a single real-valued score per peptide, which can be used either
    for eluted ligand prediction (classification) or binding affinity prediction
    (regression), depending on the chosen loss function.

    The same architecture supports single-allele and multi-allele without 
    architectural changes.
    """


    def __init__(self, activation, n_hidden=56, aa_embedding_dim=24, window_size=9, pseudoseq_size=34):

        super(NNAlign_MA, self).__init__()
        self.aa_embedding_dim = aa_embedding_dim
        self.window_size = window_size                    #default MHC class II ligands 
        self.pseudoseq_size = pseudoseq_size
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear((self.window_size + self.pseudoseq_size) * self.aa_embedding_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.activation = activation


    def forward(self, X_tensor, group):

        if group.dtype != torch.long:
            group = group.long()

        z = self.in_layer(X_tensor)
        z = self.activation(z)
        z = self.out_layer(z).squeeze(-1)       #From a 2d tensor to a 1d tensor, the embedding results in a loggit
        z = torch.sigmoid(z)

        z_max, idx_max = scatter_max(z, group)  #Per group max pooling
        
        return z_max 


    def inference(self, X_tensor, group):

        if group.dtype != torch.long:
            group = group.long()

        z = self.in_layer(X_tensor)
        z = self.activation(z)
        z = self.out_layer(z).squeeze(-1)       #From a 2d tensor to a 1d tensor, the embedding results in a loggit
        z = torch.sigmoid(z)

        z_max, idx_max = scatter_max(z, group)  #Per group max pooling
        
        return z_max, idx_max
    

class NNAlign_MA_Extra_Features(nn.Module):

    """
    NNAlign-style neural network for peptide–MHC interaction prediction with
    hard max pooling over candidate peptide binding cores and allele representations.

    In addition to core and MHC pseudosequence encodings, the model incorporates
    extra peptide-context features, including peptide flanking region (PFR)
    composition, peptide length encoding, and PFR length encoding.

    All (core, allele) combinations are scored independently, and the highest
    score per peptide is selected via hard max pooling.
    """


    def __init__(self, activation, n_hidden=56, aa_embedding_dim=24, window_size=9, pseudoseq_size=34, pfr_embedding_dim=20, peptide_length_encoding_dim=8, pfr_length_encoding_dim=4):

        super().__init__()
        self.aa_embedding_dim = aa_embedding_dim
        self.window_size = window_size                    #default MHC class II ligands 
        self.pseudoseq_size = pseudoseq_size
        self.n_hidden = n_hidden
        self.pfr_embedding_dim = pfr_embedding_dim
        self.in_dim = ((self.window_size + self.pseudoseq_size) * self.aa_embedding_dim) + (self.pfr_embedding_dim * 2) + peptide_length_encoding_dim + pfr_length_encoding_dim 
        self.in_layer = nn.Linear(self.in_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.activation = activation


    def forward(self, X_tensor, group):

        if group.dtype != torch.long:
            group = group.long()

        z = self.in_layer(X_tensor)
        z = self.activation(z)
        z = self.out_layer(z).squeeze(-1)       #From a 2d tensor to a 1d tensor, the embedding results in a loggit
        z = torch.sigmoid(z)

        z_max, idx_max = scatter_max(z, group)  #Per group max pooling
        
        return z_max 


    def inference(self, X_tensor, group):

        if group.dtype != torch.long:
            group = group.long()

        z = self.in_layer(X_tensor)
        z = self.activation(z)
        z = self.out_layer(z).squeeze(-1)       #From a 2d tensor to a 1d tensor, the embedding results in a loggit
        z = torch.sigmoid(z)

        z_max, idx_max = scatter_max(z, group)  #Per group max pooling
        
        return z_max, idx_max