import torch
import torch.nn as nn

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


    def __init__(self, n_hidden=56, aa_embedding_dim=24, window_size=9, pseudoseq_size=34):

        super(NNAlign_MA, self).__init__()
        self.aa_embedding_dim = aa_embedding_dim
        self.window_size = window_size                    #default MHC class II ligands 
        self.pseudoseq_size = pseudoseq_size
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear((self.window_size + self.pseudoseq_size) * self.aa_embedding_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.activation = nn.Tanh()


    def forward(self, X_tensor, group, batch_size=None):

        if group.dtype != torch.long:
            group = group.long()

        if batch_size is None:
            batch_size = int(group.max().item()) + 1

        z = self.in_layer(X_tensor)
        z = self.activation(z)
        z = self.out_layer(z).squeeze(-1)       #From a 2d tensor to a 1d tensor, the embedding results in a loggit
        z = torch.sigmoid(z)

        #Create a batch size vector and then fill it with the maximum loggit in each group (peptide)
        z_max = torch.full((batch_size,), float("-inf"), device=z.device, dtype=z.dtype) 
        z_max = torch.scatter_reduce(z_max, 0, group, z, reduce="amax", include_self=True)
        
        return z_max   