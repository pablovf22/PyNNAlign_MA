import torch


def load_blosum(blosum_file):
    rows = []
    alphabet = None

    with open(blosum_file) as infile:
        for line in infile:
            line = line.strip()

            # Skip comments/blank lines in the BLOSUM file
            if not line or line.startswith("#"):
                continue

            row = line.split()

            # First non-comment line is the amino-acid alphabet (column headers)
            if alphabet is None:
                alphabet = row
                continue

            # Parse one row of substitution scores
            aa = row[0]
            scores = list(map(float, row[1:]))
            rows.append(scores)

    # Tensor of shape [|AA|, |AA|]
    blosum_matrix = torch.tensor(rows, dtype=torch.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

    return blosum_matrix, aa_to_idx


def load_allelist(allelelist_file):
    allele_dict = {}

    with open(allelelist_file, "r") as infile:
        for line in infile:
            cell_line, alleles = line.strip().split()

            allele_dict[cell_line] = alleles.split(",")

    return allele_dict


def load_pseudoseqs(pseudoseqs_file, aa_to_idx, blosum_matrix):

    pseudoseqs_dict = {}

    with open(pseudoseqs_file, "r") as infile:
        for line in infile:
            allele, pseudoseq = line.strip().split()

            pseudoseq_idx = torch.tensor(
                [aa_to_idx.get(aa, aa_to_idx["X"]) for aa in pseudoseq],
                dtype=torch.long
            )

            # Flattened BLOSUM encoding for fast concatenation
            pseudoseqs_dict[allele] = blosum_matrix[pseudoseq_idx].reshape(-1)

    return pseudoseqs_dict



class Collator_MA_Blosum:
    """
    Base collator implementing the common batching pipeline for MHC peptide data.

    This class defines the general logic to:
    - validate samples,
    - encode peptides into window representations,
    - retrieve allele pseudosequences,
    - build the window–allele Cartesian product,
    - and pack variable-size data into batch tensors.

    Subclasses must implement `_encode_windows`, which defines how peptide
    windows are constructed for a specific MHC class.
    """

    def __init__(self, blosum_matrix, aa_to_idx, allele_dict, pseudoseqs_dict):

        self.allele_dict = allele_dict
        self.pseudoseqs_dict = pseudoseqs_dict
        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx


    def __call__(self, batch):
        X_list, y_list, pep_idx_list = [], [], []
        E = self.blosum_matrix.size(0)

        for pep_i, (peptide, label, cell_line) in enumerate(batch):

            windows_embedding, W = self._encode_windows(peptide, E)
            pseudoseqs_tensor, P = self._get_pseudoseqs(cell_line)

            X = self._combine_window_pseudoseq(
                windows_embedding, pseudoseqs_tensor, W, P
            )
            y, pep_idx = self._make_targets_idx(label, pep_i, W, P)

            X_list.append(X)
            y_list.append(y)
            pep_idx_list.append(pep_idx)

        return self._finalize_batch(X_list, y_list, pep_idx_list)


    def _encode_windows(self, peptide, E):
        """
        Encode a peptide into a set of window embeddings.

        This method must be implemented by subclasses and defines the
        windowing strategy specific to a given MHC class.
        """
        raise NotImplementedError


    def _get_pseudoseqs(self, cell_line):
        """
        Retrieve all available allele pseudosequences for a given cell line.
        """

        alleles = self.allele_dict[cell_line]
        pseudoseqs_list = [
            self.pseudoseqs_dict[allele]
            for allele in alleles
            if allele in self.pseudoseqs_dict
        ]

        pseudoseqs_tensor = torch.stack(pseudoseqs_list, dim=0)
        P = pseudoseqs_tensor.size(0)
        return pseudoseqs_tensor, P


    def _combine_window_pseudoseq(self, windows_embedding, pseudoseqs_tensor, W, P):
        """
        Build the Cartesian product between peptide windows and allele pseudosequences.
        """

        window_rep = windows_embedding.repeat_interleave(P, dim=0)
        pseudoseqs_rep = pseudoseqs_tensor.repeat(W, 1)
        X = torch.cat([window_rep, pseudoseqs_rep], dim=1)
        return X


    def _make_targets_idx(self, label, pep_i, W, P):
        """
        Create target labels and peptide indices aligned with the expanded input.
        """

        y = torch.tensor([float(label)], dtype=torch.float32)
        pep_idx = torch.full((W * P,), pep_i, dtype=torch.long)
        return y, pep_idx


    def _finalize_batch(self, X_list, y_list, pep_idx_list):
        """
        Pack variable-length peptide data into contiguous batch tensors.
        """

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        pep_idx = torch.cat(pep_idx_list, dim=0)

        return X, y, pep_idx



class Collator_MA_Blosum_ClassII(Collator_MA_Blosum):
    """
    Collator for MHC class II peptides.

    Implements sliding-window encoding using overlapping 9-mers,
    as required by the NNAlign-style MHC class II formulation.
    """

    def _encode_windows(self, peptide, E):
        peptide_idxs = torch.tensor(
            [self.aa_to_idx.get(aa, self.aa_to_idx["X"]) for aa in peptide],
            dtype=torch.long
        )

        windows_idxs = peptide_idxs.unfold(0, 9, 1)
        W = windows_idxs.size(0)

        windows_embedding = self.blosum_matrix[windows_idxs].reshape(W, 9 * E)
        return windows_embedding, W
    

class Collator_SA_Blosum_ClassII(Collator_MA_Blosum_ClassII):

    def __init__(self, blosum_matrix, aa_to_idx, pseudoseqs_dict):

        self.pseudoseqs_dict = pseudoseqs_dict
        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx


    def __call__(self, batch):

        X_list, y_list, pep_idx_list = [], [], []
        E = self.blosum_matrix.size(0)

        for pep_i, (peptide, label, allele) in enumerate(batch):

            windows_embedding, W = self._encode_windows(peptide, E)
            pseudoseqs_tensor = self.pseudoseqs_dict[allele].unsqueeze(0) 

            X = self._combine_window_pseudoseq(
                windows_embedding, pseudoseqs_tensor, W, 1
            )
            y, pep_idx = self._make_targets_idx(label, pep_i, W, 1)

            X_list.append(X)
            y_list.append(y)
            pep_idx_list.append(pep_idx)

        return self._finalize_batch(X_list, y_list, pep_idx_list)