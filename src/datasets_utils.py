import torch


def load_blosum(blosum_file):
    """Load a BLOSUM substitution matrix and its amino acid index mapping."""
    rows = []
    alphabet = None

    with open(blosum_file) as infile:
        for line in infile:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            row = line.split()

            # First valid line contains the amino acid alphabet
            if alphabet is None:
                alphabet = row
                continue

            # Parse one matrix row
            aa = row[0]
            scores = list(map(float, row[1:]))
            rows.append(scores)

    # Tensor of shape [n_aa, n_aa]
    blosum_matrix = torch.tensor(rows, dtype=torch.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

    return blosum_matrix, aa_to_idx


def load_blosum_freq_rownorm(blosum_file):
    """Load row-normalized BLOSUM frequencies used for PFR encoding."""
    rows = []
    alphabet = None

    with open(blosum_file) as infile:
        for line in infile:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            row = line.split()

            # First valid line contains the amino acid alphabet
            if alphabet is None:
                alphabet = ["X"] + row
                rows.append([0] * len(row))
                continue

            # Parse one matrix row
            scores = list(map(float, row))
            rows.append(scores)

    # Tensor of shape [n_aa_with_X, n_aa]
    blosum_matrix_freq = torch.tensor(rows, dtype=torch.float32)
    aa_to_idx_freq = {aa: i for i, aa in enumerate(alphabet)}

    return blosum_matrix_freq, aa_to_idx_freq


def load_allelist(allelelist_file):
    """Load mapping from cell line identifiers to their allele lists."""
    allele_dict = {}

    with open(allelelist_file, "r") as infile:
        for line in infile:
            cell_line, alleles = line.strip().split()

            allele_dict[cell_line] = alleles.split(",")

    return allele_dict


def load_pseudoseqs(pseudoseqs_file, aa_to_idx, blosum_matrix):
    """Load and BLOSUM-encode MHC pseudosequences."""
    pseudoseqs_dict = {}

    with open(pseudoseqs_file, "r") as infile:
        for line in infile:
            allele, pseudoseq = line.strip().split()

            pseudoseq_idx = torch.tensor(
                [aa_to_idx.get(aa, aa_to_idx["X"]) for aa in pseudoseq],
                dtype=torch.long
            )

            # Flatten encoded pseudosequence for direct concatenation
            pseudoseqs_dict[allele] = blosum_matrix[pseudoseq_idx].reshape(-1)

    return pseudoseqs_dict


class Collator_MA_Blosum:
    """Base multi-allele collator using BLOSUM-encoded peptide windows."""

    def __init__(self, blosum_matrix, aa_to_idx, allele_dict, pseudoseqs_dict):

        self.allele_dict = allele_dict
        self.pseudoseqs_dict = pseudoseqs_dict
        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx


    def __call__(self, batch):
        """Build model inputs for a batch of multi-allele peptides."""
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
        """Encode peptide candidate windows. Implemented by subclasses."""
        raise NotImplementedError


    def _get_pseudoseqs(self, cell_line):
        """Retrieve all encoded pseudosequences for a given cell line."""
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
        """Construct the Cartesian product between windows and pseudosequences."""
        window_rep = windows_embedding.repeat_interleave(P, dim=0)
        pseudoseqs_rep = pseudoseqs_tensor.repeat(W, 1)
        X = torch.cat([window_rep, pseudoseqs_rep], dim=1)
        return X


    def _make_targets_idx(self, label, pep_i, W, P):
        """Create peptide-level target and grouping indices."""
        y = torch.tensor([float(label)], dtype=torch.float32)
        pep_idx = torch.full((W * P,), pep_i, dtype=torch.long)
        return y, pep_idx


    def _finalize_batch(self, X_list, y_list, pep_idx_list):
        """Concatenate all peptide-specific tensors into a batch."""
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        pep_idx = torch.cat(pep_idx_list, dim=0)

        return X, y, pep_idx


class Collator_MA_Blosum_ClassII(Collator_MA_Blosum):
    """Multi-allele class II collator using sliding 9-mer cores."""

    def _encode_windows(self, peptide, E):
        # Encode the full peptide sequence as amino acid indices
        peptide_idxs = torch.tensor(
            [self.aa_to_idx.get(aa, self.aa_to_idx["X"]) for aa in peptide],
            dtype=torch.long
        )

        # Generate all overlapping 9-mer candidate cores
        windows_idxs = peptide_idxs.unfold(0, 9, 1)
        W = windows_idxs.size(0)

        # Flatten BLOSUM-encoded windows
        windows_embedding = self.blosum_matrix[windows_idxs].reshape(W, 9 * E)
        return windows_embedding, W
    

class Collator_SA_Blosum_ClassII(Collator_MA_Blosum_ClassII):
    """Single-allele class II collator using BLOSUM-encoded 9-mer windows."""

    def __init__(self, blosum_matrix, aa_to_idx, pseudoseqs_dict):

        self.pseudoseqs_dict = pseudoseqs_dict
        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx


    def __call__(self, batch):
        """Build model inputs for a batch of single-allele peptides."""
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
    

class Collator_SA_Blosum_ClassII_Inference(Collator_SA_Blosum_ClassII):
    """Inference-time single-allele collator returning raw core-allele pairs."""
    
    def __call__(self, batch):
        """Build inference inputs and keep track of raw peptide combinations."""
        X_list, y_list, pep_idx_list, pep_list, comb_list = [], [], [], [], []
        E = self.blosum_matrix.size(0)

        for pep_i, (peptide, label, allele) in enumerate(batch):

            windows_embedding, W = self._encode_windows(peptide, E)
            pseudoseqs_tensor = self.pseudoseqs_dict[allele].unsqueeze(0)

            X = self._combine_window_pseudoseq(
                windows_embedding, pseudoseqs_tensor, W, 1
            )
            y, pep_idx = self._make_targets_idx(label, pep_i, W, 1)
            pep_comb = self._get_raw_combinations(peptide, allele)

            X_list.append(X)
            y_list.append(y)
            pep_idx_list.append(pep_idx)
            pep_list.append(peptide)
            comb_list.extend(pep_comb)

        return self._finalize_batch(X_list, y_list, pep_idx_list), pep_list, comb_list
    

    def _get_raw_combinations(self, peptide, allele):
        """Return all candidate 9-mer core and allele combinations."""
        comb_list = []

        for i in range(len(peptide)-9+1):

            core = peptide[i:i+9]
            comb_list.append([core, allele])

        return comb_list
    

class Collator_SA_Blosum_ClassII_Extra_Features(Collator_SA_Blosum_ClassII):
    """Single-allele class II collator with additional peptide-context features."""

    def __init__(self, blosum_matrix, aa_to_idx, pseudoseqs_dict, blosum_matrix_freq, aa_to_idx_freq, pfr_length=3, min_length=12, max_length=19):

        self.pseudoseqs_dict = pseudoseqs_dict
        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx
        self.blosum_matrix_freq = blosum_matrix_freq
        self.aa_to_idx_freq = aa_to_idx_freq
        self.pfr_length = pfr_length
        self.min_length = min_length
        self.max_length = max_length


    def __call__(self, batch):
        """Build single-allele inputs including PFR and length features."""
        X_list, y_list, pep_idx_list = [], [], []
        E = self.blosum_matrix.size(0)

        for pep_i, (peptide, label, allele) in enumerate(batch):

            windows_embedding, W = self._encode_windows(peptide, E)
            pseudoseqs_tensor = self.pseudoseqs_dict[allele].unsqueeze(0)

            # Build extra peptide-side features
            pfrs_embedding = self._encode_pfrs(peptide, W)
            peptide_length_encoding = self._encode_peptide_length(peptide, W)
            pfrs_length_encoding = self._encode_pfrs_length(peptide, W)

            # Concatenate core, PFR, and length features
            peptide_embedding = torch.cat([windows_embedding, pfrs_embedding, peptide_length_encoding, pfrs_length_encoding], dim=1)

            X = self._combine_window_pseudoseq(
                peptide_embedding, pseudoseqs_tensor, W, 1
            )

            y, pep_idx = self._make_targets_idx(label, pep_i, W, 1)

            X_list.append(X)
            y_list.append(y)
            pep_idx_list.append(pep_idx)

        return self._finalize_batch(X_list, y_list, pep_idx_list)
    

    def _encode_pfrs(self, peptide, W):
        """Encode left and right peptide flanking regions using averaged row-normalized BLOSUM vectors."""
        peptide_padded = "X" * self.pfr_length  + peptide + "X" * self.pfr_length

        peptide_idxs = torch.tensor(
            [self.aa_to_idx_freq.get(aa, self.aa_to_idx_freq["X"]) for aa in peptide_padded],
            dtype=torch.long
        )

        # Build all candidate PFR windows of fixed length
        pfrs_idxs = peptide_idxs.unfold(0, self.pfr_length, 1)

        left_pfrs_idxs = pfrs_idxs[:W]
        right_pfrs_idxs = pfrs_idxs[9+self.pfr_length:9+self.pfr_length+W]

        # Average row-normalized BLOSUM vectors across each PFR
        left_pfrs = self.blosum_matrix_freq[left_pfrs_idxs].mean(dim=1)
        right_pfrs = self.blosum_matrix_freq[right_pfrs_idxs].mean(dim=1)

        pfrs = torch.cat([left_pfrs, right_pfrs], dim=1)

        return pfrs
    

    def _encode_peptide_length(self, peptide, W):
        """Encode peptide length as soft one-hot bins."""
        n_bins = self.max_length - self.min_length + 1
        peptide_length_encoding = [0.05] * n_bins
        i = min(len(peptide), self.max_length) - self.min_length

        peptide_length_encoding[i] = 0.9
        peptide_length_encoding = torch.tensor(peptide_length_encoding, dtype=torch.float32).repeat(W, 1)

        return peptide_length_encoding
    

    def _encode_pfrs_length(self, peptide, W):
        """Encode left and right PFR lengths as four normalized features."""
        L = len(peptide)

        # Offsets correspond to candidate core start positions
        offsets = torch.arange(W, dtype=torch.float32)

        ll = offsets
        lr = L - (offsets + 9)

        # Clip PFR lengths to the maximum encoded flank length
        ll = torch.clamp(ll, max=self.pfr_length)
        lr = torch.clamp(lr, max=self.pfr_length)

        o_left = (self.pfr_length - ll) / self.pfr_length
        o_right = (self.pfr_length - lr) / self.pfr_length

        pfrs_length_encoding = torch.stack(
            [o_left, 1 - o_left, o_right, 1 - o_right],
            dim=1
        )

        return pfrs_length_encoding
    

class Collator_SA_Blosum_ClassII_Extra_Features_Inference(Collator_SA_Blosum_ClassII_Extra_Features):
    """Inference-time version of the extra-features single-allele collator."""


    def __call__(self, batch):
        """Build inference inputs and return raw core-allele combinations."""
        X_list, y_list, pep_idx_list, pep_list, comb_list = [], [], [], [], []
        E = self.blosum_matrix.size(0)

        for pep_i, (peptide, label, allele) in enumerate(batch):

            windows_embedding, W = self._encode_windows(peptide, E)
            pseudoseqs_tensor = self.pseudoseqs_dict[allele].unsqueeze(0)

            # Build extra peptide-side features
            pfrs_embedding = self._encode_pfrs(peptide, W)
            peptide_length_encoding = self._encode_peptide_length(peptide, W)
            pfrs_length_encoding = self._encode_pfrs_length(peptide, W)

            peptide_embedding = torch.cat([windows_embedding, pfrs_embedding, peptide_length_encoding, pfrs_length_encoding], dim=1)

            X = self._combine_window_pseudoseq(
                peptide_embedding, pseudoseqs_tensor, W, 1
            )

            y, pep_idx = self._make_targets_idx(label, pep_i, W, 1)
            pep_comb = self._get_raw_combinations(peptide, allele)

            X_list.append(X)
            y_list.append(y)
            pep_idx_list.append(pep_idx)
            pep_list.append(peptide)
            comb_list.extend(pep_comb)

        return self._finalize_batch(X_list, y_list, pep_idx_list), pep_list, comb_list


    def _get_raw_combinations(self, peptide, allele):
        """Return all candidate 9-mer core and allele combinations."""
        comb_list = []

        for i in range(len(peptide)-9+1):

            core = peptide[i:i+9]
            comb_list.append([core, allele])

        return comb_list
    

class Collator_SA_Blosum_ClassII_Encoded:

    """
    Collator that builds core-level batches by concatenating peptide windows
    with their corresponding MHC pseudosequence. Handles variable numbers of
    candidate cores per peptide and returns (X, y, pep_idx).
    """

    def __init__(self, pseudoseqs_dict):
        self.pseudoseqs_dict = pseudoseqs_dict
        self.use_hydrofobic_mask = None


    def __call__(self, batch):

        # First pass: count total windows and valid peptides
        total_windows = 0
        valid_points = []

        for point in batch:

            windows = point["windows"]
            hydrophobic_p1_mask = point["hydrophobic_p1_mask"]

            if self.use_hydrofobic_mask:
                windows = windows[hydrophobic_p1_mask]

            W = windows.size(0)

            if W < 1:
                continue

            valid_points.append((point, windows))
            total_windows += W

        if total_windows == 0:
            return None

        n_peptides = len(valid_points)

        # Determine dimensions
        window_dim = valid_points[0][1].size(1)
        pseudoseq_dim = self.pseudoseqs_dict[valid_points[0][0]["allele"]].size(0)
        input_dim = window_dim + pseudoseq_dim

        # Preallocate batch tensors
        X = torch.empty((total_windows, input_dim), dtype=valid_points[0][1].dtype)
        pep_idx = torch.empty((total_windows,), dtype=torch.long)
        y = torch.empty((n_peptides,), dtype=torch.float32)

        cursor = 0

        # Second pass: fill tensors
        for i, (point, windows) in enumerate(valid_points):

            W = windows.size(0)

            allele = point["allele"]
            pseudoseq = self.pseudoseqs_dict[allele]
            label = point["label"]

            # Fill windows part
            X[cursor:cursor+W, :window_dim] = windows

            # Fill pseudosequence part (broadcast)
            X[cursor:cursor+W, window_dim:] = pseudoseq

            # Fill peptide indices
            pep_idx[cursor:cursor+W] = i

            # Fill label
            y[i] = label

            cursor += W

        return X, y, pep_idx