import random
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset


class NNAlign_MA_IterableDataset(IterableDataset):

    """
    Iterable dataset for large-scale pMHC training data.

    Streams a large text file line by line, applies lightweight parsing and yields
    raw examples (peptide, label, cell line). A fixed-size buffer is used to perform
    approximate shuffling without loading the full dataset into memory.
    """

    def __init__(self, file_path, allele_dict, buffer_size=10000, SA_mode = False):

        super(NNAlign_MA_IterableDataset, self).__init__()
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.allele_dict = allele_dict
        self.SA_mode = SA_mode
        

    def __iter__(self):
        
        buffer = []

        with open(self.file_path, "r") as infile:

            # Fill initial buffer
            for line in infile:

                peptide, label, cell_line = line.split()[:3]

                assert cell_line in self.allele_dict, f"Unknown cell_line={cell_line}"
                if self.SA_mode and len(self.allele_dict[cell_line]) != 1:
                    continue

                buffer.append((peptide, int(label), cell_line))
                if len(buffer) >= self.buffer_size:
                    break   

            # Reservoir-style approximate shuffling
            for line in infile:

                peptide, label, cell_line = line.split()[:3]
                
                assert cell_line in self.allele_dict, f"Unknown cell_line={cell_line}"
                if self.SA_mode and len(self.allele_dict[cell_line]) != 1:
                    continue

                idx = random.randint(0, len(buffer)-1)
                yield buffer[idx]  # yield random element from buffer

                buffer[idx] = (peptide, int(label), cell_line)

            random.shuffle(buffer)  # shuffle remaining elements
            while buffer:
                yield buffer.pop()


class NNAlign_MA_OffsetDataset(Dataset):
    """
    Map-style dataset backed by a large text file using byte offsets.

    Offsets are computed once at initialization, allowing random access
    via seek() without loading the full file into memory. Optionally
    filters single-allele (SA) entries during indexing.
    """

    def __init__(self, file_path, allele_dict, SA_mode=False):

        super(NNAlign_MA_OffsetDataset, self).__init__()

        self.file_path = file_path
        self.offsets = []          # Byte offsets of valid line starts
        self.infile = None         # Lazily opened file handle

        with open(self.file_path, "rb") as infile:

            offset = 0  # Current byte position

            for line in infile:

                cell_line = line.decode("utf-8").strip().split()[2]
                assert cell_line in allele_dict, \
                    f"Unknown cell_line={cell_line}"

                # Skip non-SA entries if requested
                if SA_mode and len(allele_dict[cell_line]) != 1:
                    offset += len(line)
                    continue

                self.offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):

        # Open file lazily (one handle per process)
        if self.infile is None:
            self.infile = open(self.file_path, "rb")

        # Jump to stored offset and read one line
        self.infile.seek(self.offsets[idx])
        line = self.infile.readline().decode("utf-8").strip()

        peptide, label, cell_line = line.split()[:3]
        return peptide, int(label), cell_line

    def __del__(self):
        infile = getattr(self, "infile", None)
        try:
            if infile is not None and not infile.closed:
                infile.close()
        except Exception:
            pass


class NNAlign_MA_Dataset(Dataset):
    """
    In-memory dataset for NNAlign MA training.

    Loads the entire text file into memory as (peptide, label, cell_line)
    tuples.
    """

    def __init__(self, file_path, min_length):

        super(NNAlign_MA_Dataset, self).__init__()

        self.dataset = []  # Stores all samples in memory
        self.min_length = min_length

        # Load and optionally filter samples
        with open(file_path, "r") as infile:

            for line in infile:

                # Parse first three whitespace-separated fields
                peptide, label, cell_line = line.strip().split()[:3]

                assert len(peptide) >= self.min_length, \
                    f"Peptide '{peptide}' has length {len(peptide)}, which is shorter than the minimum allowed ({self.min_length})."

                # Store sample as (peptide, int label, cell_line)
                self.dataset.append((peptide, int(float(label)), cell_line))

    def __len__(self):
        """Return number of loaded samples."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return sample at given index."""
        return self.dataset[idx]
    

class NNAlign_SA_Dataset_ClassII_Blosum_Encoded(Dataset):
    """
    PyTorch Dataset for NNAlign-style SA peptide–MHC data.

    Each peptide generates all possible 9-mer cores, encoded with BLOSUM.
    Returns per peptide:
        - encoded candidate cores
        - label
        - allele
        - mask indicating hydrophobic residue at P1 for each core
    """

    def __init__(self, file_path, min_length, blosum_matrix, aa_to_idx, pseudoseqs_dict):
        """
        Load dataset, count cores, allocate tensors, and encode peptide windows.
        """

        super(NNAlign_SA_Dataset_ClassII_Blosum_Encoded, self).__init__()

        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx
        embedding_dim = self.blosum_matrix.shape[1]
        core_length = 9
        hydrophobic_aa = {"I", "L", "V", "M", "F", "Y", "W"}

        # First pass: count peptides and candidate cores
        with open(file_path, "r") as infile:

            total_cores = 0
            total_peptides = 0
            n_cores_list = []

            for i, line in enumerate(infile, start=1):
                fields = line.strip().split()
                peptide, label, allele = fields[:3]

                # Validate peptide length
                assert len(peptide) >= min_length, \
                    f"Peptide '{peptide}' (length {len(peptide)}) shorter than min_length {min_length}"

                # Validate allele exists
                assert allele in pseudoseqs_dict, (
                    f"{file_path} line {i}: allele '{allele}' not found in pseudoseqs file"
                )

                n_cores = len(peptide) - 9 + 1
                total_peptides += 1
                total_cores += n_cores
                n_cores_list.append(n_cores)

        # Allocate tensors
        self.windows = torch.empty((total_cores, embedding_dim * core_length), dtype=torch.float32)
        self.offsets = torch.empty(total_peptides + 1, dtype=torch.long)
        self.y = torch.empty(total_peptides, dtype=torch.float32)
        self.hydrophobic_p1 = torch.empty(total_cores, dtype=torch.bool)
        self.alleles = []

        # Compute offsets mapping peptides → core slices
        self.offsets[0] = 0
        self.offsets[1:] = torch.cumsum(torch.tensor(n_cores_list, dtype=torch.long), dim=0)

        # Second pass: encode windows and fill tensors
        with open(file_path, "r") as infile:

            pep_idx = 0
            core_idx = 0

            for line in infile:
                peptide, label, allele = line.strip().split()[:3]

                self.y[pep_idx] = float(label)

                # Encode all 9-mer windows
                windows_embedding = self._encode_windows(peptide, embedding_dim)
                n_cores = windows_embedding.shape[0]

                self.windows[core_idx:core_idx + n_cores] = windows_embedding
                self.alleles.append(allele)

                # Hydrophobic check at P1 (first residue of each window)
                for i in range(len(peptide) - 9 + 1):
                    self.hydrophobic_p1[core_idx + i] = peptide[i] in hydrophobic_aa

                core_idx += n_cores
                pep_idx += 1

    def __len__(self):
        """Number of peptides."""
        return self.y.shape[0]

    def __getitem__(self, idx):
        """Return all candidate cores for one peptide."""

        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        return {
            "windows": self.windows[start:end],
            "label": self.y[idx],
            "allele": self.alleles[idx],
            "hydrophobic_p1_mask": self.hydrophobic_p1[start:end]
        }

    def _encode_windows(self, peptide, E):
        """Encode all overlapping 9-mer windows using BLOSUM."""

        # Convert peptide to BLOSUM indices
        peptide_idxs = torch.tensor(
            [self.aa_to_idx.get(aa, self.aa_to_idx["X"]) for aa in peptide],
            dtype=torch.long
        )

        # Generate 9-mer windows
        windows_idxs = peptide_idxs.unfold(0, 9, 1)
        W = windows_idxs.size(0)

        # Flatten BLOSUM embeddings
        windows_embedding = self.blosum_matrix[windows_idxs].reshape(W, 9 * E)
        return windows_embedding
    

class NNAlign_SA_Dataset_ClassII_Blosum_Encoded_Extra_Features(Dataset):
    """
    PyTorch Dataset for NNAlign-style SA peptide–MHC data.

    Each peptide generates all possible 9-mer cores, encoded with BLOSUM.
    Returns per peptide:
        - encoded candidate cores
        - label
        - encoded pseudosequence
        - mask indicating hydrophobic residue at P1 for each core
    """

    def __init__(self, file_path, min_length, blosum_matrix, aa_to_idx, pseudoseqs_dict, blosum_matrix_freq, aa_to_idx_freq, max_length=19, pfr_length=3):
        """
        Load dataset, count cores, allocate tensors, and encode peptide windows.
        """

        super(NNAlign_SA_Dataset_ClassII_Blosum_Encoded_Extra_Features, self).__init__()

        self.blosum_matrix = blosum_matrix
        self.aa_to_idx = aa_to_idx
        self.blosum_matrix_freq = blosum_matrix_freq
        self.aa_to_idx_freq = aa_to_idx_freq
        embedding_dim = self.blosum_matrix.shape[1]
        pfrs_encoding_dim = self.blosum_matrix_freq.shape[1]
        peptide_length_encoding_dim = max_length - min_length + 1
        pfrs_length_encoding_dim = 2
        core_length = 9
        hydrophobic_aa = {"I", "L", "V", "M", "F", "Y", "W"}
        self.pfr_length = pfr_length
        self.min_length = min_length
        self.max_length = max_length

        # First pass: count peptides and candidate cores
        with open(file_path, "r") as infile:

            total_cores = 0
            total_peptides = 0
            n_cores_list = []

            for i, line in enumerate(infile, start=1):
                fields = line.strip().split()
                peptide, label, allele = fields[:3]

                # Validate peptide length
                assert len(peptide) >= min_length, \
                    f"Peptide '{peptide}' (length {len(peptide)}) shorter than min_length {min_length}"

                # Validate allele exists
                assert allele in pseudoseqs_dict, (
                    f"{file_path} line {i}: allele '{allele}' not found in pseudoseqs file"
                )

                n_cores = len(peptide) - 9 + 1
                total_peptides += 1
                total_cores += n_cores
                n_cores_list.append(n_cores)

        # Allocate tensors
        self.windows = torch.empty((total_cores, (embedding_dim * core_length) + peptide_length_encoding_dim + 2 * pfrs_encoding_dim + 2 * pfrs_length_encoding_dim), dtype=torch.float32)
        self.offsets = torch.empty(total_peptides + 1, dtype=torch.long)
        self.y = torch.empty(total_peptides, dtype=torch.float32)
        self.hydrophobic_p1 = torch.empty(total_cores, dtype=torch.bool)
        self.alleles = []

        # Compute offsets mapping peptides → core slices
        self.offsets[0] = 0
        self.offsets[1:] = torch.cumsum(torch.tensor(n_cores_list, dtype=torch.long), dim=0)

        # Second pass: encode windows and fill tensors
        with open(file_path, "r") as infile:

            pep_idx = 0
            core_idx = 0

            for line in infile:
                peptide, label, allele = line.strip().split()[:3]

                self.y[pep_idx] = float(label)

                # Encode all 9-mer windows
                windows_embedding = self._encode_windows(peptide, embedding_dim)
                n_cores = windows_embedding.shape[0]
                pfrs_embedding = self._encode_pfrs(peptide, n_cores)
                pfrs_length_encoding = self._encode_pfrs_length(peptide, n_cores)
                peptide_length_encoding = self._encode_peptide_length(peptide, n_cores)

                core_end = embedding_dim * core_length
                pfr_end = core_end + 2 * pfrs_encoding_dim
                pfr_len_end = pfr_end + 2 * pfrs_length_encoding_dim

                self.windows[core_idx:core_idx + n_cores, :core_end] = windows_embedding
                self.windows[core_idx:core_idx + n_cores, core_end:pfr_end] = pfrs_embedding
                self.windows[core_idx:core_idx + n_cores, pfr_end:pfr_len_end] = pfrs_length_encoding
                self.windows[core_idx:core_idx + n_cores, pfr_len_end:] = peptide_length_encoding
                self.alleles.append(allele)

                # Hydrophobic check at P1 (first residue of each window)
                for i in range(len(peptide) - 9 + 1):
                    self.hydrophobic_p1[core_idx + i] = peptide[i] in hydrophobic_aa

                core_idx += n_cores
                pep_idx += 1

    def __len__(self):
        """Number of peptides."""
        return self.y.shape[0]

    def __getitem__(self, idx):
        """Return all candidate cores for one peptide."""

        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        return {
            "windows": self.windows[start:end],
            "label": self.y[idx],
            "allele": self.alleles[idx],
            "hydrophobic_p1_mask": self.hydrophobic_p1[start:end]
        }

    def _encode_windows(self, peptide, E):
        """Encode all overlapping 9-mer windows using BLOSUM."""

        # Convert peptide to BLOSUM indices
        peptide_idxs = torch.tensor(
            [self.aa_to_idx.get(aa, self.aa_to_idx["X"]) for aa in peptide],
            dtype=torch.long
        )

        # Generate 9-mer windows
        windows_idxs = peptide_idxs.unfold(0, 9, 1)
        W = windows_idxs.size(0)

        # Flatten BLOSUM embeddings
        windows_embedding = self.blosum_matrix[windows_idxs].reshape(W, 9 * E)
        return windows_embedding
    
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
    
    def _encode_peptide_length(self, peptide, W):
        """Encode peptide length as soft one-hot bins."""
        n_bins = self.max_length - self.min_length + 1
        peptide_length_encoding = [0.05] * n_bins
        i = min(len(peptide), self.max_length) - self.min_length

        peptide_length_encoding[i] = 0.9
        peptide_length_encoding = torch.tensor(peptide_length_encoding, dtype=torch.float32).repeat(W, 1)

        return peptide_length_encoding