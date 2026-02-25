import random
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

    def __init__(self, file_path):

        super(NNAlign_MA_Dataset, self).__init__()

        self.dataset = []  # Stores all samples in memory

        # Load and optionally filter samples
        with open(file_path, "r") as infile:

            for line in infile:

                # Parse first three whitespace-separated fields
                peptide, label, cell_line = line.strip().split()[:3]

                # Store sample as (peptide, int label, cell_line)
                self.dataset.append((peptide, int(float(label)), cell_line))

    def __len__(self):
        """Return number of loaded samples."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return sample at given index."""
        return self.dataset[idx]
