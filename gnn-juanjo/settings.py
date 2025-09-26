# FEATURE = "Q-Positive"
FEATURE = "Volume"

# Path to the dataset CSV file
PATH = "./datasets/knotinfoWithGraph.csv"

# Batch size for the dataloaders
BATCH_SIZE = 128

GENERATE_GRPHS = True

AUG_PATH = "./datasets/augmented_knotinfo.csv"

# Split type for train/test split in dataset.py
SPLIT_TYPE = "random"

# Subset size to limit the dataset for quicker experiments (-1 for full dataset)
SUBSET_SIZE = 9000

# Add aditional knot features from the dataset (True/False)
ADDITIONAL = False

# Options for loss function: "abs" or "squared"
LOSS = "squared"  

# Type of GNN convolutional layer: "GAT", "TRANS", "PNA"
CONV_TYPE = "GAT"