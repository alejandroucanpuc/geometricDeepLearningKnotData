# Define random seed
SEED = 31

# FEATURE = "Q-Positive"
FEATURE = "Volume"

# Path to the CSV train file
TRAIN_PATH = "./datasets/knotInfo_augmented_30.csv"

# Path to the CSV test file
TEST_PATH = "./datasets/linkDataset_augmented.csv"


# Batch size for the dataloaders
BATCH_SIZE = 128

# Define si se van a generar los grafos para los datasets. "False" solo si estos ya se precalcularon.
GENERATE_GRPHS = True

# Split type for train/test split in dataset.py
SPLIT_TYPE = "random"

# Subset size to limit the dataset for quicker experiments (-1 for full dataset)
SUBSET_SIZE = -1

# Add aditional knot features from the dataset (True/False)
ADDITIONAL = False

# Options for loss function: "abs" or "squared" or "percentage"
LOSS = "percentage"  

# Type of GNN convolutional layer: "GAT", "TRANS", "PNA"
CONV_TYPE = "GAT"

# Define si se va a agregar el atributo de distancia los edges del grafo o no.
USE_DISTANCE_ATTRIBUTE = False