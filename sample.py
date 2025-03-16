
from datasets import load_dataset

# Load the dataset
ds = load_dataset("IraGia/gprMax_Train")

# Print dataset structure
print(ds)

# Show the first few examples
print(ds["train"].select(range(5)))
