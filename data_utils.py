from datasets import load_dataset

# Load the computational thinking dataset by default
dataset = load_dataset("json", data_files="dataset_computational_thinking_python.json")

dataset = dataset["train"].train_test_split(test_size=0.2)

train_dataset = dataset["train"]

temp = dataset["test"].train_test_split(test_size=0.5)

validation_dataset = temp["train"]

test_dataset = temp["test"]
