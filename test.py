import dask
import dask.array as da
from dask.distributed import Client
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

# Initialize Dask Client
client = Client()

# Load dataset using Hugging Face
data = load_dataset("imdb")

# Tokenizer and Model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Freeze model weights for gradient calculation
for param in model.base_model.parameters():
    param.requires_grad = False

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_train = data["train"].map(tokenize_function, batched=True)
encoded_test = data["test"].map(tokenize_function, batched=True)

# Prepare PyTorch DataLoaders
train_loader = DataLoader(encoded_train, batch_size=8, shuffle=True)
test_loader = DataLoader(encoded_test, batch_size=8)

# Function to compute gradients
def compute_gradients(batch, model):
    inputs = torch.tensor(batch["input_ids"])
    attention_mask = torch.tensor(batch["attention_mask"])
    labels = torch.tensor(batch["label"])

    # Forward pass
    outputs = model(inputs, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Collect gradients
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients.append(param.grad.view(-1))

    return torch.cat(gradients)

# Dask-friendly gradient computation wrapper
def compute_gradients_dask(data_loader, model):
    gradients = []
    for batch in data_loader:
        gradients.append(compute_gradients(batch, model))
    return torch.stack(gradients)

# Distribute gradient computations with Dask
train_gradients = dask.delayed(compute_gradients_dask)(train_loader, model)
test_gradients = dask.delayed(compute_gradients_dask)(test_loader, model)

# Compute the dot product of gradients
train_gradients_da = da.from_delayed(train_gradients, dtype=float, shape=(None,))
test_gradients_da = da.from_delayed(test_gradients, dtype=float, shape=(None,))

dot_product = da.dot(train_gradients_da, test_gradients_da.T)

# Compute the result
def main():
    result = dot_product.compute()
    print("Dot Product of Gradients:", result)

if __name__ == "__main__":
    main()
