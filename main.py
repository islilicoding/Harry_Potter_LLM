# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.onnx
import random
import torch.optim as optim
from tqdm import tqdm

# Import custom dataset and model classes
from data import MyDataset
from transformer import TransformerModel
from torch.utils.data import DataLoader

# Function to set a random seed for reproducibility
def set_seed(seed):
    random.seed(seed)                 # Seed for Python's random module
    torch.manual_seed(seed)           # Seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)      # Seed for PyTorch on GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior with cuDNN
    torch.backends.cudnn.benchmark = False     # Disables cuDNN benchmark for reproducibility

# Set the random seed
set_seed(42)

###############################################################################
# Load data
###############################################################################

# Create the dataset and dataloader
dataset = MyDataset("harry_potter.txt")  # Load custom dataset
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # DataLoader with batch size 8

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the Transformer model and move it to the chosen device
model = TransformerModel(ntoken=dataset._vocab_size()).to(device)

# Initialize the AdamW optimizer with learning rate 5e-4. AdamW is a type of gradient descent.
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Define the loss function (negative log-likelihood loss for one-hot encoded targets)
criterion = nn.NLLLoss()

# Define the output directory and filename for saving the model
output_dir = "./saved_model"
model_filename = "mymodel.pth"

# Try loading a pre-trained model if it exists
try:
    model_path = os.path.join(output_dir, model_filename)  # Full path to model file
    model.load_state_dict(torch.load(model_path))          # Load model state_dict
    print(f"Model {model_filename} loaded successfully.")

# If the model file isn't found, train the model
except FileNotFoundError:
    print(f"No saved model {model_filename} found at folder {output_dir}. Training the model...")

    # Define number of training epochs
    epochs = 30
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        # Iterate over batches in the dataloader
        for input_seq, target_seq in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # Move input and target sequences to the chosen device
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass to predict next tokens
            output = model(input_seq)

            # Reshape output and target for loss computation
            output = output.reshape(-1, dataset._vocab_size())
            target_seq = target_seq.reshape(-1)

            # Compute loss (negative log-likelihood)
            loss = criterion(output, target_seq)

            loss.backward()        # Backpropagate gradients
            optimizer.step()       # Update model parameters

            total_loss += loss.item()  # Accumulate total loss

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")  # Print average loss per epoch

    # Save the trained model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
    torch.save(model.state_dict(), os.path.join(output_dir, model_filename))  # Save model state_dict

# Text generation example with trained model
model.eval()  # Set model to evaluation mode
#prompt = "Harry Potter asked Hermione "  # Initial prompt
#prompt = "Hermione asked Harry Potter "  # Alternartive prompt
prompt = "Hermione asked Harry Potter"
prompt = input("Enter prompt: ")

# Encode the input sequence for text generation
input_seq = dataset.encode(prompt).to(device)
generated_text = prompt  # Initialize generated text with the prompt

# Generate 50 tokens
for _ in range(50):
    with torch.no_grad():  # Disable gradient computation for efficiency
        output = model(input_seq)  # Get model predictions for current input sequence
        # TODO Your code here
        # Get the top-k probabilities and indices
        # Get logits and probabilities for the last token position in the sequence
        logits = output[:, -1, :]  # Get logits for the last position
        probabilities = torch.softmax(logits, dim=-1).squeeze()  # Convert logits to probabilities
        
        # Greedy selection of the most probable token
        next_token = torch.argmax(probabilities).item()  # Get token ID of the highest probability
        
        # Print chosen token and its probability
        print(f"Chosen token: '{dataset.decode(next_token)}' with probability {probabilities[next_token]:.4f}")
        
        # Append the predicted token to the generated text
        generated_text += dataset.decode(next_token)
        
        # Update input sequence by appending the new token
        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)

# Print the generated text
print("My generated text:", generated_text)
