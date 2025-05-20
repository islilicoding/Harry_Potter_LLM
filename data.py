from io import open
import torch
from torch.utils.data import Dataset

# Define a custom dataset class for language modeling with sequences of text
class MyDataset(Dataset):
    def __init__(self, filename, sequence_length=128): 
        """
        Initialize the dataset with a text file, process it into tokens, and create vocabulary mappings.
        
        Args:
            filename (str): Path to the text file to be used.
            sequence_length (int): Length of each input sequence for training.
        """
        # Load and preprocess text from the specified file
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        # Build a character-level vocabulary
        # The tokens will be a single character because predicting characters is easier for smaller models
        # Each unique character in the text is assigned a unique integer index (token value)
        chars = sorted(list(set(text)))         # Get sorted list of unique characters
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # Character to index mapping (string-to-index)
        self.itos = {i: ch for i, ch in enumerate(chars)}  # Index to character mapping (index-to-string)
        self.vocab_size = len(chars)            # Size of the vocabulary

        # Encode the entire text into integer indices based on the vocabulary
        encoded_text = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

        # Store the sequence length and encoded text as class attributes
        self.sequence_length = sequence_length
        self.encoded_text = encoded_text

    def encode(self, src):
        """
        Convert a string into a tensor of integer indices according to the vocabulary.
        
        Args:
            src (str): Input string to encode.
            
        Returns:
            Tensor: A tensor containing the integer indices of each character in the string.
        """
        return torch.tensor([self.stoi[ch] for ch in src], dtype=torch.long).unsqueeze(0)

    def decode(self, next_token):
        """
        Convert an integer index back into the corresponding character.
        
        Args:
            next_token (int): Integer index of the character.
            
        Returns:
            str: The character corresponding to the integer index.
        """
        return self.itos[next_token]

    def _vocab_size(self):
        """
        Return the vocabulary size of the dataset.
        
        Returns:
            int: Size of the vocabulary.
        """
        return self.vocab_size

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Each sample is a sequence of length `self.sequence_length`.
        """
        # Total length is reduced by sequence_length to avoid overflow in target sequence
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, idx):
        """
        Retrieve an input and target sequence pair from the dataset.
        
        Args:
            idx (int): Index of the starting position for the sequence.
        
        Returns:
            tuple: (input_seq, target_seq), where each is a tensor of length `sequence_length`.
        """
        # Input sequence: sequence of characters starting from index `idx`
        input_seq = self.encoded_text[idx:idx + self.sequence_length]
        # Target sequence: sequence shifted by one character to predict the next character
        target_seq = self.encoded_text[idx + 1:idx + 1 + self.sequence_length]
        return input_seq, target_seq
