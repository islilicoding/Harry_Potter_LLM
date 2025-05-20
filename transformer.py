# Import necessary libraries
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Transformer):
    """A Transformer-based model that includes both encoder and decoder layers."""

    def _generate_positional_encoding(self, sequence_length, embed_size):
        """
        Generates a fixed positional encoding matrix.
        
        Positional encodings provide information about the position of tokens
        in a sequence since Transformers don’t have inherent order information.
        Uses sinusoidal and cosine functions to produce a unique encoding for each position.
        
        Args:
            sequence_length (int): Maximum length of the input sequence.
            embed_size (int): Size of the embedding vector for each token.
            
        Returns:
            Tensor: Positional encoding of shape (1, sequence_length, embed_size).
        """
        # Initialize a matrix of zeros for positional encoding
        pos_encoding = torch.zeros(sequence_length, embed_size, requires_grad=False)
        # Generate positional encodings using sin and cos functions
        for pos in range(sequence_length):
            for i in range(0, embed_size, 2):  # Even indices for sin, odd for cos
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embed_size)))
        return pos_encoding.unsqueeze(0)  # Add a batch dimension
    
    def _generate_positional_encoding_constant(self, sequence_length, embed_size):
        """
        Generates a fixed positional encoding matrix.
        
        Positional encodings provide information about the position of tokens
        in a sequence since Transformers don’t have inherent order information.
        Uses sinusoidal and cosine functions to produce a unique encoding for each position.
        
        Args:
            sequence_length (int): Maximum length of the input sequence.
            embed_size (int): Size of the embedding vector for each token.
            
        Returns:
            Tensor: Positional encoding of shape (1, sequence_length, embed_size).
        """
        # Initialize the positional encoding for a single position
        pos_encoding = torch.zeros(embed_size, requires_grad=False)
        
        # Generate constant encoding values using sin and cos for position 0 only
        for i in range(0, embed_size, 2):  # Even indices for sin, odd for cos
            pos_encoding[i] = math.sin(0 / (10000 ** ((2 * i) / embed_size)))
            if i + 1 < embed_size:
                pos_encoding[i + 1] = math.cos(0 / (10000 ** ((2 * i) / embed_size)))

        # Expand the encoding to all positions along the sequence length
        pos_encoding = pos_encoding.unsqueeze(0).repeat(sequence_length, 1)  # Shape: [sequence_length, embed_size]

        # Add a batch dimension
        return pos_encoding.unsqueeze(0)  # Shape: [1, sequence_length, embed_size]

    def __init__(self, ntoken, embed_size=256, hid_feedforward=512, num_heads=16, num_layers=2, max_sequence_length=128, dropout=0.5):
        """
        Initializes the Transformer model with an embedding layer, encoder, and decoder.
        
        Args:
            ntoken (int): Number of unique tokens (vocabulary size).
            embed_size (int): Embedding dimension for each token.
            hid_feedforward (int): Size of the hidden layer in feedforward networks.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            max_sequence_length (int): Maximum length of input sequences.
            dropout (float): Dropout probability for regularization.
        """
        super(TransformerModel, self).__init__(
            d_model=embed_size,             # Dimension of embeddings
            nhead=num_heads,                # Number of attention heads
            dim_feedforward=hid_feedforward, # Dimension of the feedforward layer
            num_encoder_layers=num_layers,   # Number of encoder layers
            batch_first=True                 # Ensures batch size is the first dimension
        )

        # Set model type and initialize a positional encoding
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = self._generate_positional_encoding(max_sequence_length, embed_size)
        self.max_sequence_length = max_sequence_length

        # Input embedding layer, maps tokens to embedding vectors
        self.input_emb = nn.Embedding(ntoken, embed_size)
        self.embed_size = embed_size

        # Decoder layer to map model output back to vocabulary size
        self.decoder = nn.Linear(embed_size, ntoken)

        # Initialize weights of the embedding and decoder layers
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence to prevent attending to future tokens.
        
        Args:
            sz (int): Size of the mask (equal to sequence length).
            
        Returns:
            Tensor: A lower triangular mask with logarithmic values.
        """
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        """
        Initializes weights of the embedding and decoder layers with uniform values.
        """
        initrange = 0.1  # Range for uniform distribution
        # Initialize embedding weights uniformly
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        # Initialize decoder bias to zero and weights uniformly
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        """
        Defines the forward pass of the model for training and inference.
        
        Args:
            src (Tensor): Input tensor of token indices, shape [batch_size, sequence_length, embed_size].
            
        Returns:
            Tensor: Log-softmax probabilities over vocabulary for each token position.
        """
        # Embed the input tokens and scale by sqrt(embed_size) for stable gradients. This scaling is important when the embedding vectors are summed with positional encodings, as it helps maintain a balanced scale between the two components.
        src = self.input_emb(src) * math.sqrt(self.embed_size)
        sequence_length = src.size(1)  # Get sequence length from input

        # Generate a mask if required, ensuring it matches the sequence length
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            mask = self._generate_square_subsequent_mask(sequence_length).to(src.device)
            self.src_mask = mask

        # Add positional encoding to the input embeddings
     
        # Pass through the transformer encoder with the optional mask
        output = self.encoder(src, mask=self.src_mask)
        # Decode the encoder output to predict the next token in vocabulary space
        output = self.decoder(output)
        
        # Return log-softmax probabilities over the vocabulary
        return F.log_softmax(output, dim=-1)

    # Forward function ignoring ordering of words in the prompt
    def forward_same_prev(self, src):
        """
        Defines the forward pass of the model for training and inference.

        Args:
            src (Tensor): Input tensor of token indices, shape [batch_size, sequence_length, embed_size].

        Returns:
            Tensor: Log-softmax probabilities over vocabulary for each token position.
        """
        # Embed the input tokens and scale by sqrt(embed_size) for stable gradients. This scaling is important when the embedding vectors are summed with positional encodings, as it helps maintain a balanced scale between the two components.
        # **** Commented second part out ****
        src = self.input_emb(src) #* math.sqrt(self.embed_size)
        sequence_length = src.size(1)  # Get sequence length from input

        # Generate a mask if required, ensuring it matches the sequence length
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            mask = self._generate_square_subsequent_mask(sequence_length).to(src.device)
            self.src_mask = mask

        # Add positional encoding to the input embeddings
        src += self.pos_encoder[:, :sequence_length, :].to(src.device)

        # Pass through the transformer encoder with the optional mask
        output = self.encoder(src, mask=self.src_mask)

        # Decode the encoder output to predict the next token in vocabulary space
        output = self.decoder(output)

        # Return log-softmax probabilities over the vocabulary
        return F.log_softmax(output, dim=-1)
    
    # Forward function ignoring ordering of words in the prompt
    def forward_original(self, src):
        """
        Defines the forward pass of the model for training and inference.

        Args:
            src (Tensor): Input tensor of token indices, shape [batch_size, sequence_length, embed_size].

        Returns:
            Tensor: Log-softmax probabilities over vocabulary for each token position.
        """
        # Embed the input tokens and scale by sqrt(embed_size) for stable gradients. This scaling is important when the embedding vectors are summed with positional encodings, as it helps maintain a balanced scale between the two components.
        src = self.input_emb(src) * math.sqrt(self.embed_size) # **** Comment out second part? ****
        sequence_length = src.size(1)  # Get sequence length from input

        # Generate a mask if required, ensuring it matches the sequence length
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            mask = self._generate_square_subsequent_mask(sequence_length).to(src.device)
            self.src_mask = mask

        # Add positional encoding to the input embeddings
        src += self.pos_encoder[:, :sequence_length, :].to(src.device) # **** comment this out/change it? ****

        # Pass through the transformer encoder with the optional mask
        output = self.encoder(src, mask=self.src_mask) # **** comment out mask? ****

        # Decode the encoder output to predict the next token in vocabulary space
        output = self.decoder(output)

        # Return log-softmax probabilities over the vocabulary
        return F.log_softmax(output, dim=-1)