from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

type T_NN = nn.GRU

type T_forward = tuple[Tensor, Tensor]

class StatefulNN(nn.Module):
    def __init__(self, input_dim: int = 112, hidden_dim: int = 128, depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.depth: int = depth

        self.network: T_NN = self._gru(input_dim)
        self.fc: nn.Linear = nn.Linear(hidden_dim, 1)

    def __call__(self, x: Tensor, h: Tensor) -> T_forward:
        return self.forward(x, h)

    def _gru(self, input_dim: int) -> nn.GRU:
        return nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.depth,
            batch_first=True
        )

    def forward(self, x: Tensor, h: Tensor) -> T_forward:
        out, h_inf = self.network(x, h)
        inference: Tensor = self.fc(out)
        return inference, h_inf

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        return torch.zeros(self.depth, batch_size, self.hidden_dim)


def train() -> None:
    # 1. Initialize model, loss, and optimizer
    model = StatefulNN()
    model.train() # Sets module in training mode (enables gradients, dropout, etc.)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Create Dummy DataLoader (e.g., 100 sequences, 50 steps long, 112 features)
    batch_size = 16
    seq_len = 50
    
    x_train = torch.randn(100, seq_len, 112)
    y_train = torch.randn(100, seq_len, 1) # Target value for each step in the sequence
    
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Training Loop
    epochs = 3
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            # Get actual batch size (last batch might be smaller than 16)
            curr_batch_size = batch_x.size(0)
            
            # Initialize a fresh hidden state for each independent batch of sequences
            h_state = model.get_initial_state(curr_batch_size)
            
            # Zero out gradients from the previous step
            optimizer.zero_grad()
            
            # Forward pass: PyTorch GRU processes all 50 time steps at once
            predictions, _ = model(batch_x, h_state)
            
            # Compute loss, backpropagate, and update weights
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {epoch_loss / len(dataloader):.4f}")


def test() -> None:
    # 1. Initialize model and set to evaluation mode
    model = StatefulNN()
    model.eval() # Disables dropout and batch norm tracking
    
    criterion = nn.MSELoss()

    # 2. Create Dummy DataLoader for Testing
    batch_size = 16
    seq_len = 50
    
    x_test = torch.randn(32, seq_len, 112)
    y_test = torch.randn(32, seq_len, 1)
    
    dataset = TensorDataset(x_test, y_test)
    # Shuffle is usually False for testing/validation to keep order consistent
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. Evaluation Loop
    test_loss = 0.0
    
    # torch.no_grad() prevents PyTorch from storing computational graphs, 
    # massively reducing memory usage and speeding up inference.
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            curr_batch_size = batch_x.size(0)
            
            h_state = model.get_initial_state(curr_batch_size)
            
            predictions, _ = model(batch_x, h_state)
            loss = criterion(predictions, batch_y)
            
            test_loss += loss.item()
            
    print(f"Test Loss: {test_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    print("--- Starting Training ---")
    train()
    print("\n--- Starting Testing ---")
    test()