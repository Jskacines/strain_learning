import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Custom Dataset
class StrainForceDataset(Dataset):
    """Dataset for strain signal to force prediction"""
    def __init__(self, xdata, ydata, scaler_x=None, scaler_y=None):
        """
        Args:
            xdata: (N, L) array of strain measurements
            ydata: (N, C) array of force values at different contacts
            scaler_x: Optional pre-fitted scaler for X
            scaler_y: Optional pre-fitted scaler for Y
        """
        self.xdata = xdata
        self.ydata = ydata
        
        # Normalize data
        if scaler_x is None:
            self.scaler_x = StandardScaler()
            self.xdata_norm = self.scaler_x.fit_transform(xdata)
        else:
            self.scaler_x = scaler_x
            self.xdata_norm = self.scaler_x.transform(xdata)
            
        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.ydata_norm = self.scaler_y.fit_transform(ydata)
        else:
            self.scaler_y = scaler_y
            self.ydata_norm = self.scaler_y.transform(ydata)
    
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, idx):
        # Add channel dimension for 1D Conv: (L,) -> (1, L)
        x = torch.FloatTensor(self.xdata_norm[idx]).unsqueeze(0)
        y = torch.FloatTensor(self.ydata_norm[idx])
        return x, y


# CNN Model
class StrainForceCNN(nn.Module):
    """1D CNN for strain signal to force prediction"""
    def __init__(self, input_length, num_outputs, dropout_rate=0.3):
        """
        Args:
            input_length: Length of input strain signal
            num_outputs: Number of output force values (contacts)
            dropout_rate: Dropout probability for regularization
        """
        super(StrainForceCNN, self).__init__()
        
        # Convolutional feature extraction layers
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate flattened size after convolutions
        self.flattened_size = self._get_flattened_size(input_length)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def _get_flattened_size(self, input_length):
        """Calculate size after conv layers"""
        x = torch.zeros(1, 1, input_length)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x.numel()
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Validation function
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Main training pipeline
def train_model(xdata, ydata, epochs=100, batch_size=32, learning_rate=0.001, 
                val_split=0.2, patience=15):
    """
    Complete training pipeline
    
    Args:
        xdata: (N, L) strain data
        ydata: (N, C) force data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_split: Validation split fraction
        patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Training history
        dataset: Dataset with fitted scalers
    """
    # Create dataset
    dataset = StrainForceDataset(xdata, ydata)
    
    # Train/validation split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_length = xdata.shape[1]
    num_outputs = ydata.shape[1]
    
    model = StrainForceCNN(input_length, num_outputs).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print(f"Input length: {input_length}, Output size: {num_outputs}\n")
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    return model, history, dataset


# Prediction function
def predict(model, dataset, new_xdata, device='cpu'):
    """
    Make predictions on new data
    
    Args:
        model: Trained model
        dataset: Dataset with fitted scalers
        new_xdata: New strain data (N, L)
        device: Device to use
    
    Returns:
        predictions: Predicted forces in original scale
    """
    model.eval()
    model.to(device)
    
    # Normalize input
    xdata_norm = dataset.scaler_x.transform(new_xdata)
    x_tensor = torch.FloatTensor(xdata_norm).unsqueeze(1).to(device)
    
    with torch.no_grad():
        predictions_norm = model(x_tensor).cpu().numpy()
    
    # Inverse transform to original scale
    predictions = dataset.scaler_y.inverse_transform(predictions_norm)
    
    return predictions


# Visualization function
def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


# Example usage
if __name__ == "__main__":
    # Example: Create dummy data (replace with your actual data)
    # xdata = cutoff_strain  # Your actual strain data
    # ydata = ...  # Your actual force data
    
    # For demonstration:
    n_samples = 1000
    signal_length = 100
    n_contacts = 10
    
    xdata = np.random.randn(n_samples, signal_length)
    ydata = np.random.randn(n_samples, n_contacts)
    
    # Train model
    model, history, dataset = train_model(
        xdata, ydata,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        val_split=0.2,
        patience=15
    )
    
    # Plot training history
    fig = plot_training_history(history)
    fig.savefig('/mnt/user-data/outputs/training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved!")
    
    # Make predictions on test data
    test_predictions = predict(model, dataset, xdata[:10])
    print("\nExample predictions (first 10 samples):")
    print(test_predictions)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_x': dataset.scaler_x,
        'scaler_y': dataset.scaler_y,
        'input_length': xdata.shape[1],
        'num_outputs': ydata.shape[1]
    }, '/mnt/user-data/outputs/strain_force_model.pt')
    print("\nModel saved to strain_force_model.pt")

