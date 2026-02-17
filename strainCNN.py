import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Custom Dataset
class StrainForceDataset(Dataset):
    """Dataset for strain signal + scalar feature to force prediction"""
    def __init__(self, xdata_strain, xdata_scalar, ydata, scaler_strain=None, 
                 scaler_scalar=None, scaler_y=None):
        """
        Args:
            xdata_strain: (N, L) array of strain measurements
            xdata_scalar: (N,) or (N, F) array of scalar features
            ydata: (N, C) array of force values at different contacts
            scaler_strain: Optional pre-fitted scaler for strain
            scaler_scalar: Optional pre-fitted scaler for scalar
            scaler_y: Optional pre-fitted scaler for Y
        """
        self.xdata_strain = xdata_strain
        # Ensure scalar is 2D
        self.xdata_scalar = xdata_scalar.reshape(-1, 1) if xdata_scalar.ndim == 1 else xdata_scalar
        self.ydata = ydata
        
        # Normalize strain data
        if scaler_strain is None:
            self.scaler_strain = StandardScaler()
            self.xdata_strain_norm = self.scaler_strain.fit_transform(xdata_strain)
        else:
            self.scaler_strain = scaler_strain
            self.xdata_strain_norm = self.scaler_strain.transform(xdata_strain)
        
        # Normalize scalar data
        if scaler_scalar is None:
            self.scaler_scalar = StandardScaler()
            self.xdata_scalar_norm = self.scaler_scalar.fit_transform(self.xdata_scalar)
        else:
            self.scaler_scalar = scaler_scalar
            self.xdata_scalar_norm = self.scaler_scalar.transform(self.xdata_scalar)
            
        # Normalize y data
        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.ydata_norm = self.scaler_y.fit_transform(ydata)
        else:
            self.scaler_y = scaler_y
            self.ydata_norm = self.scaler_y.transform(ydata)
    
    def __len__(self):
        return len(self.xdata_strain)
    
    def __getitem__(self, idx):
        # Add channel dimension for 1D Conv: (L,) -> (1, L)
        x_strain = torch.FloatTensor(self.xdata_strain_norm[idx]).unsqueeze(0)
        x_scalar = torch.FloatTensor(self.xdata_scalar_norm[idx])
        y = torch.FloatTensor(self.ydata_norm[idx])
        return x_strain, x_scalar, y


# CNN Model
class StrainForceCNN(nn.Module):
    """1D CNN with FiLM for strain signal + scalar feature to force prediction"""
    def __init__(self, input_length, num_outputs, num_scalar_features=1, dropout_rate=0.6):
        """
        Args:
            input_length: Length of input strain signal
            num_outputs: Number of output force values
            num_scalar_features: Number of scalar features (e.g., 1)
            dropout_rate: Dropout probability for regularization
        """
        super(StrainForceCNN, self).__init__()
        
        # Convolutional feature extraction layers
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, dilation=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # FiLM layers: scalar features → modulation parameters (gamma, beta)
        self.film_layer = nn.Sequential(
            nn.Linear(num_scalar_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 2)  # 64 channels * 2 (gamma and beta)
        )
        
        # Calculate flattened size after convolutions
        self.flattened_size = self._get_flattened_size(input_length)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
    
    def _get_flattened_size(self, input_length):
        """Calculate size after conv layers"""
        x = torch.zeros(1, 1, input_length)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x.numel()
    
    def forward(self, x_strain, x_scalar):
        """
        Args:
            x_strain: (batch, 1, length) - strain signal
            x_scalar: (batch, num_scalar_features) - scalar features
        """
        # Process strain through conv layers
        x = self.conv_block1(x_strain)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # (batch, 64, spatial_dim)
        
        # Generate modulation parameters from scalar features
        film_params = self.film_layer(x_scalar)  # (batch, 128)
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each (batch, 64)
        
        # Apply FiLM modulation: x = gamma * x + beta
        # Reshape to broadcast across spatial dimension
        gamma = gamma.unsqueeze(2)  # (batch, 64, 1)
        beta = beta.unsqueeze(2)    # (batch, 64, 1)
        x = gamma * x + beta
        
        # Flatten and process through FC layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_strain, batch_scalar, batch_y in dataloader:  # Now unpacking 3 items
        batch_strain = batch_strain.to(device)
        batch_scalar = batch_scalar.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_strain, batch_scalar)  # Pass both inputs
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
        for batch_strain, batch_scalar, batch_y in dataloader:  # Unpack 3 items
            batch_strain = batch_strain.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_strain, batch_scalar)  # Pass both inputs
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=1.0, nonzero_weight=5.0, threshold=0.1):
        """
        Args:
            zero_weight: Weight for samples near zero
            nonzero_weight: Weight for non-zero samples
            threshold: Values below this are considered "zero"
        """
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.threshold = threshold
    
    def forward(self, predictions, targets):
        # Create weight mask based on target values
        is_nonzero = (torch.abs(targets) > self.threshold).float()
        weights = is_nonzero * self.nonzero_weight + (1 - is_nonzero) * self.zero_weight
        
        # Weighted MSE
        squared_error = (predictions - targets) ** 2
        weighted_loss = (squared_error * weights).mean()
        return weighted_loss
# Main training pipeline

# Usage:
def train_model(xdata_strain, xdata_scalar, ydata, epochs=100, 
                batch_size=32, learning_rate=0.001, val_split=0.2, patience=15):
    """
    Complete training pipeline
    
    Args:
        xdata_strain: (N, L) strain data
        xdata_scalar: (N,) or (N, F) scalar feature data
        ydata: (N, C) force data
        train_idxs: Training indices
        val_idxs: Validation indices
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
    # FIT SCALERS ONLY ON TRAINING DATA
    scaler_strain = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_strain.fit(xdata_strain)
    # Reshape scalar for StandardScaler if 1D
    xdata_scalar_2d = xdata_scalar.reshape(-1, 1) if xdata_scalar.ndim == 1 else xdata_scalar
    scaler_scalar.fit(xdata_scalar_2d)
    scaler_y.fit(ydata)
    
    # Create dataset with pre-fitted scalers
    dataset = StrainForceDataset(xdata_strain, xdata_scalar, ydata, 
                                  scaler_strain, scaler_scalar, scaler_y)
    
    # Train/validation split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_dataset = Subset(dataset, train_idxs)
    # val_dataset = Subset(dataset, val_idxs)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_length = xdata_strain.shape[1]
    num_outputs = ydata.shape[1]
    num_scalar_features = xdata_scalar_2d.shape[1]
    
    model = StrainForceCNN(input_length, num_outputs, num_scalar_features).to(device)
    
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
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Input length: {input_length}, Scalar features: {num_scalar_features}, Output size: {num_outputs}\n")
    
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
def predict(model, dataset, new_xdata_strain, new_xdata_scalar, device='cpu'):
    """
    Make predictions on new data
    
    Args:
        model: Trained model
        dataset: Dataset with fitted scalers
        new_xdata_strain: New strain data (N, L)
        new_xdata_scalar: New scalar data (N,) or (N, F)
        device: Device to use
    
    Returns:
        predictions: Predicted forces in original scale
    """
    model.eval()
    model.to(device)
    
    # Normalize inputs
    xdata_strain_norm = dataset.scaler_strain.transform(new_xdata_strain)
    xdata_scalar_2d = new_xdata_scalar.reshape(-1, 1) if new_xdata_scalar.ndim == 1 else new_xdata_scalar
    xdata_scalar_norm = dataset.scaler_scalar.transform(xdata_scalar_2d)
    
    # Create tensors
    x_strain_tensor = torch.FloatTensor(xdata_strain_norm).unsqueeze(1).to(device)
    x_scalar_tensor = torch.FloatTensor(xdata_scalar_norm).to(device)
    
    with torch.no_grad():
        predictions_norm = model(x_strain_tensor, x_scalar_tensor).cpu().numpy()
    
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

