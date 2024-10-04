from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class MalConv(nn.Module):
    # trained to minimize cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    def __init__(self, out_size=2, channels=128, window_size=512, embd_size=64):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        print(self.embd)
        self.window_size = window_size
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        
        x = self.embd(x.long())
        x = torch.transpose(x,-1,-2)
        print(x.shape)
        import IPython
        IPython.embed()
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        x = self.pooling(x)
        
        #Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        x = self.softmax(x)
        
        return x

if __name__ == "__main__":
    # Generate random data to simulate malware dataset (features and labels)
    # In practice, replace this with your actual data loading logic
    number_of_samples = 10000  # Replace with the number of samples in your dataset
    X = torch.rand(number_of_samples, 2351)  # Feature tensor
    y = torch.randint(0, 2, (number_of_samples,))  # Binary labels tensor
    print(y)
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data to TensorDataset for ease of use
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = MalConv()
    model.train()

    # Use CrossEntropyLoss for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training setup
    n_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long().unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            # outputs = F.softmax(model(inputs), dim=-1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader)}')
