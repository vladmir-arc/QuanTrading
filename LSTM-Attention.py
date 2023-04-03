# Import libraries
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Define hyperparameters
batch_size = 32
seq_length = 100
hidden_size = 64
attention_size = 32
learning_rate = 0.01
num_epochs = 10


# Define your own stock dataset class
class StockDataset(Dataset):
    def __init__(self, data_file):
        # Load and preprocess your data from the file
        self.data = pd.read_csv(data_file).loc[:, "Close"]

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the input and output tensors for the given index
        return self.data[idx][0], self.data[idx][1]


# Create train and test datasets
train_dataset = StockDataset('train_data.csv')
test_dataset = StockDataset('test_data.csv')

# Create train and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# # Define bidirectional LSTM layer with batch_first=True
# lstm_layer = nn.LSTM(1, hidden_size, batch_first=True)
#
# # Define attention layer
# attention_layer = nn.MultiheadAttention(hidden_size * 2, 1)
#
# # Define output layer
# output_layer = nn.Linear(hidden_size * 2, 1)
#
#
# # Define model as a class
# class AttentionLSTM(nn.Module):
#     def __init__(self):
#         super(AttentionLSTM, self).__init__()
#         self.lstm_layer = lstm_layer
#         self.attention_layer = attention_layer
#         self.output_layer = output_layer
#
#     def forward(self, input):
#         # Get the output and the final hidden state of the LSTM layer
#         lstm_outputs, (h_n, c_n) = self.lstm_layer(input)
#         # Concatenate the forward and backward hidden states
#         state_h = torch.cat((h_n[0], h_n[1]), dim=1)
#         # Transpose the lstm_outputs and state_h to match the expected shape of attention_layer
#         lstm_outputs = lstm_outputs.transpose(0, 1)
#         state_h = state_h.unsqueeze(0)
#         # Get the attention output and weights
#         attention_output, attention_weights = self.attention_layer(lstm_outputs, state_h, state_h)
#         # Get the final output
#         output = self.output_layer(attention_output)
#         return output
#
#
# # Create an instance of the model
# model = AttentionLSTM()
#
# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#
# # Define a function to calculate MAE
# def mae(y_true, y_pred):
#     return torch.mean(torch.abs(y_true - y_pred))
#
#
# # Train the model
# for epoch in range(num_epochs):
#     # Initialize the epoch loss and MAE
#     epoch_loss = 0.0
#     epoch_mae = 0.0
#     # Loop over the batches of data
#     for inputs, outputs in train_dataloader:
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#         # Forward pass
#         preds = model(inputs)
#         # Calculate loss and MAE
#         loss = criterion(preds, outputs)
#         mae_value = mae(outputs, preds)
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
#         # Accumulate loss and MAE
#         epoch_loss += loss.item()
#         epoch_mae += mae_value.item()
#     # Print average loss and MAE per epoch
#     print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataloader)}, MAE: {epoch_mae / len(train_dataloader)}')
#
# # Test the model (no need to compute gradients)
# with torch.no_grad():
#     # Initialize the test loss and MAE
#     test_loss = 0.0
#     test_mae = 0.0
#     # Loop over the batches of data
#     for inputs, outputs in test_dataloader:
#         # Forward pass
#         preds = model(inputs)
#         # Calculate loss and MAE
#         loss = criterion(preds, outputs)
#         mae_value = mae(outputs, preds)
#         # Accumulate loss and MAE
#         test_loss += loss.item()
#         test_mae += mae_value.item()
#
#     # Print average loss and MAE for test set
#     print(f'Test Loss: {test_loss / len(test_dataloader)}, Test MAE: {test_mae / len(test_dataloader)}')
