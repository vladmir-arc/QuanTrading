# Import libraries
import torch
import torch.nn as nn


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

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, binary=False):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.binary = binary
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Add attention layer
        self.attention = nn.MultiheadAttention(hidden_size=hidden_size,
                                               num_heads=1,
                                               dropout=0.2,
                                               batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x, (self.h0.expand(1, x.size(0), self.hidden_size),
                               self.c0.expand(1, x.size(0), self.hidden_size)))

        # Add attention mechanism
        out_attended, _ = self.attention(out, out, out)  # Attend to all timesteps

        out = self.dropout(out_attended[:, -1, :])
        out = self.fc(out)

        if self.binary:
            out = self.sigmoid(out)

        return out