import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class TopicSegModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional,
                 dropout):

        super(TopicSegModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0,
                            batch_first = True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hasGPU = torch.cuda.is_available()
    
    def forward(self, x):

        # text = [batch_size, seq_len, hidden_dim]
        # input data is already embedding

        # dropout_x = self.dropout(x)
        # outputs shape: [batch_size, seq_len, hidden_dim]
        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        outputs, (hidden, cell) = self.lstm(x)

        out1, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) 
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(out1))
        
        #predictions = [sent len, batch size, output dim]
        return predictions, out1
        