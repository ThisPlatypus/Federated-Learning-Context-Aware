import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Skip1D(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128, skip_steps=2, fc_out_dim=1):
        super(CNN_LSTM_Skip1D, self).__init__()
        self.skip_steps = skip_steps
        self.lstm_hidden = lstm_hidden

        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=1, padding='same')
        self.relu = nn.ReLU()

        # Pooling (no actual effect here, but included per spec)
        self.pool = nn.AvgPool1d(kernel_size=1, padding=0)  # effectively a no-op

        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, batch_first=True)

        # Skip connection layers (for FC transformation)
        self.fc_v = nn.Linear(lstm_hidden, fc_out_dim)
        self.fc_s = nn.ModuleList([nn.Linear(lstm_hidden, fc_out_dim) for _ in range(skip_steps)])
        self.b_out = nn.Parameter(torch.zeros(fc_out_dim))

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        Expected input_dim = 1 for grayscale/univariate input
        """
        # Apply 1D CNN: (B, seq_len, in) -> (B, in, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Back to (B, seq_len, cnn_filters)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_hidden)
        skip_outputs = []

        for t in range(x.size(1)):
            current = lstm_out[:, t]  # shape (B, hidden)

            # Gather skip connections
            skip_sum = 0
            for i in range(self.skip_steps):
                if t - i - 1 >= 0:
                    skip_sum += self.fc_s[i](lstm_out[:, t - i - 1])

            # Final output at timestep t
            pV_t = self.fc_v(current)
            pD_t = pV_t + skip_sum + self.b_out
            skip_outputs.append(pD_t.unsqueeze(1))

        return torch.cat(skip_outputs, dim=1)  # (B, seq_len, fc_out_dim)




########################################## CNN Model ##########################################



# 1D CNN Model for sequence data
class CNN1DSequenceClassifier(nn.Module):
    def __init__(self, input_channels=1, seq_len=30, num_classes=2):
        super(CNN1DSequenceClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((seq_len // 2) * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
