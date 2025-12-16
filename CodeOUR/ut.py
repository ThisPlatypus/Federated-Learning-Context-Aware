from torch.utils.data import Dataset
import torch
import numpy as np

def label_binary(df, rul_threshold=30):
    grouped = df.groupby('unit')
    binary_labels = []

    for unit, group in grouped:
        max_cycle = group['cycle'].max()
        for cycle in group['cycle']:
            rul = max_cycle - cycle
            binary_labels.append(1 if rul < rul_threshold else 0)

    # Balance the classes
    labels = np.array(binary_labels)
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    min_len = min(len(idx_0), len(idx_1))
    balanced_idx = np.concatenate([np.random.choice(idx_0, min_len, replace=False),
                                   np.random.choice(idx_1, min_len, replace=False)])
    balanced_idx.sort()
    return labels[balanced_idx].tolist()



class CMAPSSBinaryDataset(Dataset):
    def __init__(self, df, sensor_id, rul_threshold=30, window=30):
        self.sensor = f's{sensor_id}'
        self.window = window

        self.data = df[['unit', 'cycle', self.sensor]].copy()
        self.labels = label_binary(df, rul_threshold)

        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for unit in self.data['unit'].unique():
            unit_data = self.data[self.data['unit'] == unit][self.sensor].values
            unit_labels = self.labels[:len(unit_data)]
            for i in range(len(unit_data) - self.window):
                X.append(unit_data[i:i+self.window].reshape(-1, 1))
                y.append(unit_labels[i + self.window])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
    

def get_mask_around_mean(weights: torch.Tensor, level: float) -> torch.Tensor:

    assert 0 <= level <= 100, "Level must be between 0 and 75"

    # Flatten for processing
    flat_weights = weights.flatten()
    mean_val = flat_weights.mean()
    
    # Determine percentage of data to keep based on level
    if 75 <= level <= 100:
        keep_percent = 1.0  # 100%
    elif 50 <= level < 75:
        keep_percent = 0.8  # 80%
    elif 25 <= level < 50:
        keep_percent = 0.5  # 50%
    elif 0 <= level < 25:
        keep_percent = 0.3  # 30%

    num_elements = flat_weights.numel()
    num_to_keep = int(keep_percent * num_elements)

    # Sort by distance to mean
    distances = torch.abs(flat_weights - mean_val)
    _, sorted_indices = torch.sort(distances)

    # Select closest indices to mean
    keep_indices = sorted_indices[:num_to_keep]
    
    # Create binary mask
    mask = torch.zeros_like(flat_weights, dtype=torch.bool)
    mask[keep_indices] = 1
    mask = mask.reshape(weights.shape).int()

    # Calculate and print stats
    total = mask.numel()
    num_ones = mask.sum().item()
    num_zeros = total - num_ones
    percent_ones = 100 * num_ones / total
    percent_zeros = 100 - percent_ones

    print(f"% of 0 in mask (outside bounds): {percent_zeros:.2f}%")
    print(f"% of 1 in mask (inside bounds): {percent_ones:.2f}%")

    return mask


def generate_weighted_distribution(size=1):
    numbers = np.arange(1, 101)
    
    # Define weights based on the intervals:
    # 30% for 75-100, 40% for 50-74, 20% for 25-49, 10% for 1-24
    
    weights = np.zeros_like(numbers, dtype=float)
    
    weights[(numbers >= 75) & (numbers <= 100)] = 0.30 / 26    # 26 numbers in 75-100
    weights[(numbers >= 50) & (numbers <= 74)] = 0.40 / 25    # 25 numbers in 50-74
    weights[(numbers >= 25) & (numbers <= 49)] = 0.20 / 25    # 25 numbers in 25-49
    weights[(numbers >= 1) & (numbers <= 24)]  = 0.10 / 24    # 24 numbers in 1-24
    
    # Sample 'size' numbers with the defined weighted probabilities
    sample = np.random.choice(numbers, size=size, p=weights)
    return sample
