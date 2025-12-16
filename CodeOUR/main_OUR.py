import flwr as fl
from client import run_client
import sys
from typing import List, Tuple, Optional

from torch.utils.data import DataLoader
from model import CNN_LSTM_Skip1D, CNN1DSequenceClassifier
from datasets.cmapss import load_data
from torch.utils.data import DataLoader
from ut import CMAPSSBinaryDataset, generate_weighted_distribution, get_mask_around_mean
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from typing import List
from flwr.common.typing import Metrics
import torch


NUM_CLIENTS = 20
MIN_CLIENTS = 5
DATASET_SIZE = 2600  # Assume 100 samples per client
NUM_ROUNDS = 5
LOC_EPOCHS = 20
MODEL_PATH = '/home/camerotac/JJ/GLOB_MODEL'
BATCH_SIZE = 64 #128

from logging import INFO
import pickle
from pathlib import Path
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays

def weighted_average(metrics: List[tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    # Find all metric keys present
    keys = set().union(*(m.keys() for _, m in metrics))
    result = {}
    for k in keys:
        filtered = [(num_examples, m[k]) for num_examples, m in metrics if k in m]
        if filtered:
            values = [num_examples * v for num_examples, v in filtered]
            examples = [num_examples for num_examples, _ in filtered]
            result[k] = sum(values) / sum(examples)
    return result


class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """
    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""

        # convert parameters to list of NumPy arrays
        # this will make things easy if you want to load them into a
        # PyTorch or TensorFlow model later
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'globa_parameters': ndarrays}
        filename = str(self.save_path/f"parameters_round_{server_round}.pkl")
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")
        


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, dict[str, Scalar]]]:
        if not results:
            return None
        
        bw = generate_weighted_distribution(20)
        
        
        # Convert parameters to NumPy arrays
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]

        # Transpose weights so each layer is grouped across clients
        layers_grouped = list(zip(*weights))

        aggregated_layers = []
        for layer_i, layer_group in enumerate(layers_grouped):
            # Convert to torch tensors for processing
            torch_layers = [torch.tensor(layer) for layer in layer_group]

            # Stack to get a 3D tensor: [num_clients, ...layer_shape...]
            stacked = torch.stack(torch_layers)

            # Compute average layer after masking
            masked_sum = torch.zeros_like(stacked[0])
            total_weight = 0

            for client_idx, layer_tensor in enumerate(stacked):
                # Apply mask to individual client weights
                mask = get_mask_around_mean(layer_tensor, level=bw[client_idx])  # Adjust level here
                masked_tensor = layer_tensor * mask

                # Apply weighted sum
                masked_sum += masked_tensor * num_examples[client_idx]
                total_weight += num_examples[client_idx]

            # FedAvg using masked values
            averaged_layer = masked_sum / total_weight
            aggregated_layers.append(averaged_layer.numpy())

        aggregated_parameters = ndarrays_to_parameters(aggregated_layers)
        return aggregated_parameters, {}

        

'''    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)'''
    
    

def start_server(model_name: str = "cnn_lstm"):
    

    strategy = FedAvgWithModelSaving(save_path=f'/home/camerotac/JJ/GLOB_MODEL/{model_name}')
                   #,evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
 
def prepare_data_for_client(cid: int, batch_size: int = 32):
    if cid < 1:
        raise ValueError("cid (sensor id) must be >= 1")
    train = load_data("FD001")[0]  # train
    test = load_data("FD001")[1]  # test
    train_renamed = train.rename(columns={
        'unit_number': 'unit',
        'time': 'cycle',
        f'sensor{cid}': f's{cid}'
    })
    test_renamed = test.rename(columns={
        'unit_number': 'unit',
        'time': 'cycle',
        f'sensor{cid}': f's{cid}'
    })
    train_ds = CMAPSSBinaryDataset(train_renamed, sensor_id=cid)
    test_ds = CMAPSSBinaryDataset(test_renamed, sensor_id=cid)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def select_model(name: str, input_dim: int):
    if name == "cnn_lstm":
        return CNN_LSTM_Skip1D(input_dim=input_dim)
    elif name == "simple_CNN":
        return CNN1DSequenceClassifier(input_channels=1, seq_len=30, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {name}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        start_server("simple_CNN") ######## !!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THIS ACCORDING WITH THE MODEL !!!!!!!!!!!!!!!!!!!!!!!!!
    else:
 
        model_name = sys.argv[1]  # e.g. cnn_lstm or simple_lstm
        cid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        train_loader, test_loader = prepare_data_for_client(cid, batch_size=BATCH_SIZE)
        input_dim = 1  # If using only one sensor per client
        model = select_model(model_name, input_dim=input_dim)
        run_client(cid, 
                   model= model,  # Replace with your model instance
                   model_name=model_name,  # Model name
                   train_loader=train_loader,  # Replace with your DataLoader instance
                   test_loader=test_loader,  # Replace with your DataLoader instance
                   loc_epochs=LOC_EPOCHS,  # Number of local epochs
                   loc_bwt=0)
