import flwr as fl
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np 

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device,cid, mod_name, bwt=0, criterion=nn.MSELoss(), learning_rate=0.0001, local_epochs=1):
        self.local_epochs = local_epochs
        self.cid = cid
        self.f1_loc = 0
        self.model_name = mod_name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.bwt = bwt
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print(f"================================================ Client {self.cid} ================================================\n")
        


    def get_parameters(self, config=None):
        print(f"Client {self.cid} get_parameters")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = self.optimizer
        criterion = self.criterion
        self.model.train()
        print(f"Client {self.cid} fit ++++++++++++ {self.model} ++++++++++++++")
        for epoch in tqdm(range(self.local_epochs), colour="green", desc=f"Client {self.cid}"):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                if self.model_name == "cnn_lstm":
                    output = output[:, -1, :]
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.local_epochs}, Loss: {loss.item()}")
            

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = self.criterion
        self.model.eval()
        loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if self.model_name == "cnn_lstm":
                    output = output[:, -1, :]
                loss += criterion(output, y).item()

                preds = torch.argmax(output, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        self.f1_loc = f1_score(y_true, y_pred)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy = np.mean(y_true == y_pred)/len(y_true)
        print(f"Client {self.cid} evaluate ++++++++++++ {accuracy} ++++++++++++++")
        print(f"Client {self.cid} evaluate ++++++++++++ {self.f1_loc} ++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return float(loss), len(self.test_loader.dataset), {"mse": float(loss), "f1_score": f1_score(y_true, y_pred, average="weighted")}
    
    
    
        
def run_client(cid, model_name, model, train_loader, test_loader, loc_epochs, loc_bwt=0):
    model = model
    print(f"=============================== START RUN Client {cid} ===============================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {cid} using device: {device}")
    client = FlowerClient(model, train_loader, test_loader, device, mod_name = model_name, cid=cid,  bwt=loc_bwt, criterion=nn.MSELoss(), learning_rate=0.0001, local_epochs=loc_epochs)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
