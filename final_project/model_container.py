import os
import torch
import timeit
from pytorch_utils import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer

_pickle_path = os.path.join(os.getcwd(), "pickled")
os.makedirs(_pickle_path, exist_ok = True)

class ModelContainer:
    def __init__(self, nnet: torch.nn.Module, returns_dict: bool = True) -> None:
        self._network = nnet
        self._network.to(device)
        self._returns_dict = returns_dict
        self._dataset: TensorDataset = None
        self._loader: DataLoader = None
        self._losses: list = []
        
    def set_dataset(self, input_, output_, batch_size: int = 1):
        self._loader = DataLoader(
            TensorDataset(input_, output_), 
            shuffle = True,
            batch_size = batch_size
        )
        
    def save_loader(self, name: str): self._save_pickle_file(self._loader, name)
    def save_model(self, name: str): self._save_pickle_file(self._network.state_dict(), name)
    def save_losses(self, name: str): self._save_pickle_file(self._losses, name)
        
    def load_loader(self, file: str): 
        self._loader: DataLoader = self._load_pickle_file(file)
        
    def load_model(self, state_dict: dict): 
        self._network.load_state_dict(state_dict)
        
    def set_optimizer(self, optimizer: Optimizer): 
        self._optimizer = optimizer 
        
    def set_cost(self, cost: torch.nn.modules.loss._Loss): 
        self._cost = cost
    
    def train(self, num_epochs: int):
        assert self._loader is not None, \
            "Call `set_dataset` prior to training"
            
        for epoch in range(num_epochs):
            t0 = timeit.default_timer()
            for (x, y) in self._loader:
                if self._returns_dict:
                    out: torch.Tensor = self._network(x)['out']
                else:
                    out: torch.Tensor = self._network(x)
                                
                self._network.zero_grad()
                loss: torch.Tensor = self._cost(out, y)
                loss.backward()
                                
                self._losses.append(loss.data.cpu().item())
                self._optimizer.step()
                                
            print(f"Epoch: {epoch+1}/{num_epochs}. Last loss: {self._losses[-1]}. Duration: {(timeit.default_timer() - t0):.2f} sec")
            
    def test(self, input_: torch.Tensor, output_: torch.Tensor) -> float:
        n = input_.shape[0]
        pred: torch.Tensor = self._generate_prediction(input_)
                
        im_segs: torch.Tensor = torch.zeros(
            n, 1, pred.shape[2], pred.shape[3]
        )

        for idx in range(n):
            im_segs[idx, :, :, :] = \
                generate_segmentation(
                    pred[idx].reshape(1, pred[idx].shape[0], pred[idx].shape[1], pred[idx].shape[2])
                ).reshape(1, pred[idx].shape[1], pred[idx].shape[2])

        return compute_dice(
            im_segs, 
            output_,
            num_classes = output_.shape[1],
            mdmc_average = "samplewise"
        ).item()
    
    def predict(self, input_: torch.Tensor):
        pred = self._generate_prediction(
            input_.reshape(1, input_.shape[0], input_.shape[1], input_.shape[2])
        )
        return generate_segmentation(pred).squeeze()
    
    def _generate_prediction(self, input_) -> torch.Tensor:
        with torch.no_grad():
            self._network.eval()
            if self._returns_dict:
                pred = self._network(input_)['out']
            else:
                pred = self._network(input_)
            
        return pred
            
    def _save_pickle_file(self, obj, name):
        if obj is None:
            print(f"Object for {name} is empty, returning")
            return
        
        _path = os.path.join(_pickle_path, name + ".pkl")
        if os.path.exists(_path):
            print(f"Pickle file {name} already exists, ignoring")
            return
        
        with open(_path, "wb") as fp: 
            torch.save(obj, fp)
            
    def _load_pickle_file(self, file):
        with open(file, "rb") as fp:
            return torch.load(fp)            