import torch
import torchmetrics
import torchvision
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, Dataset

from enum import Enum

device = \
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Models(Enum):
    RESNETV50 = 0
    UNET = 1
    CUSTOM = 2

def get_instrument_segmentation_model_base(model: Models, **kwargs) -> torch.nn.Module:
    if model == Models.RESNETV50:
        return torchvision.models.segmentation.deeplabv3_resnet50(
            num_classes = 3
        )
    elif model == Models.UNET:
        return smp.Unet(
            encoder_name = "resnet50",        
            encoder_weights = "imagenet",     
            in_channels = 3,                  
            classes = 3,                      
        )
    elif model == Models.CUSTOM:
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3,3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(32, 64, (5,5), padding = 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(64, 128, (7,7), padding = 2),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(128, 128, (3,3)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace = True),
            torch.nn.ConvTranspose2d(128, 128, (3,3)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace = True),
            torch.nn.ConvTranspose2d(128, 32, (5,5), stride = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.ConvTranspose2d(32, 3, (3,3), stride = 2),
            torch.nn.Upsample(kwargs['shape']),
            torch.nn.Softmax()
        )

def to_n_channel_pred(tensor_in: torch.Tensor) -> torch.Tensor:
    # Convert to integer
    tensor_out = torch.zeros(
        size = (tensor_in.shape[0], tensor_in.shape[2], tensor_in.shape[3]), 
        dtype = torch.uint8
    )
    
    # Mask channels w/ primary, with value = channel depth
    for n in range(1, tensor_in.shape[1]):
        tensor_out += (tensor_in[:, n, :, :] * n).int()
    
    return tensor_out

def to_n_channel_pred_single(tensor_in: torch.Tensor) -> torch.Tensor:
    # Convert to integer
    tensor_out = torch.zeros(
        size = (tensor_in.shape[1], tensor_in.shape[2]), 
        dtype = torch.uint8
    )
    
    # Mask channels w/ primary, with value = channel depth
    for n in range(1, tensor_in.shape[0]):
        tensor_out += (tensor_in[n, :, :] * n).int()
    
    return tensor_out

def generate_segmentation(im: torch.Tensor) -> torch.Tensor:
    return torch.argmax(im[0], 0, keepdim = True).cpu()

def reshape_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[0] == 3, \
        "Must have C == 3"
    tensor_out = torch.zeros(tensor.shape[1], tensor.shape[2], 3)
    
    # Input tensor is expected to be BGR, so swap B <--> R channels
    tensor_out[:, :, 2] = tensor[0, :, :]
    tensor_out[:, :, 1] = tensor[1, :, :]
    tensor_out[:, :, 0] = tensor[2, :, :]
    return tensor_out

def get_random_image(input_: DataLoader | Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(input_, DataLoader): dataset = input_.dataset
    elif isinstance(input_, Dataset): dataset = input_
    else: 
        print(f"Invalid input type: {input_}")
        return
    
    idx = torch.randint(0, len(dataset)-1, size=(1,)).item()

    return dataset[idx]

def compute_dice(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return torchmetrics.functional.dice(
        pred.int(), 
        to_n_channel_pred(target.data.cpu()), 
        ignore_index = 0, # 0: background
        # num_classes = target.data.shape[0],
        **kwargs
    )

