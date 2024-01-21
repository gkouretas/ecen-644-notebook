"""
George Kouretas -- ECEN 644
Module for unpacking CIFAR datasets
"""

import pickle
import os
import numpy as np

# path to CIFAR dataset
_PATH = "datasets/cifar-10-batches-py"

# Container for metadata
_META_DATA: dict = None

# Image metadata
_NUM_BYTES  : int                  = 3072
_IMAGE_SHAPE: tuple[int, int, int] = (32, 32, 3)
_IMAGE_DTYPE                       = np.uint8

import copy
from typing import Iterable, TypedDict

# TypedDict for CIFAR dataset
class UnpackedCIFARData(TypedDict):
    batch_label: str
    labels: list[str]
    data: np.ndarray
    filenames: list[str]

def get_metadata():
    global _META_DATA
    
    # if metadata is uninitialized, unpack it
    if _META_DATA is None: 
        _META_DATA = _unpack_metadata()
        
    # return a copy of the metadata
    return copy.deepcopy(_META_DATA)

def get_label_names() -> list[str]:
    md = get_metadata()
    return md['label_names']

def label_name_to_index(label: str): return get_label_names().index(label)
def label_index_to_name(label: int): return get_label_names()[label]

def get_images_by_label(data: UnpackedCIFARData, label: int | str):
    if isinstance(label, str):
        label = label_name_to_index(label)
                
    labels = np.array(data['labels'])
    label_indexes = np.where(labels == label)[0]
    
    return [data["data"][ii] for ii in label_indexes], [data["filenames"][ii] for ii in label_indexes]

def unpack(name: str) -> UnpackedCIFARData:
    file = os.path.join(_PATH, name)    # get full path to dataset
    data = _load_pickled_file(file)     # load pickled file
        
    # initialize empty dictionary to place decoded data in
    decoded_data = {}                   
            
    for k, v in data.items():
        # decode data
        key = \
            k.decode() if isinstance(k, bytes) else k
            
        if isinstance(v, bytes):
            v = v.decode() # decode bytes
        elif isinstance(v, Iterable) and isinstance(v[0], bytes):
            _dtype = type(v)
            v = _dtype([_v.decode() for _v in v]) # iterate through elements and decode data. 
        
        decoded_data[key] = v
                
    return decoded_data
            
def data_to_rgb_image(data: np.ndarray[_IMAGE_DTYPE]): 
    """Convert flat numpy array to RGB image

    Args:
        data (np.ndarray): (_description_)
    """
    assert np.shape(data) == (_NUM_BYTES,), \
        f"Invalid size of data: {np.shape(data)}"
        
    # Initialize array 
    rgb = np.zeros(_IMAGE_SHAPE, dtype = _IMAGE_DTYPE)
    
    rgb[:, :, 0] = \
        data[:(_NUM_BYTES//3)].reshape(_IMAGE_SHAPE[:2])
        
    rgb[:, :, 1] = \
        data[(_NUM_BYTES//3):(2*_NUM_BYTES//3)].reshape(_IMAGE_SHAPE[:2])
        
    rgb[:, :, 2] = \
        data[(2*_NUM_BYTES//3):].reshape(_IMAGE_SHAPE[:2])
        
    return rgb
        
# ----- Private functions -----
        
def _load_pickled_file(file: str) -> dict:
    if os.path.exists(file):
        # unpack dataset, if it exists
        with open(file, 'rb') as fp:
            return pickle.load(fp, encoding = 'bytes')
    else:
        raise FileNotFoundError(f"File not found: {file}")
    
def _unpack_metadata():
    file = os.path.join(_PATH, "batches.meta")    
    # unpack dataset, if it exists
    with open(file, 'rb') as fp:
        data: dict = pickle.load(fp, encoding = 'bytes')
        
    decoded_data = {}
        
    for k, v in data.items():
        key = \
            k.decode() if isinstance(k, bytes) else k
            
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, Iterable) and isinstance(v[0], bytes):
            _dtype = type(v)
            v = _dtype([_v.decode() for _v in v])
            
        decoded_data[key] = v
        
    return decoded_data