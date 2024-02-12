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

def get_image_size() -> tuple[int, int, int]: return _IMAGE_SHAPE
def get_image_width() -> int: return get_image_size()[0]
def get_image_height() -> int: return get_image_size()[1]

def get_metadata() -> dict:
    """Returns metadata of CIFAR datasets. Will return global `_META_DATA` variable once initialized

    Returns:
        dict: Dictionary containing metadata for CIFAR datasets
    """
    global _META_DATA
    
    # if metadata is uninitialized, unpack it
    if _META_DATA is None: 
        _META_DATA = _unpack_metadata()
        
    # return a copy of the metadata
    return copy.deepcopy(_META_DATA)

def get_label_names() -> list[str]:
    """Returns label names for CIFAR datasets

    Returns:
        list[str]: List of label names
    """
    md = get_metadata()
    return md['label_names']

def label_name_to_index(label: str) -> int: 
    """Returns label index, given the corresponding string. Must be within bounds.

    Args:
        label (str): Label name

    Returns:
        int: Label index
    """
    return get_label_names().index(label)

def label_index_to_name(label: int) -> str: 
    """Returns label name, given the corresponding index. Must be valid name.

    Args:
        label (int): Label index

    Returns:
        str: Label name
    """
    return get_label_names()[label]

def get_images_by_label(data: UnpackedCIFARData, label: int | str) -> tuple[list[np.ndarray], list[str]]:
    """Returns images and associated filenames for a given label and dataset.

    Args:
        data (UnpackedCIFARData): Unpacked CIFAR dataset
        label (int | str): Label, represented as integer or string

    Returns:
        tuple[list[np.ndarray], list[str]]: Length 2 tuple with image data (as 1D-array) and image filename
    """
    # Convert label to integer if passed as string
    if isinstance(label, str):
        label = label_name_to_index(label)
                
    # Get indexes where desired label is
    label_indexes = np.where(np.array(data['labels']) == label)[0]
    
    return [data["data"][ii] for ii in label_indexes], [data["filenames"][ii] for ii in label_indexes]

def unpack(name: str) -> UnpackedCIFARData:
    """Unpack CIFAR dataset. Unpickles data and decode byte-strings.

    Args:
        name (str): Name of dataset. Must exist in datasets/cifar-10-batches-py

    Returns:
        UnpackedCIFARData: Unpacked dataset.
    """
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
        elif isinstance(v, list):
            # NOTE: only works on 1D list, which is all that we are expecting w/ given datasets
            for ii, element in enumerate(v):
                if isinstance(element, bytes):
                    v[ii] = element.decode()
        
        decoded_data[key] = v
                
    return decoded_data
            
def data_to_rgb_image(data: np.ndarray) -> np.ndarray: 
    """Convert flat numpy array to RGB image (size=32x32x3)

    Args:
        data (np.ndarray): Input data (1D array)

    Returns:
        np.ndarray: RGB image array
    """
    assert np.shape(data) == (_NUM_BYTES,), \
        f"Invalid size of data: {np.shape(data)}"
        
    # Initialize array 
    rgb = np.zeros(_IMAGE_SHAPE, dtype = _IMAGE_DTYPE)
    
    # Red channel
    rgb[:, :, 0] = \
        data[:(_NUM_BYTES//3)].reshape(_IMAGE_SHAPE[:2])
        
    # Green channel
    rgb[:, :, 1] = \
        data[(_NUM_BYTES//3):(2*_NUM_BYTES//3)].reshape(_IMAGE_SHAPE[:2])
        
    # Blue channel
    rgb[:, :, 2] = \
        data[(2*_NUM_BYTES//3):].reshape(_IMAGE_SHAPE[:2])
        
    return rgb
        
# ----- Private functions -----
        
def _load_pickled_file(file: str) -> dict:
    """Helper function to load pickled file

    Args:
        file (str): File path to pickled file

    Raises:
        FileNotFoundError: Raises if file is not found at specified path

    Returns:
        dict: Unpacked data
    """
    if os.path.exists(file):
        # unpack dataset, if it exists
        with open(file, 'rb') as fp:
            return pickle.load(fp, encoding = 'bytes')
    else:
        raise FileNotFoundError(f"File not found: {file}")
    
def _unpack_metadata() -> dict:
    """Helper function for unpacking metadata. Should not be called directly, use `get_metadata` instead.

    Returns:
        dict: Unpacked meta data
    """
    
    # Unpack dataset
    data = _load_pickled_file(os.path.join(_PATH, "batches.meta"))
        
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