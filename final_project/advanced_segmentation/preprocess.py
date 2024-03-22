import glob
import os
import pickle
import numpy as np
import cv2 as cv
import torch

from enum import Enum

class DatasetType(Enum):
    TRAINING = 0
    TESTING = 1

DATASET_PATH_TRAINING = \
    "..\\..\\..\\datasets\\unzipped\\endovis\\Segmentation_Robotic_Training\\Training" # edit as needed
DATASET_PATH_TESTING = \
    "..\\..\\..\\datasets\\unzipped\\endovis\\Segmentation_Robotic_Training\\Testing" # edit as needed

from pytorch_utils import device

def process_video(cap: cv.VideoCapture):
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            print(f"End of vid, total frames: {frame_count}")
            break
        frame_count += 1
        frames.append(frame)
        
    cap.release()
        
    return frames

def process_multiple_instruments(cap_one: cv.VideoCapture, cap_two: cv.VideoCapture):
    frames = []
    frame_count = 0
    
    while True:
        ret1, frame1 = cap_one.read()
        ret2, frame2 = cap_two.read()
        if frame1 is None or frame2 is None:
            print(frame1 is None, frame2 is None)
            print(f"End of vid, total frames: {frame_count}")
            break
        frame_count += 1
        frames.append(frame1 + frame2)
        
    cap_one.release()
    cap_two.release()
        
    return frames

def load_endovis_videos(dataset: DatasetType, training_single_instruments_only: bool = False):
    if dataset == DatasetType.TRAINING: path = DATASET_PATH_TRAINING
    else: path = DATASET_PATH_TESTING
    
    input_videos = glob.glob(path + "\\**\\*Video*.avi")
    output_videos = glob.glob(path + "\\**\\*Segmentation*.avi")
    
    if len(input_videos) == 0 or len(output_videos) == 0:
        raise RuntimeError(f"Invalid path inputted, double check data is present ({path})")
        
    input_frames: list[np.ndarray] = []
    output_frames: list[np.ndarray] = []
    
    input_frames_raw_pkl_path = \
        os.path.join(os.path.dirname(__file__), f"pickled\\input_frames_raw_{dataset.name.lower()}_" + ("single" if training_single_instruments_only else 'full') + ".pkl")
    output_frames_raw_pkl_path = \
        os.path.join(os.path.dirname(__file__), f"pickled\\output_frames_raw_{dataset.name.lower()}_" + ("single" if training_single_instruments_only else 'full') + ".pkl")

    if os.path.exists(input_frames_raw_pkl_path) and os.path.exists(output_frames_raw_pkl_path):
        with open(input_frames_raw_pkl_path, "rb") as fp:
            input_frames = pickle.load(fp)

        with open(output_frames_raw_pkl_path, "rb") as fp:
            output_frames = pickle.load(fp)
    else:
        cap_left = None
        # TODO: support vid w/ two instruments...
        for vids, container in zip((input_videos, output_videos), (input_frames, output_frames)):
            for vid in vids:
                print(f"Processing {vid}")
                if "Left" in vid:
                    if training_single_instruments_only and dataset == DatasetType.TRAINING: 
                        continue # ignore left/right segmentation for training data
                    cap_left = cv.VideoCapture(vid)
                    continue
                elif "Right" in vid:
                    if training_single_instruments_only and dataset == DatasetType.TRAINING: 
                        continue # ignore left/right segmentation for training data 
                    cap_right = cv.VideoCapture(vid)
                    assert cap_left is not None, "Should have seen left vid"
                    container.extend(process_multiple_instruments(cap_left, cap_right))
                    cap_left = None
                    cap_right = None
                else:
                    if training_single_instruments_only and dataset == DatasetType.TRAINING and "Dataset1" in vid:
                        continue # HACK: dataset 1 has multiple instruments, so skip source vid 
                    cap = cv.VideoCapture(vid)
                    container.extend(process_video(cap))

        with open(input_frames_raw_pkl_path, "wb") as fp:
            pickle.dump(input_frames, fp)

        with open(output_frames_raw_pkl_path, "wb") as fp:
            pickle.dump(output_frames, fp)
            
    return (input_frames, output_frames)
    
def preprocess_source_endovis_images(imgs: list[np.ndarray], new_size: tuple[int, int]) -> torch.Tensor:
    input_frames_preprocessed = torch.zeros(
        [len(imgs), imgs[0].shape[2], new_size[0], new_size[1]]
    ).to(device)
    
    for idx in range(input_frames_preprocessed.shape[0]):
        input_frame_small = torch.from_numpy(
            cv.resize(
                imgs[idx] / 255.0, # normalize image 
                (new_size[1], new_size[0])
            )
        )
        
        input_frames_preprocessed[idx, 0, :, :] = input_frame_small[:, :, 0]
        input_frames_preprocessed[idx, 1, :, :] = input_frame_small[:, :, 1]
        input_frames_preprocessed[idx, 2, :, :] = input_frame_small[:, :, 2]
        
    return input_frames_preprocessed
        
def preprocess_endovis_target_images(imgs: list[np.ndarray], new_size: tuple[int, int]) -> torch.Tensor:
    output_frames_preprocessed = torch.zeros(
        [len(imgs), 3, new_size[0], new_size[1]]
    ).to(device)
    
    for idx in range(output_frames_preprocessed.shape[0]):
        output_frame_small = cv.resize(imgs[idx], (new_size[1], new_size[0]))
        eef = torch.from_numpy((cv.inRange(cv.cvtColor(output_frame_small, code = cv.COLOR_BGR2GRAY), 65, 115) / 255.0).astype(np.bool_))
        shaft = torch.from_numpy((cv.inRange(cv.cvtColor(output_frame_small, code = cv.COLOR_BGR2GRAY), 116, 165) / 255.0).astype(np.bool_))
        background = ~torch.bitwise_or(eef, shaft)
        
        output_frames_preprocessed[idx, :, :, :] = torch.stack((background.int(), eef.int(), shaft.int()), dim = 0)
        
    return output_frames_preprocessed