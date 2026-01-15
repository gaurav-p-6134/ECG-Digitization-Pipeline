
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import kagglehub
seed = 0
CUDA0 = "cuda:0"
deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed, disable_list=['cuda_block'])

import sys
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')

import os
import traceback
from pathlib import Path
from shutil import copyfile
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

if_submit = os.getenv('KAGGLE_IS_COMPETITION_RERUN')

if if_submit:
    test_meta = Path("/kaggle/input/physionet-ecg-image-digitization/test.csv")
    test_dir = Path("/kaggle/input/physionet-ecg-image-digitization/test")
else:
    test_meta = Path("/kaggle/input/physio-test-fake-dataset/test_fake/test.csv")
    test_dir = Path("/kaggle/input/physio-test-fake-dataset/test_fake")

valid_df = pd.read_csv(test_meta)
valid_df['id'] = valid_df['id'].astype(str) 
valid_id = valid_df['id'].unique().tolist()

FLOAT_TYPE = torch.float32

global_dict = {
    "stage0_dir": "/kaggle/working/stage0",
    "stage1_dir": "/kaggle/working/stage1",
    "stage2_dir": "/kaggle/working/stage2",
}