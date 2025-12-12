import numpy as np
import struct
import sys
import os
import torch
import glob
from tqdm import tqdm

def export_voices(voices_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for pt_path in tqdm(glob.glob(voices_path + "/*.pt")):
        voice = torch.load(pt_path, weights_only=True)
        voice_name = os.path.splitext(os.path.basename(pt_path))[0]
        voice_npy = voice.numpy()
        voice_npy.tofile(os.path.join(output_path, voice_name + ".bin"))

if __name__ == "__main__":
    export_voices("../checkpoints/voices", "voices")
