import os
import argparse
import torch
from tqdm import tqdm
from model import MattingNetwork
from model.mobileone import reparameterize_model
from pathlib import Path
from inference import convert_video


parser = argparse.ArgumentParser()
# Matting dataset
parser.add_argument('--dataset', type=str, default='VideoMatteLR', required=False)
parser.add_argument('--data-dir', type=str, default='matting-data/LR', required=False)
parser.add_argument('--data-out', type=str, default='inference_results', required=False)
parser.add_argument('--variant', type=str, default='mobileone', choices=['mobileone'])
parser.add_argument('--weights', type=str, required=True)

args = parser.parse_args()

_ROOT_DIR = Path(__file__).resolve().parent
_DATASET_DIR = os.path.join(_ROOT_DIR, args.data_dir, args.dataset)
_DATASET_OUT = os.path.join(_ROOT_DIR, args.data_out, args.dataset)
_WEIGHTS_ = os.path.join(_ROOT_DIR, args.weights)

if not os.path.isdir(_DATASET_OUT):
    os.makedirs(_DATASET_OUT, exist_ok=True)

torch.backends.cudnn.benchmark = True

model = MattingNetwork(args.variant)
model.load_state_dict(torch.load(_WEIGHTS_))
model = model.eval()
model = reparameterize_model(model)
model = torch.jit.script(model)
model = torch.jit.freeze(model)
model = model.to('cuda')

def inference_pipeline(input_dir: str | None, set_dir: str | None, output_dir: str | None) -> None:
    """
    Custom inference pipeline for extracting predictions on RVM with custom datasets.
    :param input_dir: optional parameter that should contain a string path to a video to evaluate
    :param set_dir: in case of several videos, this parameter should contain the path to the different directory with
    all the videos extracted in the correspondent format
    :param output_dir: directory where you want to save your predictions
    """

    if input_dir:
        sequence_dir = input_dir
    elif set_dir:
        sequence_dir = os.path.join(set_dir, 'fgr')
    sequence_out_alpha = os.path.join(output_dir, 'pha')
    sequence_out_fgr = os.path.join(output_dir, 'fgr')
    sequence_out_com = os.path.join(output_dir, 'com')

    convert_video(
        model,  # The model, can be on any device (cpu or cuda).
        input_source=sequence_dir,  # A video file or an image sequence directory.
        output_type='png_sequence',  # Choose "video" or "png_sequence"
        output_composition=sequence_out_com,  # File path if video; directory path if png sequence.
        output_alpha=sequence_out_alpha,
        output_foreground=sequence_out_fgr,
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
        seq_chunk=1  # Process n frames at once for better parallelism.
    )


# sequence_list = ['0000']
sequence_list = os.listdir(_DATASET_DIR)

for sequence in tqdm(sequence_list, total=len(sequence_list), dynamic_ncols=True, desc='Inference on dataset'):
    input_dir = os.path.join(_DATASET_DIR, sequence)
    output_dir = os.path.join(_DATASET_OUT, sequence)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.makedirs(os.path.join(output_dir, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'com'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fgr'), exist_ok=True)
    inference_pipeline(input_dir = None, set_dir=input_dir, output_dir=output_dir)


