"""
python inference_speed_test.py \
    --model-variant mobileone \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --precision float32
"""

import argparse
import torch
from tqdm import tqdm
import torchvision
from model.mobileone import reparameterize_model
from model.model import MattingNetwork
from PIL import Image
import pandas as pd
from codecarbon import EmissionsTracker

torch.backends.cudnn.benchmark = True

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, default='mobileone',required=False)
        parser.add_argument('--resolution', type=int, default=(1920, 1080),required=False, nargs=2)
        parser.add_argument('--downsample-ratio', type=float, default=0.25,required=False)
        parser.add_argument('--checkpoint', type=str,
                            default='/home/sergi-garcia/Projects/Finetunning/Experiments/Decoder0/best.pth',
                            required=False)
        parser.add_argument('--sequence', type=str, default='/home/sergi-garcia/Projects/Finetunning/matting-data/HD/Brainstorm/0000/com/0010.png',required=False)
        parser.add_argument('--precision', type=str, default='float32',required=False)
        parser.add_argument('--disable-refiner', action='store_true')
        self.args = parser.parse_args()

    def init_model(self):
        self.device = 'cuda'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]
        self.rec = None, None, None, None, None
        self.model = MattingNetwork(self.args.model_variant)
        self.model.load_state_dict(
            torch.load(self.args.checkpoint, map_location=f'{self.device}'))
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        self.model = reparameterize_model(self.model)
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)

    # def loop(self):
    #     w, h = self.args.resolution
    #     src = torch.randn((1, 3, h, w), device=self.device, dtype=self.precision)
    #     with torch.no_grad():
    #         rec = None, None, None, None
    #         for _ in tqdm(range(1000)):
    #             fgr, pha, *rec = self.model(src, *rec, self.args.downsample_ratio)
    #             torch.cuda.synchronize()

    def loop(self):
        w, h = self.args.resolution
        if self.args.sequence is None:
            src = torch.randn((1, 3, h, w), device=self.device, dtype=self.precision)
        else:
            src = torchvision.transforms.ToTensor()(Image.open(self.args.sequence)).unsqueeze(0).to(self.device, dtype=self.precision)
        torch.cuda.synchronize()

        tracker = EmissionsTracker(project_name="InferenceSpeedTest", output_dir="./")
        tracker.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            rec = None, None, None, None, None
            start.record()
            for _ in tqdm(range(1000)):
                fgr, pha, *rec = self.model(src, *rec, self.args.downsample_ratio)
                torch.cuda.synchronize()  # Synchronize after each inference
            end.record()

        torch.cuda.synchronize()
        tracker.stop()

        elapsed_time = start.elapsed_time(end) / 1000  # Convert ms to seconds
        fps = 1000 / elapsed_time
        print(f"Total inference time for 1000 runs: {elapsed_time:.2f} s")
        print(f"Average inference time per frame: {elapsed_time / 1000:.4f} s")
        print(f"Model runs at approximately: {fps:.2f} FPS")

        df = pd.read_csv("./emissions.csv")

        last_run = df.iloc[-1]
        total_energy = last_run["energy_consumed"]  # in kWh
        co2_emissions = last_run["emissions"]  # in kg

        print("\nInference Environmental Impact Summary")
        print(f"   • Total energy used: {total_energy * 1000:.2f} Wh")
        print(f"   • Estimated CO₂ emissions: {co2_emissions * 1000:.3f} g")
        print(f"   • Equivalent to keeping a 10 W LED bulb on for {total_energy * 100:.1f} minutes")

if __name__ == '__main__':
    InferenceSpeedTest()