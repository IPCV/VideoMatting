# UPF Robust Video Matting (RVM) - Complete Documentation

This branch contains the implementation of enhanced versions of Robust Video Matting (RVM) with MobileOne and Mamba backbone architectures for improved video matting performance.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [HPC Cluster Deployment](#hpc-cluster-deployment)

## Overview

This project extends the original Robust Video Matting (RVM) framework with two modern backbone architectures:

- **MobileOne**: An efficient mobile-friendly architecture optimized for real-time inference
- **Mamba**: A state-space model-based architecture (VSSM) for enhanced temporal modeling

Both variants maintain the RVM framework's recurrent architecture for video matting while improving feature extraction capabilities.

## Project Structure
```

RobustVideoMatting/
├── model/                          # Model architecture definitions
│   ├── model.py                    # Main matting network
│   ├── mobileone.py                # MobileOne backbone implementation
│   ├── vssm_encoder.py             # Mamba (VSSM) encoder implementation
│   ├── decoder.py                  # Decoder architecture
│   ├── lraspp.py                   # Lite R-ASPP segmentation head
│   ├── deep_guided_filter.py       # Deep guided filter module
│   ├── fast_guided_filter.py       # Fast guided filter module
│   └── weights/                    # Pretrained model weights
├── dataset/                        # Dataset utilities and loaders
├── documentation/                  # Additional documentation
├── evaluation/                     # Evaluation scripts and metrics
├── Experiments/                    # Experiment configurations and results
├── inference_results/              # Output directory for inference results
├── .rvm/                          # RVM-specific configurations
├── train.py                       # Original training script
├── mamba_train.py                 # Mamba variant training script
├── mamba_train.sh                 # Mamba training shell script for HPC
├── mobileone_train.py             # MobileOne variant training script
├── mobileone_train.sh             # MobileOne training shell script for HPC
├── train_config.py                # Training configuration parameters
├── train_loss.py                  # Loss functions for training
├── inference.py                   # Standard inference script
├── custom_inference.py            # Custom inference pipeline for batch processing
├── inference_utils.py             # Inference utility functions
├── inference_speed_test.py        # Performance benchmarking script
├── hubconf.py                     # PyTorch Hub configuration
├── create_singularity_env.def     # Singularity container definition
└── requirements_training.txt      # Python dependencies
```
## Requirements

### Python Environment
- Python 3.13.2
- Package manager: virtualenv

### Key Dependencies
- PyTorch (with CUDA support recommended)
- torchvision
- numpy
- opencv-python
- Pillow
- pandas
- tqdm
- pyyaml
- tensorboard (for training monitoring)

See `requirements_training.txt` for complete dependency list.

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RobustVideoMatting
   ```

2. **Create virtual environment**
   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_training.txt
   ```

### HPC Cluster Setup

For deployment on HPC clusters using Singularity:

1. **Build Singularity image**
   ```bash
   sudo singularity build myenv.sif create_singularity_env.def
   ```

2. **Transfer to HPC system**
   ```bash
   # Use your HPC's transfer machine (e.g., BSC transfer nodes)
   scp myenv.sif user@transfer1.bsc.es:/path/to/destination/
   ```

**Note**: Any changes to the local environment require rebuilding the `.sif` image and transferring it to the HPC system.

## Training

### Training Configuration

Training parameters are defined in `train_config.py`. Key configurations include:
- Dataset paths and preprocessing parameters
- Model architecture selection (variant)
- Batch size, learning rate, and optimization settings
- Loss function weights
- Checkpoint and logging directories

### MobileOne Training

**From IDE or terminal:**
```
bash
python mobileone_train.py
```
**On HPC cluster:**
```
bash
sbatch mobileone_train.sh
```
The MobileOne variant offers:
- Efficient mobile deployment
- Reparameterization for inference optimization
- Real-time performance capabilities

### Mamba Training

**From IDE or terminal:**
```
bash
python mamba_train.py
```
**On HPC cluster:**
```
bash
sbatch mamba_train.sh
```
The Mamba variant provides:
- Advanced temporal modeling with state-space models
- Long-range dependency capture
- Enhanced video sequence understanding

### Training Features

Both training scripts support:
- Multi-stage training with progressive refinement
- Automatic checkpoint saving and resuming
- TensorBoard logging for monitoring
- Distributed training capabilities
- Custom loss functions (defined in `train_loss.py`)
- Data augmentation and preprocessing

## Inference

### Standard Inference

Use `inference.py` for standard video matting inference:
```
python
from inference import convert_video
from model import MattingNetwork

model = MattingNetwork(variant='mamba')  # or 'mobileone'
model.load_state_dict(torch.load('path/to/weights.pth'))

convert_video(
    model,
    input_source='path/to/video.mp4',
    output_type='video',
    output_composition='output.mp4',
    downsample_ratio=None,
    seq_chunk=12
)
```
### Custom Batch Inference

The `custom_inference.py` script provides a pipeline for batch processing multiple videos:
```
bash
python custom_inference.py \
    --dataset Brainstorm \
    --data-dir /path/to/input/data \
    --data-out inference_results/output \
    --variant mamba \
    --weights model/weights/stage4.pth
```
**Parameters:**
- `--dataset`: Dataset name or identifier
- `--data-dir`: Input directory containing video sequences
- `--data-out`: Output directory for results
- `--variant`: Model variant ('mamba' or 'mobileone')
- `--weights`: Path to trained model weights

**Output Structure:**
```

output_directory/
└── sequence_name/
    ├── pha/        # Alpha matte (transparency)
    ├── fgr/        # Foreground
    └── com/        # Composition
```
### Inference Utilities

`inference_utils.py` provides helper functions for:
- Video reading and writing
- Frame preprocessing and postprocessing
- Batch processing optimization
- Memory management for large videos

### Performance Testing

Benchmark inference speed with:
```
bash
python inference_speed_test.py
```
This script measures:
- Frames per second (FPS)
- Latency per frame
- GPU memory usage
- Comparison between variants

## Model Architecture

### Core Components

1. **MattingNetwork** (`model/model.py`)
   - Main network orchestrating encoder, decoder, and refinement
   - Recurrent architecture for temporal consistency
   - Supports multiple backbone variants

2. **Encoders**
   - **MobileOne** (`model/mobileone.py`): Efficient reparameterizable blocks
   - **Mamba/VSSM** (`model/vssm_encoder.py`): State-space model encoder

3. **Decoder** (`model/decoder.py`)
   - Upsampling and feature fusion
   - Multi-scale feature integration
   - Recurrent refinement modules

4. **Segmentation Head**
   - **LRASPP** (`model/lraspp.py`): Lite Reduced Atrous Spatial Pyramid Pooling
   - Efficient semantic segmentation

5. **Refinement Modules**
   - **Deep Guided Filter** (`model/deep_guided_filter.py`): Detail-preserving refinement
   - **Fast Guided Filter** (`model/fast_guided_filter.py`): Efficient edge-aware filtering

### Variant Comparison

| Feature | MobileOne | Mamba |
|---------|-----------|-------|
| Primary Focus | Efficiency & Speed | Temporal Modeling |
| Deployment Target | Mobile/Edge Devices | High-Performance Systems |
| Key Advantage | Real-time Processing | Long-range Dependencies |
| Reparameterization | Yes | No |
| Memory Usage | Lower | Higher |

## Loss Functions

Defined in `train_loss.py`, the training uses a combination of:
- **Alpha Loss**: L1/L2 loss on alpha matte
- **Composition Loss**: Photometric loss on composited output
- **Temporal Loss**: Consistency across frames
- **Perceptual Loss**: Feature-based loss for realism
- **Gradient Loss**: Edge preservation

## PyTorch Hub Integration

Load pretrained models directly via PyTorch Hub using `hubconf.py`:
```
python
import torch

# Load MobileOne variant
model = torch.hub.load('repo', 'mobileone_rvm', pretrained=True)

# Load Mamba variant
model = torch.hub.load('repo', 'mamba_rvm', pretrained=True)
```
## Evaluation

The `evaluation/` directory contains scripts for:
- Quantitative metrics (MSE, SAD, MAD, Gradient error)
- Qualitative visualization
- Benchmark comparisons
- Ablation studies

## Best Practices

### Training Tips
1. Start with pretrained backbone weights when available
2. Use multi-stage training: low-resolution → high-resolution
3. Monitor TensorBoard for loss curves and visual outputs
4. Adjust `seq_chunk` based on GPU memory
5. Enable mixed precision training for faster convergence

### Inference Optimization
1. Use `reparameterize_model()` for MobileOne before inference
2. Consider TorchScript compilation with `torch.jit.script()`
3. Adjust `downsample_ratio` for speed/quality tradeoff
4. Batch process frames when possible with `seq_chunk`
5. Enable CUDA benchmarking: `torch.backends.cudnn.benchmark = True`

### HPC Workflow
1. Develop and test locally with small dataset
2. Prepare and validate Singularity container
3. Transfer data and container to HPC
4. Submit batch jobs with appropriate resource requests
5. Monitor job progress and adjust as needed

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size or sequence chunk size
- Lower input resolution with `input_resize` parameter
- Enable gradient checkpointing (if implemented)

**Slow Training**
- Verify CUDA is available and being used
- Enable `torch.backends.cudnn.benchmark = True`
- Check data loading bottlenecks
- Consider distributed training

**Poor Matting Quality**
- Ensure proper preprocessing of input data
- Check if pretrained weights are loaded correctly
- Verify training loss convergence
- Adjust loss function weights

## Additional Resources

- Original RVM: See `README.md` and `README_zh_Hans.md`
- Documentation folder: Additional technical details
- Experiments folder: Experimental configurations and results

---

**Maintained by**: UPF Research Team  
**Last Updated**: 2025-11-12
``'