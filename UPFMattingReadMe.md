# UPF Robust Video Matting (RVM) Versions

This branch contain the main files for training and evaluate MobileOne and Mamba versions of RVM. 

### Training files

For training either Mamba or MobileOne you will find files with their specific name on them. This files can be run from your IDE or from your terminal. 
Running these files from a HPC cluster require a singularity image that can be built with custom_singularity_env.def

```sh
sudo singularity build myenv.sif custom_singularity_env.def
```

