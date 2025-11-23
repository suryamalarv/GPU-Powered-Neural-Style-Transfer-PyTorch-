# GPU-Powered Neural Style Transfer (PyTorch)

## Overview
This project demonstrates GPU acceleration using PyTorch. 
It applies artistic style transfer to an image — combining the content of one image with the style of another — using a deep neural network (VGG19) that runs entirely on GPU.

## Requirements
- Python 3.9+
- CUDA-compatible GPU
- NVIDIA driver + CUDA toolkit
- PyTorch with CUDA support

## Installation
```bash
git clone <repo_url>
cd gpu-style-transfer
python -m venv venv
venv\Scripts\activate      
pip install -r requirements.txt
=======
```
## Running the Program

After building the project, you can run the program using the following command:
```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
python style_cpu_vs_gpu_test.py
python style_transfer.py    
```
