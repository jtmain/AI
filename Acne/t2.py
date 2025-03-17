#!/usr/bin/env python3
import os
import shutil
from ultralytics import YOLO
import torch

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

# Define experiment directories
project_dir = 'runs/acne_exp_optimized'
exp_name = 'exp_finetune_900'
exp_dir = os.path.join(project_dir, exp_name)

# Remove any existing experiment folder to avoid loading previous config files
if os.path.exists(exp_dir):
    print(f"Removing existing experiment directory: {exp_dir}")
    shutil.rmtree(exp_dir)

# Load YOLOv8 pretrained model
model = YOLO("yolov8s.pt")  # Using pretrained weights

# Optimize anchors for your acne dataset (this tunes anchor sizes for better detection)

# IMPORTANT: Do NOT reload the model after tuning, to keep the tuned anchors and avoid resetting training settings.
model.tune(data='data.yaml', task='detect', optimizer='auto', iterations=1)

# (Optional) If your ultralytics version allows, you can try overriding internal args:
# model.args.epochs = 500
# model.args.patience = 0

# Start training with custom settings ensuring 500 epochs and zero patience:
print("\n\n\n\n\n\n\n\n\n\n\n\n\n Starting full training with custom settings...")
model.train(
    data='data.yaml',
    epochs=500,              # Force training to run for 500 epochs
    batch=16,                # Adjusted for GH200
    imgsz=960,              # Higher resolution for better accuracy
    lr0=0.00007,
    lrf=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    freeze=0,
    patience=0,              # Disable early stopping
    mosaic=True,
    mixup=True,
    hsv_s=0.15,
    hsv_v=0.15,
    scale=0.9,
    project=project_dir,
    name=exp_name,
    save_period=10,
    verbose=True,
    amp=True,
    device='cuda:0',
)

print("✅ Training completed! Check results in:", exp_dir)

