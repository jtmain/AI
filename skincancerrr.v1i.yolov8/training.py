from ultralytics import YOLO

# Build the model from scratch using the YOLOv8 nano configuration
model = YOLO("yolov8n.yaml")

# Train the model on the acne dataset with optimized settings,
# limited color variance, and explicit GPU usage.
model.train(
    data='data.yaml',            # Path to dataset configuration
    epochs=800,                  # Total training epochs with early stopping (patience=10)
    batch=16,                    # Batch size (adjustable depending on GH200 GPU memory)
    imgsz=1280,                  # High resolution to capture detailed acne features
    optimizer='AdamW',           # Optimizer with weight decay for improved generalization
    lr0=0.0003,                  # Initial learning rate
    lrf=0.02,                    # Final learning rate factor (final lr becomes lr0 * lrf)
    momentum=0.9,                # Momentum for optimizer
    weight_decay=1e-5,           # Weight decay to mitigate overfitting
    mosaic=False,                # Disable mosaic augmentation for focused learning
    mixup=False,                 # Disable mixup to avoid distortion on a small dataset
    hsv_s=0.1,                   # Limited HSV saturation augmentation to reduce color variance
    hsv_v=0.1,                   # Limited HSV value augmentation to reduce color variance
    translate=0.05,              # Small translation augmentation
    scale=0.9,                   # Minor scaling augmentation
    patience=0,                 # Early stopping patience to avoid overfitting
    project='runs/acne_exp_optimized',  # Directory to save training results
    name='exp_final',            # Name of the experiment
    save_period=10,              # Save model weights every 10 epochs
    verbose=True,                # Enable verbose logging for detailed output
    amp=True,                    # Enable automatic mixed precision for efficient training
    device='cuda:0',             # Explicitly specify GPU device
)
