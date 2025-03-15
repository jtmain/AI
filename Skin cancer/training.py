from ultralytics import YOLO

# Build the model from scratch using the YOLOv8s configuration
model = YOLO("yolov8s.yaml")

# Train the model on the skin cancer dataset with enhanced augmentation,
# limited color variance, and explicit GPU usage.
model.train(
    data='data_skin_cancer.yaml',    # Path to the skin cancer dataset configuration
    epochs=300,                      # Increased epochs for thorough training
    batch=32,                        # Larger batch size to leverage the increased dataset size and GPU memory
    imgsz=1280,                      # High resolution to capture detailed skin lesion features
    optimizer='AdamW',               # Optimizer with adaptive learning and weight decay
    lr0=0.0003,                      # Initial learning rate
    lrf=0.01,                        # Final learning rate factor (final lr = lr0 * lrf)
    momentum=0.937,                  # Slightly higher momentum for improved stability
    weight_decay=0.0005,             # Increased weight decay for better regularization on larger data
    mosaic=True,                     # Enable mosaic augmentation to mix images and improve generalization
    mixup=True,                      # Enable mixup augmentation for enhanced robustness
    hsv_s=0.1,                       # Limited HSV saturation augmentation to restrict color variance
    hsv_v=0.1,                       # Limited HSV value augmentation to restrict color variance
    translate=0.1,                   # Increased translation augmentation (small image shifts)
    scale=0.5,                       # More aggressive scaling to simulate variations in lesion size
    patience=20,                     # Extended early stopping patience to allow longer training
    project='runs/skin_cancer_exp_optimized',  # Directory to save training results
    name='exp_final',                # Name of the experiment
    save_period=10,                  # Save model weights every 10 epochs
    verbose=True,                    # Enable verbose logging for detailed output
    amp=True,                        # Enable automatic mixed precision for efficient training
    device='cuda:0',                 # Explicitly specify GPU device
)
