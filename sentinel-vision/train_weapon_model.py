from ultralytics import YOLO

# Load the base model (We will use YOLOv8 Nano or YOLO26 Nano)
model = YOLO("yolo26n.pt")  

# Train the model on the weapon detection dataset
print("Starting transfer learning for Weapon Detection...")
results = model.train(
    data="d:/New folder/sentinel-vision/Dataset/weapon-detection.v1i.yolo26/data.yaml", 
    epochs=15,       # Reduced from 100 to fit in a 3 hour window
    imgsz=640,       # Image size for real-time cameras
    batch=16,        # Batch size (adjustable based on GPU RAM)
    device='cpu',    # 'cpu' for standard processor since CUDA is not installed
    project="Custom_Weapon_Model",
    name="run_1"
)

# Export the best weights
print("Training complete! The best model is saved at Custom_Weapon_Model/run_1/weights/best.pt")
