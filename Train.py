from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')  # or another pre-trained model if you prefer

# Train the model using your dataset and configuration
model.train(data='Datasets/SplitData/data0ffline.yaml ', epochs=300)

