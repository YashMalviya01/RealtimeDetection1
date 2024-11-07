from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# For Webcam or Video
cap = cv2.VideoCapture(0)  # Change to the appropriate camera index or file path
cap.set(3, 680)  # Set width
cap.set(4, 540)  # Set height

# Load a better YOLO model (consider using a more accurate model like YOLOv8)
model = YOLO("yolov5s.pt")  # Change model if needed, e.g., yolov8 or custom-trained

# Automatically load class names from the model
classNames = model.names  # This ensures you get the correct classes based on the model

# Frame time initialization for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Confidence threshold (raise it to avoid false positives)
confidence = 0.6  # Adjust this threshold based on your model and environment

while True:
    new_frame_time = time.time()
    success, img = cap.read()  # Read frames from video/webcam
    if not success:
        break

    # Object detection with YOLO model
    results = model(img, stream=True, verbose=False)

    # Process each detected result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Ensure class index does not exceed the number of class names
            if cls < len(classNames) and conf > confidence:
                # Class-specific coloring
                color = (0, 255, 0) if classNames[cls] == 'mobile phone' else (255, 0, 0)  # Green for 'mobile phone'

                # Draw bounding box with corners
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

                # Put text (class name and confidence)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    # FPS calculation
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display the image with detections
    cv2.imshow("Image", img)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
