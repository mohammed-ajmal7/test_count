import os
import torch
import cv2
import numpy as np
from PIL import Image

# Step 1: Load the YOLOv5 Model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Step 2: Set the Path to Your Image
image_path = 'D:\BE-CSE\project\Count-the-number-of-people-currently-in-the-room\images\img3.jpg'  #change the img for processing different img
if not os.path.exists(image_path):
    print(f"Error: The file at path {image_path} does not exist.")
    exit()

# Step 3: Load and Preprocess the Image
try:
    image = Image.open(image_path)
    image = np.array(image)
    print("Image loaded and preprocessed successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Step 4: Perform Inference
try:
    results = model(image)
    print("Inference performed successfully.")
except Exception as e:
    print(f"Error performing inference: {e}")
    exit()

# Step 5: Process the Results
try:
    detections = results.xyxy[0]  # Extract predictions
    person_class = 0  # Class index for 'person' in COCO dataset

    # Count the number of people detected
    person_count = sum([1 for det in detections if int(det[5]) == person_class])
    print(f'Number of people detected: {person_count}')

    # Display the image with bounding boxes
    results.show()

    # Optionally, if you want to display the image with OpenCV:
    image_with_boxes = np.squeeze(results.render())  # Rendered image with bounding boxes
    cv2.imshow('Detected People', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error processing results: {e}")
