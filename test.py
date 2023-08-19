from ultralytics import YOLO
import cv2
import math

# For test image inference
model = YOLO('yolov8n.pt')
result = model('Sam.jpg',show= True)
cv2.waitKey(0)