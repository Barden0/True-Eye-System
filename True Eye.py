import cv2
import math
import csv
import cvzone
import tkinter as tk
from ultralytics import YOLO
import threading
from PIL import Image, ImageTk
from tkinter import filedialog

# Load YOLO model and class names (object labels) from files
model = YOLO('yolov8n.pt')
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Global variables to control webcam feed and frame
is_webcam_running = False
frame = None
detected_objects = []


# Function to perform object detection
def detect_objects():
    global is_webcam_running, frame, detected_objects
    cap = cv2.VideoCapture(0)

    while is_webcam_running:
        # Capture frame from webcam
        _, frame = cap.read()
        # Perform object detection using YOLO model on the frame
        result = model(frame, stream=True)

        for info in result:
            # Loop through each detected object
            boxes = info.boxes
            for box in boxes:
                # Extract object information - class name, confidence, and bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                names = box.cls[0]

                # Convert confidence score to percentage
                conf = math.ceil(conf * 100)
                if conf > 65:
                    # If confidence is high enough, consider the object detected
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    names = int(names)

                    # Draw a bounding box and display the object label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{classnames[names]} {conf}%', [x1 + 8, y1 - 12],
                                       scale=1.5, thickness=2)

                    # Collect detected object information in a list
                    detected_objects.append([classnames[names], conf, x1, y1, x2, y2])

        if frame is not None:
            # Display the frame with detected objects on the Tkinter window
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            video_label.config(image=image)
            video_label.image = image

    # Save detected objects to a CSV file when webcam stops
    save_to_csv(detected_objects)

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()


# Function to start the webcam feed and object detection
def start_webcam():
    global is_webcam_running, detected_objects
    if not is_webcam_running:
        # Set the flag to indicate that the webcam is running
        is_webcam_running = True
        # Clear the list of detected objects (if any)
        detected_objects = []
        # Start the webcam feed and object detection in a separate thread
        webcam_thread = threading.Thread(target=detect_objects)
        webcam_thread.start()


# Function to stop the webcam feed
def stop_webcam():
    global is_webcam_running
    # Set the flag to indicate that the webcam should stop
    is_webcam_running = False


def save_to_csv():
    # Ask user to choose a save location for the CSV file
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
                                            initialfile="detected_objects.csv")

    # Write the detected object information to the CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Class Name", "Confidence", "X1", "Y1", "X2", "Y2"])
        csv_writer.writerows(detected_objects)


# Create the Tkinter application
app = tk.Tk()
app.title("Real-Time Object Detection with YOLO")

# Set the initial window size
app.geometry("800x600")

# Add a label to display the camera feed
video_label = tk.Label(app)
video_label.pack(fill=tk.BOTH, expand=True)

# Add a frame to group the buttons
button_frame = tk.Frame(app)
button_frame.pack()

# Add a button to start the webcam feed and perform object detection
btn_start = tk.Button(button_frame, text="Start Webcam", command=start_webcam)
btn_start.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

# Add a button to stop the webcam feed
btn_stop = tk.Button(button_frame, text="Stop Webcam", command=stop_webcam)
btn_stop.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

# Create the "Download CSV" button and associate it with the save_to_csv function
btn_download_csv = tk.Button(button_frame, text="Download CSV", command=save_to_csv)
btn_download_csv.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

# Start the Tkinter main loop
app.mainloop()
