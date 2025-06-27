## Overview
This project implements a real-time vehicle speed detection system using computer vision and deep learning. The system can detect vehicles in video footage, track their movement, and calculate their speeds. It also identifies and records speed violations in a CSV file.

## Demo
![Screenshot 2025-06-27 093153](https://github.com/user-attachments/assets/f9ec392e-0cb9-4ac1-8ea3-18eb2fb71409)

![Uploading Screenshot 2025-06-27 093222.png…]()

![Uploading Screenshot 2025-06-27 093254.png…]()

![Screenshot 2025-06-27 093017](https://github.com/user-attachments/assets/03cffefd-4332-4344-9c57-a451d8a867a5)


## Features
Real-time vehicle detection and tracking
Speed calculation for each detected vehicle
Speed violation monitoring and recording
Visual output with annotated vehicles and speed information
Perspective transformation for accurate measurements
Progress tracking during video processing


## Output

The system generates two types of output:

Annotated video file showing:
Detected vehicles with bounding boxes
Vehicle IDs and calculated speeds
Movement traces
Detection zone boundaries

CSV file ("speed_breakers.csv") containing:
Tracker ID of vehicles
Recorded speeds of vehicles exceeding the speed limit
Timestamp of violations


## Components

### VideoProcessor
Main class coordinating the detection and processing pipeline
Handles frame-by-frame processing
Manages video input/output operations

### ViewTransformer
Performs perspective transformation
Converts video coordinates to real-world measurements
Ensures accurate speed calculations

### SpeedTracker
Tracks vehicle movements
Calculates speeds based on position changes
Records speed violations


RESOURSES:
https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation
