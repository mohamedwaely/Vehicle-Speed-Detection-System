from collections import defaultdict, deque
import csv
from typing import Dict, Deque
import numpy as np

class SpeedTracker:
    """Class responsible for tracking and calculating vehicle speeds."""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.coordinates: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=int(fps)))
        
    def update_coordinates(self, tracker_id: int, y_coordinate: int):
        """Update coordinates for a tracked object."""
        self.coordinates[tracker_id].append(y_coordinate)
        
    def calculate_speed(self, tracker_id: int) -> tuple[str, int]:
        """Calculate speed for a tracked object."""
        if len(self.coordinates[tracker_id]) < self.fps / 2:
            return f"#{tracker_id}", 0
            
        coordinate_start = self.coordinates[tracker_id][-1]
        coordinate_end = self.coordinates[tracker_id][0]
        distance = abs(coordinate_start - coordinate_end)
        time = len(self.coordinates[tracker_id]) / self.fps
        speed = int(distance / time * 3.6)
        
        return f"#{tracker_id} {speed} km/h", speed
        
    def record_speed_violation(self, tracker_id: int, speed: int, threshold: int = 120):
        """Record speed violations to CSV file."""
        if speed > threshold:
            with open("speed_breakers.csv", "a", newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([tracker_id, speed])