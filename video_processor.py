import supervision as sv
from ultralytics import YOLO
import numpy as np
from view_transformer import ViewTransformer
from speed_tracker import SpeedTracker
import csv

class VideoProcessor:
    """Main class responsible for processing video and detecting vehicles."""
    
    SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 250
    TARGET = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])

    def __init__(self, args):
        self.args = args
        self.video_info = sv.VideoInfo.from_video_path(args.source_video_path)
        self.model = YOLO("yolov8x.pt")
        self.setup_components()
        
    def setup_components(self):
        """Initialize all necessary components for video processing."""
        self.byte_track = sv.ByteTrack(
            frame_rate=self.video_info.fps,
            track_activation_threshold=self.args.confidence_threshold
        )
        
        thickness = sv.calculate_optimal_line_thickness(self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(self.video_info.resolution_wh)
        
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=self.video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )
        
        self.polygon_zone = sv.PolygonZone(polygon=self.SOURCE)
        self.view_transformer = ViewTransformer(source=self.SOURCE, target=self.TARGET)
        self.speed_tracker = SpeedTracker(self.video_info.fps)
        
    def process_frame(self, frame):
        """Process a single frame of video."""
        # Detect objects
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > self.args.confidence_threshold]
        detections = detections[self.polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=self.args.iou_threshold)
        detections = self.byte_track.update_with_detections(detections=detections)

        # Transform and track points
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points=points).astype(int)

        # Update coordinates and calculate speeds
        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            self.speed_tracker.update_coordinates(tracker_id, y)
            label, speed = self.speed_tracker.calculate_speed(tracker_id)
            labels.append(label)
            if speed > 0:
                self.speed_tracker.record_speed_violation(tracker_id, speed)

        return self.annotate_frame(frame, detections, labels)
        
    def annotate_frame(self, frame, detections, labels):
        """Annotate the frame with detections and labels."""
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame, 
            polygon=self.SOURCE, 
            color=sv.Color.RED
        )
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )
        return annotated_frame

    def process_video(self):
        """Process the entire video file."""
        print("Initializing video processing...")
        
        # Initialize CSV file
        with open("speed_breakers.csv", "w", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Tracker ID", "Speed (km/h)"])
        
        frame_generator = sv.get_video_frames_generator(self.args.source_video_path)
        total_frames = int(self.video_info.total_frames)
        processed_frames = 0
        
        with sv.VideoSink(self.args.target_video_path, self.video_info) as sink:
            for frame in frame_generator:
                processed_frames += 1
                if processed_frames % 50 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"Processing: {progress:.1f}% complete", end='\r')
                
                annotated_frame = self.process_frame(frame)
                sink.write_frame(annotated_frame)
        
        print("\nProcessing complete! Output video saved to:", self.args.target_video_path)