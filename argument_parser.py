import argparse

def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        default=r"C:\Users\al-aqsa\speed_estimation\vehicles.mp4",
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=r"C:\Users\al-aqsa\speed_estimation\vehicles_result.mp4",
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="IOU threshold for the model",
        type=float,
    )
    return parser.parse_args()