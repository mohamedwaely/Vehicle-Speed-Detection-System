import argparse
from video_processor import VideoProcessor
from argument_parser import parse_arguments

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize and run video processor
    processor = VideoProcessor(args)
    processor.process_video()

if __name__ == "__main__":
    main()