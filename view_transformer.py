import numpy as np
import cv2

class ViewTransformer:
    """Class responsible for transforming view perspectives."""
    
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the ViewTransformer with source and target coordinates.
        
        Args:
            source (np.ndarray): Source coordinates for transformation
            target (np.ndarray): Target coordinates for transformation
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using the perspective transformation matrix.
        
        Args:
            points (np.ndarray): Points to transform
            
        Returns:
            np.ndarray: Transformed points
        """
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)