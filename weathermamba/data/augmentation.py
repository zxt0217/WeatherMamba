"""
Point Cloud Augmentation

Data augmentation transformations for point clouds.
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PointCloudAugmentation:
    """
    Point cloud data augmentation pipeline.

    Includes rotation, scaling, jittering, flipping, etc.
    """

    @staticmethod
    def _to_dict(config_value):
        """Return dict-like config safely."""
        if isinstance(config_value, dict):
            return config_value
        return {}

    @staticmethod
    def _parse_rotation_config(config_value):
        if isinstance(config_value, dict):
            return config_value
        if isinstance(config_value, (list, tuple)) and len(config_value) == 2:
            return {'enabled': True, 'angle_range': list(config_value)}
        if isinstance(config_value, bool):
            return {'enabled': config_value}
        return {}

    @staticmethod
    def _parse_scaling_config(config_value):
        if isinstance(config_value, dict):
            return config_value
        if isinstance(config_value, (list, tuple)) and len(config_value) == 2:
            return {'enabled': True, 'scale_range': list(config_value)}
        if isinstance(config_value, bool):
            return {'enabled': config_value}
        return {}

    @staticmethod
    def _parse_jitter_config(config_value):
        if isinstance(config_value, dict):
            return config_value
        if isinstance(config_value, (int, float)):
            return {'enabled': True, 'std': float(config_value)}
        if isinstance(config_value, bool):
            return {'enabled': config_value}
        return {}

    @staticmethod
    def _parse_flip_config(config_value):
        if isinstance(config_value, dict):
            return config_value
        if isinstance(config_value, (int, float)):
            return {'enabled': True, 'probability': float(config_value)}
        if isinstance(config_value, bool):
            return {'enabled': config_value}
        return {}

    def __init__(self, config):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config

        # Extract augmentation configuration
        aug_config = self._to_dict(config.get('augmentation', {}))
        self.enabled = aug_config.get('enabled', True)

        # Random rotation configuration
        rotation_config = self._parse_rotation_config(aug_config.get('random_rotation', {}))
        self.rotation_enabled = rotation_config.get('enabled', True)
        self.rotation_angle_range = rotation_config.get('angle_range', [-45, 45])
        self.rotation_axis = rotation_config.get('axis', 'z')
        self.rotation_probability = rotation_config.get('probability', 0.5)

        # Random scaling configuration
        scaling_config = self._parse_scaling_config(aug_config.get('random_scaling', {}))
        self.scaling_enabled = scaling_config.get('enabled', True)
        self.scale_range = scaling_config.get('scale_range', [0.8, 1.2])
        self.scaling_probability = scaling_config.get('probability', 0.5)

        # Random jittering configuration
        jitter_config = self._parse_jitter_config(aug_config.get('random_jitter', {}))
        self.jitter_enabled = jitter_config.get('enabled', True)
        self.jitter_std = jitter_config.get('std', 0.03)  # 0.01-0.05 range
        self.jitter_clip = jitter_config.get('clip', 0.05)
        self.jitter_probability = jitter_config.get('probability', 0.5)

        # Random flipping configuration
        flip_config = self._parse_flip_config(aug_config.get('random_flip', {}))
        self.flip_enabled = flip_config.get('enabled', True)
        self.flip_x = flip_config.get('flip_x', True)
        self.flip_y = flip_config.get('flip_y', True)
        self.flip_probability = flip_config.get('probability', 0.5)

        # Random translation configuration (optional)
        translation_config = self._to_dict(aug_config.get('random_translation', {}))
        self.translation_enabled = translation_config.get('enabled', False)
        self.translation_range = translation_config.get('translation_range', [-2.0, 2.0])
        self.translation_probability = translation_config.get('probability', 0.3)

        # Random dropout configuration (optional)
        dropout_config = self._to_dict(aug_config.get('random_dropout', {}))
        self.dropout_enabled = dropout_config.get('enabled', False)
        self.dropout_ratio = dropout_config.get('dropout_ratio', 0.1)
        self.dropout_probability = dropout_config.get('probability', 0.3)

        logger.info("Initialized PointCloudAugmentation with enabled augmentations: "
                   f"rotation={self.rotation_enabled}, scaling={self.scaling_enabled}, "
                   f"jitter={self.jitter_enabled}, flip={self.flip_enabled}")

    def random_rotation(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        axis: str = 'z'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random rotation around specified axis.

        Args:
            points: Point cloud array (N, 3 or 4)
            axis: Rotation axis ('x', 'y', or 'z')

        Returns:
            Rotated point cloud and unchanged labels
        """
        # Sample rotation angle
        angle = np.random.uniform(self.rotation_angle_range[0], self.rotation_angle_range[1])
        angle_rad = np.deg2rad(angle)

        # Create rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        if axis == 'z':
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
        elif axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        else:
            raise ValueError(f"Invalid rotation axis: {axis}")

        # Apply rotation to xyz coordinates
        points_xyz = points[:, :3].copy()
        points_xyz = points_xyz @ rotation_matrix.T

        # Update points
        points[:, :3] = points_xyz

        return points, labels

    def random_scaling(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random scaling to point cloud.

        Args:
            points: Point cloud array (N, 3 or 4)

        Returns:
            Scaled point cloud and unchanged labels
        """
        # Sample scale factor
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply scaling to xyz coordinates
        points[:, :3] *= scale

        return points, labels

    def random_jitter(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random jittering (Gaussian noise) to point cloud.

        Args:
            points: Point cloud array (N, 3 or 4)

        Returns:
            Jittered point cloud and unchanged labels
        """
        # Generate Gaussian noise
        noise = np.random.normal(0, self.jitter_std, size=points[:, :3].shape)

        # Use a representable bound slightly inside jitter_clip to avoid
        # float rounding edge cases (e.g. 0.0500000007 > 0.05 in tests).
        clip_bound = np.nextafter(
            np.array(self.jitter_clip, dtype=points.dtype),
            np.array(0.0, dtype=points.dtype)
        ).item()

        noise = np.clip(noise, -clip_bound, clip_bound).astype(points.dtype, copy=False)

        # Apply jitter to xyz coordinates
        points[:, :3] += noise

        return points, labels

    def random_flip(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random flipping along x and/or y axis.

        Args:
            points: Point cloud array (N, 3 or 4)

        Returns:
            Flipped point cloud and unchanged labels
        """
        # Flip along x-axis
        if self.flip_x and np.random.rand() < 0.5:
            points[:, 0] = -points[:, 0]

        # Flip along y-axis
        if self.flip_y and np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]

        return points, labels

    def random_translation(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random translation to point cloud.

        Args:
            points: Point cloud array (N, 3 or 4)

        Returns:
            Translated point cloud and unchanged labels
        """
        # Sample translation vector
        translation = np.random.uniform(
            self.translation_range[0],
            self.translation_range[1],
            size=3
        )

        # Apply translation to xyz coordinates
        points[:, :3] += translation

        return points, labels

    def random_dropout(self, points: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Randomly drop points from point cloud.

        Args:
            points: Point cloud array (N, 3 or 4)
            labels: Label array (N,) (optional)

        Returns:
            Point cloud with dropped points and corresponding labels
        """
        # Calculate number of points to keep
        num_points = points.shape[0]
        num_keep = int(num_points * (1 - self.dropout_ratio))

        # Randomly select points to keep
        indices = np.random.choice(num_points, num_keep, replace=False)
        indices = np.sort(indices)  # Keep order

        # Apply dropout
        points = points[indices]
        if labels is not None:
            labels = labels[indices]

        return points, labels

    def __call__(self, points, labels=None):
        """
        Apply augmentation to point cloud.

        Args:
            points: (N, 3 or 4) point cloud (numpy array or torch tensor)
            labels: (N,) semantic labels (optional, numpy array or torch tensor)

        Returns:
            augmented_points: Augmented point cloud
            augmented_labels: Augmented labels (if provided)
        """
        if not self.enabled:
            return points, labels

        # Convert to numpy if torch tensor
        is_torch = isinstance(points, torch.Tensor)
        if is_torch:
            points = points.detach().cpu().numpy()
            if labels is not None:
                labels = labels.detach().cpu().numpy()

        # Make a copy to avoid modifying original
        points = points.copy()
        if labels is not None:
            labels = labels.copy()

        # Apply random rotation
        if self.rotation_enabled and np.random.rand() < self.rotation_probability:
            points, labels = self.random_rotation(points, labels, axis=self.rotation_axis)

        # Apply random scaling
        if self.scaling_enabled and np.random.rand() < self.scaling_probability:
            points, labels = self.random_scaling(points, labels)

        # Apply random jittering
        if self.jitter_enabled and np.random.rand() < self.jitter_probability:
            points, labels = self.random_jitter(points, labels)

        # Apply random flipping
        if self.flip_enabled and np.random.rand() < self.flip_probability:
            points, labels = self.random_flip(points, labels)

        # Apply random translation (optional)
        if self.translation_enabled and np.random.rand() < self.translation_probability:
            points, labels = self.random_translation(points, labels)

        # Apply random dropout (optional)
        if self.dropout_enabled and np.random.rand() < self.dropout_probability:
            points, labels = self.random_dropout(points, labels)

        # Convert back to torch if needed
        if is_torch:
            points = torch.from_numpy(points)
            if labels is not None:
                labels = torch.from_numpy(labels)

        return points, labels
