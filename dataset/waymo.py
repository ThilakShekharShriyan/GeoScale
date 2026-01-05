import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import geohash2
from waymo_open_dataset import dataset_pb2 as open_dataset

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class WaymoGCSAdapter:
    """
    Adapter to stream and parse Waymo Open Dataset .tfrecord files from Google Cloud Storage.
    Extracts authentic camera images, poses, intrinsics, and geohashes.
    """
    def __init__(self, bucket_name="waymo_open_dataset_v_1_4_3", split="training"):
        self.bucket_name = bucket_name
        self.split = split
        self.gcs_prefix = f"gs://{self.bucket_name}/uncompressed/tf_example/{self.split}/"
        
    def _get_tfrecord_paths(self):
        """Returns a list of GCS paths to the tfrecord segments."""
        # For a full implementation, you would use google-cloud-storage to list the blobs.
        # Since we are pipelining, we assume a known segment name structure or list it.
        # Example dummy path for demonstration:
        return [f"{self.gcs_prefix}segment-1208303279778032257_1360_000_1380_000_with_camera_labels.tfrecord"]

    def _approximate_gps_from_pose(self, transform):
        """
        Waymo frames provide a 4x4 vehicle pose in a global coordinate frame (typically 
        metric offsets from a predefined origin). For GeoScale's Geohash partitioning, 
        we approximate a pseudo-GPS coordinate based on translation offsets.
        Real implementations would use the map_dir anchor if exact GPS is needed.
        """
        # Pseudo conversion for infrastructure demonstration scale
        tx, ty, tz = transform[0][3], transform[1][3], transform[2][3]
        # Very rough scaling to map metric offsets to fake Lat/Lon near San Francisco
        lat = 37.7749 + (ty / 111111.0)
        lon = -122.4194 + (tx / (111111.0 * math.cos(math.radians(37.7749))))
        return lat, lon

    def iter_frames(self, tfrecord_path, limit=None):
        """
        Yields structured data dictionaries for each frame in the tfrecord.
        """
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        
        count = 0
        for data in dataset:
            if limit and count >= limit:
                break
                
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            # Extract Vehicle Pose
            vehicle_transform = np.array(frame.pose.transform).reshape(4, 4)
            lat, lon = self._approximate_gps_from_pose(vehicle_transform)
            
            # Find the FRONT camera image and calibration
            front_image = None
            for camera_image in frame.images:
                if camera_image.name == open_dataset.CameraName.FRONT:
                    front_image = camera_image
                    break
                    
            if not front_image:
                continue
                
            # Find the corresponding camera calibration
            calibration = None
            for calib in frame.context.camera_calibrations:
                if calib.name == open_dataset.CameraName.FRONT:
                    calibration = calib
                    break
                    
            if not calibration:
                continue
                
            # Extrinsic: Camera to Vehicle transform (4x4)
            extrinsic = np.array(calibration.extrinsic.transform).reshape(4, 4)
            
            # Global Camera Pose = Vehicle Pose * Extrinsic
            camera_global_pose = vehicle_transform @ extrinsic
            
            # Intrinsic: 1D Array to 3x3 pinhole matrix
            # Waymo provides 1D array: [f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3]
            intrinsics_1d = np.array(calibration.intrinsic)
            intrinsic_matrix = np.array([
                [intrinsics_1d[0], 0, intrinsics_1d[2]],
                [0, intrinsics_1d[1], intrinsics_1d[3]],
                [0, 0, 1]
            ])
            
            # Geohash
            geohash_full = geohash2.encode(lat, lon, precision=12)
            
            yield {
                "image_data": front_image.image, # JPEG bytes
                "pose": camera_global_pose,
                "intrinsics": intrinsic_matrix,
                "lat": float(lat),
                "lon": float(lon),
                "geohash": geohash_full,
                "timestamp_micros": frame.timestamp_micros
            }
            count += 1
