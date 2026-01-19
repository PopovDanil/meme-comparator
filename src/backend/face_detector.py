import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from settings import settings


class FaceDetector:
    def __init__(self):
        """
        Initializes instance of face detector.
        """
        self.device = settings.face_detector_device # Defaults to CPU to avoid deploy issues
        self.img_size = settings.face_detector_img_size # shape of input image (will be adjusted automatically)

        # Load and prepare model
        self.detector = FaceAnalysis(settings.face_detector_model, providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=self.device, det_size=settings.face_detector_img_size)


    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects face on the image.

        Args:
            img (np.ndarray): image.

        Returns:
            tuple[np.ndarray, np.ndarray]: cropped face and embedding.
        """
        faces = self.detector.get(img)

        if not faces:
            print('No faces were found!')
            return None

        landmarks = faces[0].kps # corners
        embedding = faces[0].embedding

        # Extract face from photo
        cropped = face_align.norm_crop(img, landmarks, image_size=settings.face_detector_out_size)

        # Optionally save face
        if settings.debug:
            cv2.imwrite('./src/debug/face.jpg', cropped)

        return cropped, embedding