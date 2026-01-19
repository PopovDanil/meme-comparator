import cv2
import numpy as np
from insightface.app import FaceAnalysis

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
        cropped = self.expanded_crop(img=img, face=faces[0])
        # cropped = face_align.norm_crop(img, landmarks, image_size=settings.face_detector_out_size)

        # Optionally save face
        if settings.debug:
            cv2.imwrite('./src/debug/face.jpg', cropped)

        return cropped, embedding


    def expanded_crop(self, img: np.ndarray, face: dict, scale=1.6)-> np.ndarray:
        """
        Expands the cropped image to include more pixels from original photo to
        avoid under-sensitivity of search.

        Args:
            img (np.ndarray): original image
            face (dict): cropped face
            scale (float, optional): scaling. Defaults to 1.6.

        Returns:
            np.ndarray: cropped face.
        """
        h, w, _ = img.shape
        x1, y1, x2, y2 = map(int, face.bbox)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        size = int(max(x2 - x1, y2 - y1) * scale)

        nx1 = max(0, cx - size // 2)
        ny1 = max(0, cy - size // 2)
        nx2 = min(w, cx + size // 2)
        ny2 = min(h, cy + size // 2)

        return img[ny1:ny2, nx1:nx2]