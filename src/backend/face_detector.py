import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from torch import tensor

from settings import settings


class FaceDetector:
    def __init__(self):
        self.device = settings.face_detector_device
        self.img_size = settings.face_detector_img_size

        self.detector = FaceAnalysis(settings.face_detector_model, providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=self.device, det_size=settings.face_detector_img_size)


    def detect(self, img: tensor) -> tuple[tensor, tensor]:
        faces = self.detector.get(img)

        if not faces:
            return None

        landmarks = faces[0].kps
        embedding = faces[0].embedding
        cropped = face_align.norm_crop(img, landmarks, image_size=settings.face_detector_out_size)
        # print(faces)

        if settings.debug:
            cv2.imwrite('./src/debug/face.jpg', cropped)

        return cropped, embedding