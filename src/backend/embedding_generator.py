import cv2
import numpy as np
import open_clip
import torch
from deepface import DeepFace
from fer.fer import FER
from PIL import Image

from settings import settings


class EmbeddingGenerator:
    def __init__(self):
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            model_name=settings.open_clip_model,
            pretrained=settings.open_clip_weights, # Important
            device=settings.open_clip_device
        )
        self.model.eval()
        self.fer_detector = FER(mtcnn=True)

    def __cast_to_pil(self, img: np.ndarray) -> tuple[np.ndarray, Image.Image]:
        """
        Prepares image. Coverts to RGB.

        Args:
            img (np.ndarray): image.

        Returns:
            Image.Image: Pillow image.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb, Image.fromarray(rgb)


    def __generate_CLIP_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Generates CLIP embedding.

        Args:
            img (np.ndarray): image.

        Returns:
            np.ndarray: embedding.
        """
        transformed = self.preprocess(img).unsqueeze(0).to(settings.open_clip_device)

        # Generate embedding
        with torch.no_grad():
            emb = self.model.encode_image(transformed)

        # Normalize and transform from tensor to ndarray
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy()

        return emb

    def __generate_DeepFace_embedding(self, img: np.ndarray) -> np.ndarray:
        emotions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if emotions:
            emotions = emotions[0]['emotion']
            emb = np.array([emotions[e] for e in sorted(emotions.keys())], dtype=np.float32)
        else:
            emb = np.zeros(settings.face_embedding_size, dtype=np.float32)
        return emb


    def __generate_FER_embedding(self, img: np.ndarray) -> np.ndarray:
        emotions = self.fer_detector.detect_emotions(img)
        if emotions:
            emotions = emotions[0]['emotions']
            return np.array(list(emotions.values()), dtype=np.float32)
        return np.zeros(settings.face_embedding_size, dtype=np.float32)


    def generate_embedding(self, img: np.ndarray) -> tuple[np.ndarray]:
        arr, casted = self.__cast_to_pil(img)
        deepface_emb = self.__generate_DeepFace_embedding(arr)
        clip_emb = self.__generate_CLIP_embedding(casted)
        fer_emb = self.__generate_FER_embedding(arr)

        emotion_emb = (deepface_emb + fer_emb) / 2.
        emotion_emb = emotion_emb / np.linalg.norm(emotion_emb)

        return emotion_emb, clip_emb