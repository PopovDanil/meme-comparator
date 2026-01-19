import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from settings import settings


def cast_to_pil(img: np.ndarray) -> Image.Image:
    """
    Prepares image. Coverts to RGB.

    Args:
        img (np.ndarray): image.

    Returns:
        Image.Image: Pillow image.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


model, preprocess = open_clip.create_model_from_pretrained(
    model_name=settings.open_clip_model,
    pretrained=settings.open_clip_weights, # Important
    device=settings.open_clip_device
)
model.eval()


def generate_embedding(img: np.ndarray) ->  np.ndarray:
    """
    Generates embedding.

    Args:
        img (np.ndarray): image.

    Returns:
        np.ndarray: embedding.
    """
    casted = cast_to_pil(img) # prepare image
    transformed = preprocess(casted).unsqueeze(0).to(settings.open_clip_device)

    # Generate embedding
    with torch.no_grad():
        emb = model.encode_image(transformed)

    # Normalize and transform from tensor to ndarray
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy()

    return emb