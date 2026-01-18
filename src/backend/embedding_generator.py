import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from settings import settings


def cast_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


model, preprocess = open_clip.create_model_from_pretrained(
    model_name=settings.open_clip_model,
    pretrained=settings.open_clip_weights,
    device=settings.open_clip_device
)
model.eval()


def generate_embedding(img: np.ndarray) ->  np.ndarray:
    casted = cast_to_pil(img)
    transformed = preprocess(casted).unsqueeze(0).to(settings.open_clip_device)

    with torch.no_grad():
        emb = model.encode_image(transformed)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy()

    return emb