import os
import shutil

import numpy as np
from PIL import Image

from backend.database import Database
from backend.embedding_generator import EmbeddingGenerator
from backend.face_detector import FaceDetector
from settings import settings


def prepare() -> None:
    if os.path.exists(settings.faiss_index_path):
        return

    db = Database()
    fd = FaceDetector()
    eg = EmbeddingGenerator()

    originals = sorted(os.listdir(settings.meme_storage))
    indexed = []

    for file in originals:
        print(file)

        img = np.asarray(Image.open(os.path.join(settings.meme_storage, file)))

        face_data = fd.detect(img)
        if face_data is None:
            indexed.append(None)
            continue
        else:
            cropped, emb = face_data

        emb, _ = eg.generate_embedding(cropped)

        indexed.append(db.add_vector(emb)[0])

    for original_path, id in zip(originals, indexed):
        if id is None:
            continue
        src = os.path.join(settings.meme_storage, original_path)
        dst = os.path.join(settings.meme_storage, str(id) + '.jpeg')
        shutil.copy(src, dst)

    del db