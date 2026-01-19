import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.templating import _TemplateResponse

from backend.database import Database
from backend.embedding_generator import generate_embedding
from backend.face_detector import FaceDetector

# from backend.utils import top_k_sampling
from settings import settings

app = FastAPI()
db = Database()
face_detector = FaceDetector()
templates = Jinja2Templates(
    directory=settings.template_path
)


@app.get('/health')
async def health():
    """
    Health check.

    Returns:
        str: health.
    """
    return 'Healthy as fkc'


@app.get('/')
async def root(request: Request) -> _TemplateResponse:
    """
    Root page.

    Args:
        request (Request): request.

    Returns:
        _TemplateResponse: html-template.
    """
    template = 'index.html'
    return templates.TemplateResponse(
        template,
         {'request': request}
    )


@app.websocket('/ws')
async def get_frames(ws: WebSocket):
    """
    Websocket for live communication.

    Args:
        ws (WebSocket): Websocket.
    """
    await ws.accept()

    print('Websocket client connected!')

    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)

            if data.get('type') == 'frame':
                image_data = data['image']

                # Get encoded image, decode it, open with Pillow, and cast to ndarray
                _, encoded = image_data.split(',', 1)

                img_bytes = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_bytes))

                img = np.asarray(img)

                # Crop face, get embedding (optimally _ can be replaced)
                cropped_face, _ = face_detector.detect(img)
                emb = generate_embedding(cropped_face) # get CLIP embedding

                # Look for most similar vectors (optionally choose randomly)
                _, ids = db.query(emb)
                # selected = top_k_sampling(ids[0])
                selected = ids[0][0]

                # Select meme, encode, and send in json
                most_similar = Path(f'{settings.meme_storage}{selected}.jpeg')

                if most_similar.exists():
                    img_bytes = most_similar.read_bytes()
                    encoded = base64.b64encode(img_bytes).decode('utf-8')

                    await ws.send_text(json.dumps({
                        'type': 'image',
                        'image': f'data:image/jpeg;base64,{encoded}'
                    }))
                else:
                    print(f'No image in path {most_similar}!')


    except WebSocketDisconnect:
        print("WebSocket client disconnected")