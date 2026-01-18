import base64
import json
from io import BytesIO

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.templating import _TemplateResponse

from backend.embedding_generator import generate_embedding
from backend.face_detector import FaceDetector
from settings import settings

app = FastAPI()
face_detector = FaceDetector()
templates = Jinja2Templates(
    directory=settings.template_path
)


@app.get('/health')
async def health():
    return 'Healthy as fkc'


@app.get('/')
async def root(request: Request) -> _TemplateResponse:
    template = 'index.html'
    return templates.TemplateResponse(
        template,
         {'request': request}
    )


@app.websocket('/ws')
async def get_frames(ws: WebSocket):
    await ws.accept()

    print('Websocket client connected!')

    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)

            if data.get('type') == 'frame':
                image_data = data['image']

                _, encoded = image_data.split(',', 1)

                img_bytes = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_bytes))

                img = np.asarray(img)

                cropped_face, emb = face_detector.detect(img)
                emb = generate_embedding(cropped_face)
                print(emb.shape)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")