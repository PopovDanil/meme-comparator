import tensorflow as tf
import uvicorn

from backend.prepare_db import prepare
from settings import settings

tf.config.set_visible_devices([], "GPU")

if __name__ == '__main__':
    prepare()
    uvicorn.run(
        settings.uvicorn_app_name,
        port=settings.uvicorn_server_port
    )