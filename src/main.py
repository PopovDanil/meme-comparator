import tensorflow as tf
import uvicorn

from settings import settings

tf.config.set_visible_devices([], "GPU")

if __name__ == '__main__':
    uvicorn.run(
        settings.uvicorn_app_name,
        port=settings.uvicorn_server_port
    )