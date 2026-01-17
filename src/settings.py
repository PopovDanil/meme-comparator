from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_detector_device: int = -1
    face_detector_model: str = 'buffalo_l'
    face_detector_img_size: tuple = (640, 640)
    face_detector_out_size: int = 224

    uvicorn_app_name: str = 'backend.api:app'
    uvicorn_server_port: int = 5050

    debug: bool = True


settings = Settings()