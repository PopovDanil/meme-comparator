from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_detector_device: int = -1
    face_detector_model: str = 'buffalo_l'
    face_detector_img_size: tuple = (640, 640)
    face_detector_out_size: int = 224

    face_embedding_size: int = 512

    uvicorn_app_name: str = 'backend.api:app'
    uvicorn_server_port: int = 5050
    template_path: str = './src/frontend'

    faiss_index_path: str = './meme_storage/db.faiss'
    faiss_k_neighbors: int = 5

    open_clip_device: str = 'cpu'
    open_clip_model: str = 'ViT-B-32'
    open_clip_weights: str = 'laion2b_s34b_b79k'

    debug: bool = True


settings = Settings()