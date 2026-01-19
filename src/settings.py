from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_detector_device: int = -1 # default device (cpu) for face detector
    face_detector_model: str = 'buffalo_l' # default model (weakest)
    face_detector_img_size: tuple = (640, 640)
    face_detector_out_size: int = 224 # should be divisible by 112

    face_embedding_size: int = 7

    uvicorn_app_name: str = 'backend.api:app'
    uvicorn_server_port: int = 5050
    template_path: str = './src/frontend'

    faiss_index_path: str = './meme_storage/db.faiss'
    faiss_k_neighbors: int = 5

    open_clip_device: str = 'cpu'
    open_clip_model: str = 'ViT-B-32'
    open_clip_weights: str = 'laion2b_s34b_b79k'

    meme_storage: str = './meme_storage/' # stores images in jpeg format

    debug: bool = True # debug mode


settings = Settings()