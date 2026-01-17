
import cv2

from backend.face_detector import FaceDetector

if __name__ == '__main__':
    f = FaceDetector()
    frame = cv2.imread('face2.jpeg')
    f.detect(frame)
    # uvicorn.run(
    #     settings.uvicorn_app_name,
    #     settings.uvicorn_server_port
    # )