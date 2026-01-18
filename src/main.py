import uvicorn

from settings import settings

if __name__ == '__main__':
    uvicorn.run(
        settings.uvicorn_app_name,
        port=settings.uvicorn_server_port
    )