# meme-comparator

## How to use:
### Install requirements.txt (python 3.11)
### Follow the structure
```text
.venv/
meme_storage/        # must be created manually
src/
├── backend/
├── debug/          # must be created manually
└── frontend/
main.py
prepare_memes.ipynb
settings.py
```

### Prepare mems
Download ```.jpeg``` images into ```meme_storage/```. Then run each cell in ```prepare_mems.ipynb```.

### Run server
Run main.py from root directory. It will be accessible via http://127.0.0.1:5050