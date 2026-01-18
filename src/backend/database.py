import os

import faiss
import numpy as np

from settings import settings


class Database:
    def __init__(self):
        self.k_neighbors = settings.faiss_k_neighbors
        self.index = self.__load_index()


    def add_vector(self, vector: list[np.ndarray] | np.ndarray) -> np.ndarray:
        x = np.atleast_2d(vector)

        ids_before = self.index.ntotal
        self.index.add(x)
        ids_after = self.index.ntotal

        return np.arange(ids_before, ids_after, dtype=np.int64)


    def query(self, vector: np.ndarray) -> list[np.ndarray]:
        x = np.atleast_2d(vector)
        return self.index.search(x, k=self.k_neighbors) # returns (distances, indices)


    def __load_index(self) -> faiss.Index:
        index = None

        if os.path.exists(settings.faiss_index_path):
            index = faiss.read_index(settings.faiss_index_path)
        else:
            index = faiss.IndexFlatL2(settings.face_embedding_size)

        return index


    def __save(self) -> None:
        faiss.write_index(self.index, settings.faiss_index_path)


    def __del__(self):
        self.__save()