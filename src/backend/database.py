import os

import faiss
import numpy as np

from settings import settings


class Database:
    def __init__(self, path: str | None = None):
        """
        Initializes "db" instance. This is convenient API for FAISS search engine.

        Args:
            path (str | None, optional): path to 'db.faiss' If None then will be created automatically. Defaults to None.
        """
        self.k_neighbors = settings.faiss_k_neighbors # how many neighbors will be returned after query
        self.path = path if path is not None else settings.faiss_index_path
        self.index = self.__load_index()


    def add_vector(self, vector: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Adds vector(s) to index.

        Args:
            vector (list[np.ndarray] | np.ndarray): vector or list of vectors.

        Returns:
            np.ndarray: 1d vector with added indices. This indices will be used to save images and easily find them.
        """
        x = np.atleast_2d(vector)

        ids_before = self.index.ntotal
        self.index.add(x)
        ids_after = self.index.ntotal

        return np.arange(ids_before, ids_after, dtype=np.int64)


    def query(self, vector: np.ndarray) -> tuple[np.ndarray]:
        """
        Looks for the K most similar vectors in the index.

        Args:
            vector (np.ndarray): vector (embedding).

        Returns:
            tuple[np.ndarray]: (similarity scores (distances), indices).
        """
        x = np.atleast_2d(vector)
        return self.index.search(x, k=self.k_neighbors) # returns (distances, indices)


    def __load_index(self) -> faiss.Index:
        """
        Loads index, if db.faiss exists, or creates a new one.

        Returns:
            faiss.Index: index.
        """
        index = None

        if os.path.exists(self.path):
            index = faiss.read_index(self.path)
        else:
            print('No FAISS index was found. Initializing a new one!')
            index = faiss.IndexFlatL2(settings.face_embedding_size)

        return index


    def __save(self) -> None:
        """
        Saves index.
        """
        faiss.write_index(self.index, self.path)


    def __del__(self):
        """
        Saves index.
        """
        self.__save()