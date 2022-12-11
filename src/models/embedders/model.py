from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def get_embedding(self, message: str) -> np.array:
        """
        Args:
            message: str to embed

        Returns:
            embedding of the message in form of np.array
        """
