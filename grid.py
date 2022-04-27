from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


EMPTY = 0
WALL = 255


def is_padded(search_map: np.array):
    """return if the image array is padded with ones"""
    return np.all(search_map[0]) and np.all(search_map[-1]) and np.all(search_map[:, 0]) and np.all(search_map[:, -1])


def ensure_padding(search_map: np.array):
    """pad the search map with walls if not padded already"""
    if not is_padded(search_map):
        search_map = np.pad(search_map, (1, 1),
                            mode="constant", constant_values=(WALL))
    return search_map


@dataclass
class Grid:
    """
    class to hold the search map. It can be initialized with either: 
        1- a url to an image of any format. boundaries should be light.
        2- a numpy array with boundaries set to 1 and elsewhere to 0.
        3- height and width of empty grid.
    """
    url: Optional[str] = None
    grid_array: Optional[np.array] = None
    shape: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.url is not None:
            if self.url.endswith(".npy"):
                self.grid_array = np.load(self.url)
            else:
                self._from_image()

        elif self.grid_array is not None:
            self.grid_array = self.grid_array.astype(np.uint8)

        else:
            self.grid_array = np.zeros(
                self.shape, dtype=np.uint8)

        # pad grid with walls if not padded already, and store dimensions
        self.grid_array = ensure_padding(self.grid_array)
        self.shape = self.grid_array.shape

    def _from_image(self):
        self.grid_array = cv2.imread(self.url, 0)
        self.grid_array = self.grid_array.astype(np.uint8)
        _, self.grid_array = cv2.threshold(
            self.grid_array, 127, 255, cv2.THRESH_BINARY)
