from typing import Tuple
import numpy as np
import cv2
from time import perf_counter

C_WALL = (100, 100, 100)
C_EMPTY = (255, 255, 255)
C_TEXT = (24, 24, 24)
RED_BGR = (255, 0, 0)
BLUE_BGR = (0, 0, 255)


class EnvVis:
    """class to hold visualization methods for all environments"""

    def __init__(self):
        self.colors = None

        # define lambdas
        self.get_array_copy = [
            lambda grid_array: grid_array.copy(),
            lambda grid_array: grid_array.clone().cpu().numpy(),
            lambda grid_array: grid_array.clone().cpu().numpy()
        ]
        self.get_pos_np = [
            lambda: self.pos,
            lambda: self.pos.cpu().numpy(),
            lambda: self.pos.cpu().numpy(),
        ]
        self.frame = self.get_frame_template()

    def get_frame_template(self):
        frame = np.zeros((*self.grid.shape, 3), dtype=np.uint8)
        frame[:, :, :] = C_EMPTY
        frame[self.wall_rows, self.wall_cols] = C_WALL
        return frame

    def render(self):
        frame = self.get_gbr_frame()
        cv2.imshow("BROWNIAN MOTION SIMULATOR", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return False
        return True

    def get_gbr_frame(self):
        frame = self.color_particles(self.frame.copy())
        frame = self.draw_fps(frame)
        frame = self.draw_device(frame)
        return frame

    def color_particles(self, frame: np.array) -> np.array:
        pos = self.get_pos_np[self.device_idx]()
        frame[pos[:, 0], pos[:, 1]] = self.colors
        return frame

    def get_fps(self):
        fps = int(1 / (perf_counter() - self.time))
        self.time = perf_counter()
        return fps

    def draw_fps(self, frame: np.array):
        fps = self.get_fps()
        frame = cv2.putText(
            frame, f'FPS: {fps}', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, C_TEXT, 3, cv2.LINE_AA)
        return frame

    def draw_device(self, frame: np.array) -> np.array:
        """draw device on frame"""
        frame = cv2.putText(frame, self.devices_names[self.device_idx], (
            40, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, C_TEXT, 3, cv2.LINE_AA)
        frame = cv2.putText(frame, self.devices_specs[self.device_idx], (
            40, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, C_TEXT, 3, cv2.LINE_AA)
        return frame

    def get_bg(self):
        """return half blue half red"""
        c = np.zeros((self.n_particles, 3))
        pos = self.get_pos_np[self.device_idx]()
        c[np.where(pos[:, 1] < self.grid.shape[1] // 2)] = RED_BGR
        c[np.where(pos[:, 1] >= self.grid.shape[1] // 2)] = BLUE_BGR
        return c

    def save(self):
        frame = self.get_gbr_frame()
        self.out.write(frame)

    def get_zoom_coords(self, h: int, w: int, h_perc: float, w_perc: float) -> Tuple[int, int, int, int]:
        """get coordinates of zoomed frame"""
        h_sub = int(h * h_perc / 2)
        w_sub = int(w * w_perc / 2)
        center_h, center_w = (h // 2, w // 2)
        return center_h - h_sub, center_h + h_sub, center_w - w_sub, center_w + w_sub

    def save_zoomed(self, h_perc: float, w_perc: float):
        frame = self.get_gbr_frame()
        h, w, _ = frame.shape
        r_min, r_max, c_min, c_max = self.get_zoom_coords(
            h, w, h_perc, w_perc)
        frame_sub = frame[r_min: r_max, c_min: c_max]
        frame_zoomed = cv2.resize(
            frame_sub, (w, h), interpolation=cv2.INTER_NEAREST)
        self.out.write(frame_zoomed)

    def reset(self):
        self.colors = self.get_bg()
