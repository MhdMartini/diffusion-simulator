from typing import Tuple
import numpy as np
import cv2
from grid import Grid
from tqdm import tqdm
from time import perf_counter
import torch as T
"""
class to define the class for the seach map
search map object contains a 2d array of cell objects
"""

EMPTY = 0
WALL = 255
AGENT = 2
C_PARTICLE = 255
C_WALL = 200
C_EMPTY = 0
BLUE_BGR = (0, 0, 255)
RED_BGR = (255, 0, 0)


class EnvVis:
    """class to hold visualization methods for all environments"""

    def __init__(self):
        self.colors = None

        # define lambdas
        self.get_grid_array_copy = [
            lambda: self.grid_array.copy(),
            lambda: self.grid_array.clone(),
            lambda: self.grid_array.clone()
        ]
        self.get_grid_gs = [
            lambda: self._get_grid_gs(),
            lambda: self._get_grid_gs().cpu().numpy(),
            lambda: self._get_grid_gs().cpu().numpy(),
        ]
        self.get_pos_np = [
            lambda: self.pos,
            lambda: self.pos.cpu().numpy(),
            lambda: self.pos.cpu().numpy(),
        ]

    def _get_grid_gs(self) -> np.array:
        grid = self.get_grid_array_copy[self.device_idx]()
        pos = self.get_pos[self.device_idx]()
        grid[pos[:, 0], pos[:, 1]] = C_PARTICLE
        grid[grid == WALL] = C_WALL
        return grid

    def get_fps(self):
        fps = int(1 / (perf_counter() - self.time))
        self.time = perf_counter()
        return fps

    def render(self):
        frame = self.get_gbr_frame()
        cv2.imshow("BROWNIAN MOTION SIMULATOR", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return False
        return True

    def draw_fps(self, frame: np.array):
        fps = self.get_fps()
        frame = cv2.putText(frame, f'{fps}', (frame.shape[1] - 120, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 3, cv2.LINE_AA)
        return frame

    def color_particles(self, frame: np.array) -> np.array:
        pos = self.get_pos_np[self.device_idx]()
        frame[pos[:, 0], pos[:, 1]] = self.colors
        return frame

    def get_gbr_frame(self):
        frame = self.get_grid_gs[self.device_idx]()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = self.color_particles(frame)
        frame = self.draw_fps(frame)
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


class RandomWalk(EnvVis):
    def __init__(self, grid: Grid, n_particles: int = None, out_path: str = None, fps: int = None, device_idx: int = 1):
        self.devices_names = ["numpy", "Tensor (CPU)", "Tensor (GPU)"]
        self.devices = ["numpy", "cpu", "cuda"]
        self.libs = [np, T, T]
        self.device_idx = device_idx
        self.device = self.devices[device_idx]
        print("using", self.device)
        super(RandomWalk, self).__init__()

        self.grid = grid

        # define lambdas
        self.get_grid_array = [
            lambda grid_array: grid_array,
            lambda grid_array: T.tensor(grid_array).to(self.device),
            lambda grid_array: T.tensor(grid_array).to(self.device),
        ]
        self.get_empty_coords = [
            lambda: self.get_empty_coords_np(),
            lambda: self.get_empty_coords_torch(),
            lambda: self.get_empty_coords_torch(),
        ]
        self.get_pos = [
            lambda: np.zeros((self.n_particles, 2), dtype=int),
            lambda: T.zeros((self.n_particles, 2), dtype=int).to(self.device),
            lambda: T.zeros((self.n_particles, 2), dtype=int).to(self.device),
        ]
        self.get_actions = [
            lambda: np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]),
            lambda: T.tensor(
                [[0, 1], [0, -1], [1, 0], [-1, 0]]).to(self.device),
            lambda: T.tensor(
                [[0, 1], [0, -1], [1, 0], [-1, 0]]).to(self.device),
        ]

        self.grid_array = self.get_grid_array[self.device_idx](grid.grid_array)
        self.empty_rows, self.empty_cols = self.get_empty_coords[self.device_idx](
        )

        self.n_particles = n_particles
        self.pos = self.get_pos[self.device_idx]()

        self.actions = self.get_actions[self.device_idx]()
        self.n_actions = len(self.actions)

        self.out = None
        if out_path is not None:
            self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fps, (grid.shape[1], grid.shape[0]), 3)
        self.time = 0

    def get_empty_coords_torch(self) -> Tuple[T.Tensor, T.Tensor]:
        """return coordinates of empty positions"""
        empty_rows, empty_cols = T.where(self.grid_array == EMPTY)
        empty_rows, empty_cols = empty_rows.to(
            self.device), empty_cols.to(self.device)
        return empty_rows, empty_cols

    def get_empty_coords_np(self) -> Tuple[np.array, np.array]:
        """return coordinates of empty positions"""
        empty_rows, empty_cols = np.where(self.grid_array == EMPTY)
        return empty_rows, empty_cols

    def step(self):
        a = np.random.choice(self.n_actions, size=self.n_particles)
        pos_new = self.pos + self.actions[a]
        pos_vals = self.grid_array[pos_new[:, 0], pos_new[:, 1]]
        wall_clsns = self.libs[self.device_idx].where(pos_vals == WALL)[0]
        pos_new[wall_clsns] = self.pos[wall_clsns]
        self.pos = pos_new

    def reset(self, same_point=False) -> np.array:
        """get random initial positions for ants"""
        # get random valid indices
        device_array = [np.array, T.tensor, T.tensor]
        get_pos_init = [
            lambda indices: np.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T,
            lambda indices: T.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device),
            lambda indices: T.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device),
        ]

        if same_point:
            self.pos[:] = device_array[self.device_idx](
                (self.grid.shape[0] // 2, self.grid.shape[1] // 2))
            return self.pos

        indices = np.random.choice(
            len(self.empty_rows), size=self.n_particles, replace=True)

        # assign agents positions
        self.pos = get_pos_init[self.device_idx](indices)

        EnvVis.reset(self)
        return self.pos

    def __del__(self):
        if self.out is not None:
            self.out.release()
            cv2.destroyAllWindows()


def save_video(env, args):
    print(f"saving {args.out_path}...")
    n_frames = int(args.vid_len * args.fps)
    for _ in tqdm(range(n_frames)):
        env.save()
        env.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_shape', nargs="+",
                        type=int, default=(1080, 1920), help="height and width of the grid")
    parser.add_argument('--grid_path', type=str, default=None,
                        help="path to grid npy or image file - walls should be bright")
    parser.add_argument('--n_particles', type=int,
                        default=1000_000, help="number of particles")
    parser.add_argument('--same_point', type=int, default=0,
                        help="0: particles start from random positions\n1: particles start from the same point")
    parser.add_argument('--out_path', type=str, default=None,
                        help="if a path is provided, video will be saved to this path")
    parser.add_argument('--vid_len', type=int, default=None,
                        help="length of output video in seconds")
    parser.add_argument('--fps', type=int, default=60,
                        help="frames per second of output video")
    parser.add_argument('--device', type=int, default=1,
                        help="0: numpy, 1: torch cpu, 2: torch gpu")
    args = parser.parse_args()

    grid = Grid(args.grid_path, None, args.grid_shape)
    env = RandomWalk(grid, n_particles=args.n_particles,
                     out_path=args.out_path, fps=args.fps, device_idx=args.device)
    env.reset(same_point=args.same_point)

    if args.out_path is not None:
        # save video
        save_video(env, args)
    else:
        # render
        while env.render():
            env.step()
