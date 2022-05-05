from typing import Tuple
import numpy as np
import cv2
from envs.env_vis import EnvVis
from utils.grid import Grid
from tqdm import tqdm
from time import perf_counter
import torch as T

EMPTY = 0
WALL = 255


class RandomWalk(EnvVis):
    def __init__(self, grid: Grid, n_particles: int = None, out_path: str = None, fps: int = None, device_idx: int = 1):
        self.devices_names = ["Numpy", "Torch Tensor CPU", "Torch Tensor GPU"]
        self.devices_specs = [
            "Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz",
            "Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz",
            "GeForce GTX 1660 Ti"
        ]
        self.devices = ["numpy", "cpu", "cuda"]
        self.libs = [np, T, T]
        self.device_idx = device_idx
        self.device = self.devices[device_idx]
        print("using", self.device)
        self.grid = grid

        # define lambdas
        self.get_grid_array = [
            lambda grid_array: grid_array,
            lambda grid_array: T.tensor(grid_array).to(self.device),
            lambda grid_array: T.tensor(grid_array).to(self.device),
        ]
        self.get_val_coords = [
            lambda val: self.get_val_coords_np(val),
            lambda val: self.get_val_coords_torch(val),
            lambda val: self.get_val_coords_torch(val),
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
        self.empty_rows, self.empty_cols = self.get_val_coords[self.device_idx](
            EMPTY)
        self.wall_rows, self.wall_cols = np.where(
            self.get_grid_array[0](grid.grid_array) == WALL)

        self.n_particles = n_particles
        self.pos = self.get_pos[self.device_idx]()

        self.actions = self.get_actions[self.device_idx]()
        self.n_actions = len(self.actions)

        self.out = None
        if out_path is not None:
            self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
                *'mp4v'), fps, (grid.shape[1], grid.shape[0]), 3)
        self.time = 0
        super(RandomWalk, self).__init__()

    def get_val_coords_torch(self, val: int) -> Tuple[T.Tensor, T.Tensor]:
        """return coordinates of empty positions"""
        empty_rows, empty_cols = T.where(self.grid_array == val)
        empty_rows, empty_cols = empty_rows.to(
            self.device), empty_cols.to(self.device)
        return empty_rows, empty_cols

    def get_val_coords_np(self, val: int) -> Tuple[np.array, np.array]:
        """return coordinates of empty positions"""
        empty_rows, empty_cols = np.where(self.grid_array == val)
        return empty_rows, empty_cols

    def step(self):
        a = np.random.choice(self.n_actions, size=self.n_particles)
        pos_new = self.pos + self.actions[a]
        pos_vals = self.grid_array[pos_new[:, 0], pos_new[:, 1]]
        wall_clsns = self.libs[self.device_idx].where(pos_vals == WALL)[0]
        pos_new[wall_clsns] = self.pos[wall_clsns]
        self.pos = pos_new

    def reset(self) -> np.array:
        """get random initial positions for ants"""
        # get random valid indices
        get_pos_init = [
            lambda indices: np.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T,
            lambda indices: T.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device),
            lambda indices: T.stack(
                (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device),
        ]

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
                        default=1_000_000, help="number of particles")
    parser.add_argument('--out_path', type=str, default=None,
                        help="if a path is provided, video will be saved to this path")
    parser.add_argument('--vid_len', type=int, default=None,
                        help="length of output video in seconds")
    parser.add_argument('--fps', type=int, default=60,
                        help="frames per second of output video")
    parser.add_argument('--device', type=int, default=0,
                        help="0: numpy, 1: torch cpu, 2: torch gpu")
    args = parser.parse_args()

    grid = Grid(args.grid_path, None, args.grid_shape)
    env = RandomWalk(grid, n_particles=args.n_particles,
                     out_path=args.out_path, fps=args.fps, device_idx=args.device)
    env.reset()

    if args.out_path is not None:
        # save video
        save_video(env, args)
    else:
        # render
        while env.render():
            env.step()
