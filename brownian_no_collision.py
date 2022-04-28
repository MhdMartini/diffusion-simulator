from typing import Tuple
import numpy as np
import cv2
from grid import Grid
import torch as T
from tqdm import tqdm
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


class Brownian:
    def __init__(self, grid: Grid, n_particles: int = None, out_path: str = None, fps: int = None):
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print("using", self.device)

        self.grid = grid
        self.grid_array = T.tensor(grid.grid_array).to(self.device)
        self.empty_rows, self.empty_cols = self.get_empty_coords()

        self.n_particles = n_particles if n_particles is not None else np.prod(
            grid.shape) // 10
        self.pos = T.zeros((self.n_particles, 2), dtype=int).to(self.device)

        self.actions = T.tensor(
            [[0, 1], [-1, 0], [0, -1], [1, 0]]).to(self.device)
        self.n_actions = len(self.actions)

        self.out = None
        if out_path is not None:
            self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fps, (grid.shape[1], grid.shape[0]), 3)

    def get_empty_coords(self) -> Tuple[T.Tensor, T.Tensor]:
        """return coordinates of empty positions"""
        empty_rows, empty_cols = T.where(self.grid_array == EMPTY)
        empty_rows, empty_cols = empty_rows.to(
            self.device), empty_cols.to(self.device)
        return empty_rows, empty_cols

    def step(self):
        a = np.random.choice(self.n_actions, size=self.n_particles)
        pos_new = self.pos + self.actions[a]
        pos_vals = self.grid_array[pos_new[:, 0], pos_new[:, 1]]
        wall_clsns = T.where(pos_vals == WALL)[0]
        pos_new[wall_clsns] = self.pos[wall_clsns]
        self.pos = pos_new

    def get_grid_gs(self) -> np.array:
        grid = self.grid_array.clone()
        grid[self.pos[:, 0], self.pos[:, 1]] = C_PARTICLE
        grid[grid == WALL] = C_WALL
        return grid

    def render(self, time=25):
        grid = self.get_grid_gs().cpu().numpy()
        cv2.imshow("BROWNIAN MOTION SIMULATOR", grid)
        if cv2.waitKey(time) & 0xFF == ord('q'):
            return False
        return True

    def save(self):
        grid = self.get_grid_gs().cpu().numpy()
        grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        self.out.write(grid)

    def reset(self, same_point=False) -> np.array:
        """get random initial positions for ants"""
        # get random valid indices
        indices = np.random.choice(
            len(self.empty_rows), size=self.n_particles, replace=False)

        if same_point:
            indices[:] = indices[0]

        # assign agents positions
        self.pos = T.stack(
            (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device)
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
                        default=100_000, help="number of particles")
    parser.add_argument('--same_point', type=int, default=0,
                        help="0: particles start from random positions\n1: particles start from the same point")
    parser.add_argument('--out_path', type=str, default=None,
                        help="if a path is provided, video will be saved to this path")
    parser.add_argument('--vid_len', type=int, default=None,
                        help="length of output video in seconds")
    parser.add_argument('--fps', type=int, default=60,
                        help="frames per second of output video")
    args = parser.parse_args()

    grid = Grid(args.grid_path, None, args.grid_shape)
    brownian = Brownian(grid, n_particles=args.n_particles,
                        out_path=args.out_path, fps=args.fps)
    brownian.reset(same_point=args.same_point)

    if args.out_path is not None:
        # save video
        save_video(brownian, args)
    else:
        # render
        while brownian.render():
            brownian.step()
