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

    def get_grid_gs(self) -> np.array:
        grid = self.grid_array.clone()
        grid[self.pos[:, 0], self.pos[:, 1]] = C_PARTICLE
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
        pos = self.pos.cpu().numpy()
        frame[pos[:, 0], pos[:, 1]] = self.colors
        return frame

    def get_gbr_frame(self):
        frame = self.get_grid_gs().cpu().numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = self.color_particles(frame)
        frame = self.draw_fps(frame)
        return frame

    def get_bg(self):
        """return half blue half red"""
        c = np.zeros((self.n_particles, 3))
        pos = self.pos.cpu().numpy()
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
    def __init__(self, grid: Grid, n_particles: int = None, out_path: str = None, fps: int = None, device: str = "cuda"):
        super(RandomWalk, self).__init__()
        self.device = device
        print("using", self.device)

        self.grid = grid
        self.grid_array = T.tensor(grid.grid_array).to(self.device)
        self.empty_rows, self.empty_cols = self.get_empty_coords()

        self.n_particles = n_particles
        self.pos = T.zeros((self.n_particles, 2), dtype=int).to(self.device)

        self.actions = T.tensor(
            [[0, 1], [-1, 0], [0, -1], [1, 0]]).to(self.device)
        self.n_actions = len(self.actions)

        self.out = None
        if out_path is not None:
            self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fps, (grid.shape[1], grid.shape[0]), 3)
        self.time = 0

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

    def reset(self, same_point=False) -> np.array:
        """get random initial positions for ants"""
        # get random valid indices
        if same_point:
            self.pos[:] = T.tensor(
                (self.grid_array.shape[0] // 2, self.grid_array.shape[1] // 2))
            return self.pos

        indices = np.random.choice(
            len(self.empty_rows), size=self.n_particles, replace=True)

        # assign agents positions
        self.pos = T.stack(
            (self.empty_rows[indices], self.empty_cols[indices])).T.to(self.device)

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
    parser.add_argument('--gpu', type=int, default=1,
                        help="1: use cuda, 0: use cpu")
    args = parser.parse_args()

    grid = Grid(args.grid_path, None, args.grid_shape)
    env = RandomWalk(grid, n_particles=args.n_particles,
                     out_path=args.out_path, fps=args.fps, device="cuda" if args.gpu else "cpu")
    env.reset(same_point=args.same_point)

    if args.out_path is not None:
        # save video
        save_video(env, args)
    else:
        # render
        while env.render():
            env.step()
