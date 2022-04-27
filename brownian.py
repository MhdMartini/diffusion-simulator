import numpy as np
import cv2
from grid import Grid
"""
class to define the class for the seach map
search map object contains a 2d array of cell objects
"""

EMPTY = 0
WALL = 255
AGENT = 2
C_PARTICLE = tuple(np.random.choice(255, size=(3,)))
C_WALL = (0, 0, 0)
C_EMPTY = (249, 246, 241)


class Brownian:
    def __init__(self, grid: Grid, n_particles: int = None):
        self.grid = grid
        self.empty_rows, self.empty_cols = np.where(
            self.grid.grid_array == EMPTY)
        self.n_particles = n_particles if n_particles is not None else grid.shape[
            0] * grid.shape[1] // 10
        self.pos = np.zeros((self.n_particles, 2), dtype=int)

        self.actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        self.n_actions = len(self.actions)

    def step(self):
        a = np.random.choice(self.n_actions, size=self.n_particles)
        pos_new = self.pos + self.actions[a]
        pos_vals = self.grid.grid_array[pos_new[:, 0], pos_new[:, 1]]
        wall_clsns = np.where(pos_vals == WALL)[0]
        pos_new[wall_clsns] = self.pos[wall_clsns]
        self.pos = pos_new

    def get_grid_gs(self) -> np.array:
        grid = self.grid.grid_array.copy()
        grid[self.pos[:, 0], self.pos[:, 1]] = 255
        grid[grid == WALL] = 200
        return grid

    def get_grid_3d(self) -> np.array:
        grid = self.grid.grid_array.copy()
        grid[self.pos[:, 0], self.pos[:, 1]] = AGENT
        grid3d = cv2.merge((grid, grid, grid))
        grid3d[grid == AGENT, :] = C_PARTICLE
        grid3d[grid == WALL, :] = C_WALL
        grid3d[grid == EMPTY, :] = C_EMPTY
        return grid3d

    def render(self, time=25):
        grid = self.get_grid_gs()
        # grid = self.get_grid_3d()
        cv2.imshow("BROWNIAN MOTION SIMULATOR", grid)
        if cv2.waitKey(time) & 0xFF == ord('q'):
            return False
        return True

    def reset(self, same_point=False) -> np.array:
        """get random initial positions for ants"""
        # get random valid indices
        indices = np.random.choice(
            len(self.empty_rows), size=self.n_particles, replace=False)

        if same_point:
            indices[:] = indices[0]

        # assign agents positions
        self.pos = np.array(
            (self.empty_rows[indices], self.empty_cols[indices])).T
        return self.pos


if __name__ == '__main__':
    grid = Grid(shape=(1080, 1920))
    brownian = Brownian(grid, n_particles=500000)
    brownian.reset(same_point=False)
    while(brownian.render()):
        brownian.step()
