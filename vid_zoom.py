import numpy as np
from random_walk import RandomWalk
from tqdm import tqdm
from grid import Grid


def save_video(env, args):
    print(f"saving {args.out_path}...")
    n_frames = int(args.vid_len * args.fps)
    frame_start = n_frames // 7
    frame_end = n_frames - n_frames // 4
    for i in tqdm(range(n_frames)):
        if i < frame_start:
            h_perc, w_perc = args.max_zoom, args.max_zoom
        elif i >= frame_end:
            h_perc, w_perc = 1, 1
        else:
            h_perc = w_perc = np.interp(
                i, [frame_start, frame_end], [args.max_zoom, 1])
        env.save_zoomed(h_perc, w_perc)
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
    parser.add_argument('--out_path', type=str, default=None,
                        help="if a path is provided, video will be saved to this path")
    parser.add_argument('--vid_len', type=int, default=None,
                        help="length of output video in seconds")
    parser.add_argument('--fps', type=int, default=60,
                        help="frames per second of output video")
    parser.add_argument('--max_zoom', type=float, default=0.1,
                        help="max zoom -> min ratio of original image")
    parser.add_argument('--device', type=int, default=0,
                        help="0: numpy, 1: torch cpu, 2: torch gpu")
    args = parser.parse_args()

    grid = Grid(args.grid_path, None, args.grid_shape)
    RandomWalk = RandomWalk(grid, n_particles=args.n_particles,
                            out_path=args.out_path, fps=args.fps, device_idx=args.device)
    RandomWalk.reset()

    if args.out_path is not None:
        # save video
        save_video(RandomWalk, args)
    else:
        # render
        while RandomWalk.render():
            RandomWalk.step()
