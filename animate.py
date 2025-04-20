import numpy as np
from scipy.ndimage import rotate
from typing import List

from render_autostereogram import render_autostereogram


def animated_stereogram(
        depth: np.ndarray = None,
        frame_num: int = 60,
        rotation_opt: bool = True,
        symmetry: int = 1,
        pattern_size: int = 128,
        invert: bool = True,
        blur_sigma: float = 1,
        ) -> List[np.ndarray]:

    angle = (360/symmetry) / frame_num if symmetry != 0 else 0

    if angle == 0:
        rotation_opt = False

    frames = []
    for i in range(frame_num):
        depth = rotate(depth, angle*i, reshape=False, mode='nearest') if rotation_opt else map
        frame = render_autostereogram(depth, pattern_size=pattern_size, invert=invert, blur_sigma=blur_sigma, seed=38855+i)
        frames.append(frame)

    return frames


if __name__ == '__main__':

    from pyramid_depth import pyramid_depth
    depth = pyramid_depth(stripe_width=32, pyramid_num=8, gap_scale=4)

    stack = animated_stereogram(depth, frame_num=60, symmetry=4, rotation_opt=True, pattern_size=128, blur_sigma=1)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    anim = plt.imshow(stack[0], interpolation='none', cmap='gray')
    secs = 6
    fps = 10

    def init():
        anim.set_data(stack[0])
        return anim,

    def update(i):
        anim.set_data(stack[i])
        return anim,

    ani = FuncAnimation(fig, update, frames=secs*fps, init_func=init, blit=True)
    plt.show()
