import numpy as np
from scipy.ndimage import gaussian_filter


def rand_pattern(
        h: int = None,
        w: int = None,
        seed: int = None,
        ) -> np.ndarray:

    if seed: np.random.seed(seed)

    return np.random.randint(0, 256, (h, w))


def render_autostereogram(
        depth_map: np.ndarray = None,
        pattern_size: int = 127,
        invert: bool = True,
        blur_sigma: float = 1,
        seed: int = None,
        iter_index: int = 0,
        ) -> np.ndarray:

    # parameter init
    invert = -1 if invert is None else 1
    pattern_div = int(np.round(depth_map.shape[1] / pattern_size))

    # normalize depth map
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = np.round(depth_map * 255).astype('uint8')
    norm_map = np.round(depth_map / pattern_div * invert).astype('uint8')

    # shift depth map for correct center alignment
    gap = pattern_size//2   # norm_map.max()
    norm_map[:, gap:] = norm_map[:, :-1*gap]

    # initialize output array while leaving blank space on left for pattern
    stereo_img = np.zeros(np.array(depth_map.shape) + np.array([0, pattern_size]))

    # variable init
    hscale = 1
    height = stereo_img.shape[0]*hscale
    pattern = rand_pattern(height, pattern_size, seed=seed)[iter_index:min(iter_index+height//hscale, height)]

    # fill left hand-sight of image with pattern
    stereo_img[:, :pattern_size] = pattern

    # iterate through pixels to generate auto-stereo image
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):

            # shift pattern based on normalized depth map values
            shift = int(norm_map[y, x])

            # replicate pattern based on shift value
            stereo_img[y, x + pattern_size] = stereo_img[y, x + shift]

    # low pass
    stereo_img = gaussian_filter(stereo_img, sigma=blur_sigma) if blur_sigma > 0 else stereo_img

    # omit auxiliary pattern on the left
    stereo_img = stereo_img[:, pattern_size:]

    return stereo_img


if __name__ == '__main__':

    from pyramid_depth import pyramid_depth
    depth = pyramid_depth(stripe_width=64, pyramid_num=8, gap_scale=4)
    frame = render_autostereogram(depth, pattern_size=112, invert=True, blur_sigma=1)

    import matplotlib.pyplot as plt
    _, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(frame, cmap='gray')
    axs[0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    axs[1].imshow(depth, cmap='gray')
    axs[1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()
