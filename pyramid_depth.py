import numpy as np


def pyramid_depth(
        stripe_width: int = 1,
        pyramid_num: int = 8,
        gap_scale: float = 0,
        max_val: int = None,
        fill_value: int = None,
        ) -> np.ndarray:

    num = pyramid_num * 2
    res = stripe_width * num
    max_val = pyramid_num-1 if max_val is None else max_val
    arr = np.ones((res, res)) * max_val

    vals = np.arange(num//2)

    for j, i in enumerate(range(num//2, num)):
        val = vals[j]
        arr[i * stripe_width:(i + 1) * stripe_width, :] = val
        arr[:, i * stripe_width:(i + 1) * stripe_width] = val
        arr[(num-i-1) * stripe_width:(num - i) * stripe_width, :] = val
        arr[:, (num-i-1) * stripe_width:(num - i) * stripe_width] = val

    # embed depth image in frame
    gap = res//gap_scale if gap_scale > 0 else 0
    if gap > 0:
        frame = np.zeros((res + gap, res + gap)) * np.nan
        frame[gap//2:-gap//2, gap//2:-gap//2] = arr
        frame[np.isnan(frame)] = np.nanmax(frame) if fill_value is None else fill_value
    else:
        frame = arr

    return frame


if __name__ == '__main__':

    inch_mm = 25.4
    h_mm = 12.375 * inch_mm
    w_mm = 12.375 * inch_mm
    dpi = 300

    h_pixel = int(dpi * h_mm / inch_mm)
    w_pixel = int(dpi * w_mm / inch_mm)

    pyramid_num = 8
    res_y = 2 * h_pixel // 3
    width = res_y // (2*pyramid_num)
    depth = pyramid_depth(stripe_width=width, pyramid_num=8, gap_scale=0)

    import imageio
    imageio.imwrite('pyramid.png', (depth[..., None].repeat(3, -1)/depth.max()*255).astype(np.uint8))

    import matplotlib.pyplot as plt
    plt.imshow(depth, vmax=depth.max(), cmap='gray')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()
