import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

def wrap_around_distort(img, scale=2):
    return np.round((scale*img-img.min())/(img.max()-img.min())*255).astype('uint8')

def convert2rgb_uint8(gry_img, color):
    
    gry_img = (gry_img.astype(np.float64)-gry_img.min())/(gry_img.max()-gry_img.min())
    rgb_img = np.repeat(gry_img[..., None], 3, axis=-1) * np.array(color)[None, None, :]
    rgb_img = np.round((rgb_img-rgb_img.min())/(rgb_img.max()-rgb_img.min())*255).astype('uint8')

    return rgb_img

dpi = 300
inch_mm = 25.4
dpi = 300
h_mm = 12.375 * inch_mm
w_mm = 12.375 * inch_mm
h_pixel = int(dpi * h_mm / inch_mm)
pyramid_num = 8
res_y = 2*h_pixel//3
pyr_w = res_y // (2 * pyramid_num)
res_y = pyr_w * 2 * pyramid_num

from pyramid_depth import pyramid_depth
depth = pyramid_depth(stripe_width=pyr_w, pyramid_num=pyramid_num, gap_scale=0)

from render_autostereogram import render_autostereogram
color_from_uint8 = lambda color_tuple: [c / 255 for c in color_tuple]

red_uint8 = (210, 31, 60)

cwd = Path(__file__).parent
single_opt = False
save_img_files = False
if save_img_files:
    frame_path = cwd / 'frame_image'
    frame_path.mkdir(exist_ok=True)
frame_num = 1 if single_opt else 600//5*4

frame = render_autostereogram(depth, pattern_size=128*4, invert=True, blur_sigma=4, seed=38855, iter_index=0)
frame = frame[32:-32, 32:-32] # crop frame to yield valid frame rate
imageio.imwrite('frame_image.png', (255*(frame-frame.min())/(frame.max()-frame.min())).astype(np.uint8))

# stack copies of image vertically
frame = np.vstack((frame, frame, frame))

# wrap-around distortion effect
frame = wrap_around_distort(frame, scale=2)
frame = gaussian_filter(frame, sigma=2)

# single channel to rgb extension
red_hi = color_from_uint8(red_uint8)
red_hi = color_from_uint8([255, 42, 6])
frame = convert2rgb_uint8(frame, red_hi)

# Create a writer object
output_video_path = frame_path.parent / 'video.mp4'
writer = imageio.get_writer(output_video_path, fps=20)

# Iterate through the images and add them to the video writer
for i in range(frame_num):
    
    step = i*(frame.shape[0]//frame_num//3)
    curr_frame = frame.copy()[step:step+frame.shape[0]//3]

    if save_img_files or single_opt: imageio.imwrite(frame_path / ('frame_image' + '_' + str(i) + '.png'), curr_frame)
    writer.append_data(curr_frame)

# Close the writer
writer.close()