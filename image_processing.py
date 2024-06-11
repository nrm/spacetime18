import numpy as np
from tifffile import imread, imsave
from scipy.ndimage import median_filter

def find_outlier_pixels(data, window_size=2, dark_threshold=0.15, bright_threshold=5.0):
    blurred = median_filter(data, size=window_size)
    difference = data - blurred

    zero_pixels_mask = (data == 0)
    dark_pixels_mask = (data < blurred * dark_threshold) & (data != 0)
    bright_pixels_mask = (data > blurred * bright_threshold)
    hot_pixels_mask = zero_pixels_mask | dark_pixels_mask | bright_pixels_mask

    hot_pixels = np.nonzero(hot_pixels_mask)

    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        if y > 0 and y < data.shape[0] - 1 and x > 0 and x < data.shape[1] - 1:
            fixed_image[y, x] = blurred[y, x]

    hot_pixel_report = []
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        if y > 0 and y < data.shape[0] - 1 and x > 0 and x < data.shape[1] - 1:
            original_value = data[y, x]
            fixed_value = fixed_image[y, x]
            hot_pixel_report.append(f"{y}; {x}; {original_value}; {fixed_value}")

    return hot_pixel_report, fixed_image

def process_image_file(input_file_path, output_file_path, window_size=3, dark_threshold=0.15, bright_threshold=5.0):
    image_data = imread(input_file_path)

    hot_pixel_report = []
    for channel in range(image_data.shape[2]):
        channel_report, fixed_channel = find_outlier_pixels(
            image_data[:, :, channel], window_size, dark_threshold, bright_threshold
        )
        image_data[:, :, channel] = fixed_channel
        hot_pixel_report.extend(channel_report)

    imsave(output_file_path, image_data)

    return hot_pixel_report
