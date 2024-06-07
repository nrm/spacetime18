import numpy as np
from tifffile import imread, imsave
from scipy.ndimage import median_filter
import sys

def find_outlier_pixels(data, window_size=2, dark_threshold=0.15, bright_threshold=5.0):
    # Применяем медианный фильтр для сглаживания изображения
    blurred = median_filter(data, size=window_size)

    # Находим разницу между исходным и сглаженным изображениями
    difference = data - blurred

    # Создаем маску для битых пикселей
    zero_pixels_mask = (data == 0)
    dark_pixels_mask = (data < blurred * dark_threshold) & (data != 0)
    bright_pixels_mask = (data > blurred * bright_threshold)
    hot_pixels_mask = zero_pixels_mask | dark_pixels_mask | bright_pixels_mask

    # Получаем координаты битых пикселей
    hot_pixels = np.nonzero(hot_pixels_mask)

    # Исправляем битые пиксели, заменяя их значениями из сглаженного изображения
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        # Проверяем границы изображения
        if y > 0 and y < data.shape[0] - 1 and x > 0 and x < data.shape[1] - 1:
            fixed_image[y, x] = blurred[y, x]

    # Формируем отчет о битых пикселях
    hot_pixel_report = []
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        if y > 0 and y < data.shape[0] - 1 and x > 0 and x < data.shape[1] - 1:
            original_value = data[y, x]
            fixed_value = fixed_image[y, x]
            hot_pixel_report.append(f"{y}; {x}; {channel + 1}; {original_value}; {fixed_value}")

    return hot_pixel_report, fixed_image

# Загрузка 16-битного 4-канального tiff-файла
tiff_file = sys.argv[1]
image_data = imread(tiff_file)

# Применение функции find_outlier_pixels для каждого канала и сбор отчета
hot_pixel_report = []
for channel in range(image_data.shape[2]):
    channel_report, fixed_channel = find_outlier_pixels(image_data[:, :, channel], window_size=3, dark_threshold=0.15, bright_threshold=5.0)
    image_data[:, :, channel] = fixed_channel
    hot_pixel_report.extend(channel_report)

# Вывод отчета о битых пикселях
print("Отчет о битых пикселях:")
for line in hot_pixel_report:
    print(line)

# Сохранение исправленного изображения в tiff-файл
output_file = 'fixed_image.tif'
imsave(output_file, image_data)
