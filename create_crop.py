import rasterio
from rasterio.windows import Window
from scipy.ndimage import zoom
import random
import os

def split_tiff(image_path, output_dir, tiles_x, tiles_y, compression='lzw'):
    tiles_x += 2
    tiles_y += 2
    # Открываем изображение с помощью rasterio
    with rasterio.open(image_path) as src:
        img_width = src.width
        img_height = src.height
        transform = src.transform
        num_bands = src.count

        # Вычисляем размер плиток
        tile_width = img_width // tiles_x
        tile_height = img_height // tiles_y

        # Убедимся, что выходной каталог существует
        os.makedirs(output_dir, exist_ok=True)

        # Функция для преобразования пиксельных координат в географические
        def pixel_to_geo(pixel_x, pixel_y, transform):
            geo_x, geo_y = rasterio.transform.xy(transform, pixel_y, pixel_x, offset='center')
            return geo_x, geo_y

        # Разделяем изображение на прямоугольники
        for x in range(1, tiles_x-1):
            for y in range(1, tiles_y-1):
                temp_x = x - 1
                temp_y = y - 1
                left = x * tile_width
                upper = y * tile_height
                right = left + tile_width
                lower = upper + tile_height

                # Учитываем последний ряд и столбец, чтобы они могли быть меньше, чем tile_width и tile_height
                if x == tiles_x - 1:
                    right = img_width
                if y == tiles_y - 1:
                    lower = img_height

                # Обрезаем изображение
                window = Window(left, upper, right - left, lower - upper)
                tile = src.read(window=window)

                # Изменяем размер изображения до требуемого размера плитки
                downsized_tile = zoom(tile, (1, 1/(random.uniform(5,10)), 1/(random.uniform(5,10))))

                # Выводим размеры нового тайла
                print(f"Original tile shape: {tile.shape}")
                print(f"Downsampled tile shape: {downsized_tile.shape}")

                # Вычисляем геокоординаты углов плитки
                top_left = pixel_to_geo(left, upper, transform)
                top_right = pixel_to_geo(right, upper, transform)
                bottom_left = pixel_to_geo(left, lower, transform)
                bottom_right = pixel_to_geo(right, lower, transform)

                # Выводим геокоординаты плитки
                file_coord.write('crop_{}_{}_0000\n'.format(temp_x,temp_y))
                print(f"Tile ({temp_x}, {temp_y}):")
                print(f"  Top-left: {top_left}")
                print(f"  Top-right: {top_right}")
                print(f"  Bottom-left: {bottom_left}")
                print(f"  Bottom-right: {bottom_right}")
                file_coord.write(f"{top_left[0]} {top_left[1]}\n")
                file_coord.write(f"{bottom_left[0]} {bottom_left[1]}\n")
                file_coord.write(f"{bottom_right[0]} {bottom_right[1]}\n")
                file_coord.write(f"{top_right[0]} {top_right[1]}\n")
                file_coord.flush()

                tile_transform = rasterio.transform.from_bounds(top_left[0], bottom_left[1], top_right[0], top_right[1],
                                                                downsized_tile.shape[2], downsized_tile.shape[1])
                #profile = src.profile
                #profile.update(
                #    {'height': downsized_tile.shape[1], 'width': downsized_tile.shape[2], 'transform': tile_transform,
                #     'compress': compression, 'count': num_bands})

                # Создаем профиль для новой плитки
                profile = src.profile
                profile.update({
                    'height': downsized_tile.shape[1],
                    'width': downsized_tile.shape[2],
                    'transform':tile_transform,
#                    'transform': src.transform * rasterio.Affine.translation(left,upper) * rasterio.Affine.scale((tile_width / tile.shape[2]), (tile_height / tile.shape[1])),
                    'compress': compression,
                    'count': num_bands  # Указываем количество каналов цвета
                })

                # Сохраняем плитку с указанным сжатием
                tile_path = os.path.join(output_dir, f"tile_{temp_x}_{temp_y}.tif")
                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(downsized_tile)
                print(f"Saved tile: {tile_path}")

# Пример использования
image_path = "layouts/layout_2021-08-16.tif"
output_dir = "2_40"

tiles_x = 5  # Количество плиток по горизонтали
tiles_y = 4  # Количество плиток по вертикали
file_coord = open('coordinates_newcrop.dat', 'w')
    
split_tiff(image_path, output_dir, tiles_x, tiles_y, compression='lzw')