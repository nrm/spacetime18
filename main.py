import numpy as np
from tifffile import imread, imsave
from scipy.ndimage import median_filter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os

app = FastAPI()

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

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    input_file_path = f"temp_{file.filename}"
    output_file_path = f"fixed_{file.filename}"

    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())

    image_data = imread(input_file_path)

    hot_pixel_report = []
    for channel in range(image_data.shape[2]):
        channel_report, fixed_channel = find_outlier_pixels(image_data[:, :, channel], window_size=3, dark_threshold=0.15, bright_threshold=5.0)
        image_data[:, :, channel] = fixed_channel
        hot_pixel_report.extend(channel_report)

    imsave(output_file_path, image_data)

    os.remove(input_file_path)

    report_file_path = f"report_{file.filename}.txt"
    with open(report_file_path, "w") as report_file:
        report_file.write("Отчет о битых пикселях:\n")
        for line in hot_pixel_report:
            report_file.write(line + "\n")

    return {
        "fixed_image": output_file_path,
        "report": report_file_path
    }

@app.get("/download_fixed_image/{filename}")
async def download_fixed_image(filename: str):
    return FileResponse(filename)

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    return FileResponse(filename)
