from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from image_processing import process_image_file

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    input_file_path = f"temp_{file.filename}"
    output_file_path = f"fixed_{file.filename}"

    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())

    hot_pixel_report = process_image_file(input_file_path, output_file_path)

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
