from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pixel_repair_report import process_image_file
from process_image import main_process_func

app = FastAPI()

# Директория, в которой будут сканироваться подложки
LAYOUTS_DIR = os.getenv("LAYOUTS_DIR", "/layouts")

# Хранение подложек в памяти
layouts = []
active_layout = None

class Layout(BaseModel):
    name: str
    description: str

# Функция для сканирования директории и получения списка подложек
def scan_layouts(directory: str):
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    layouts = [{"name": f, "description": f"Description of {f}"} for f in tif_files]
    return layouts

# Сканирование директории при запуске приложения
layouts = scan_layouts(LAYOUTS_DIR)

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

@app.get("/layouts/")
async def get_layouts():
    return layouts

@app.put("/layouts/active/{layout_name}")
async def set_active_layout(layout_name: str):
    global active_layout
    for layout in layouts:
        if layout["name"] == layout_name:
            active_layout = layout
            return {"message": f"Layout '{layout_name}' is now active."}
    raise HTTPException(status_code=404, detail="Layout not found")

@app.get("/layouts/active/")
async def get_active_layout():
    if active_layout:
        return active_layout
    raise HTTPException(status_code=404, detail="No active layout set")

@app.post("/process_image_api/{layout_name}")
async def main_process(layout_name: str, file: UploadFile = File(...)):
    input_file_path = f"temp_{file.filename}"
    
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    from datetime import datetime, timezone
    unix_time = datetime.now(timezone.utc).timestamp()*1000
    
    taskid =str(unix_time) + '.csv'
    
    # main_process_func(main_process_func, input_file_path, taskid)
    main_process_func(layout_name, input_file_path, taskid)

    return {
        "taskid": taskid
    }

@app.get("/download_coords/{filename}")
async def download_coords(filename: str):
    return FileResponse(filename)
