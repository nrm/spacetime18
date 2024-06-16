from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import csv
from pixel_repair_report import process_image_file
from process_image import main_process_func
from datetime import datetime, timezone

app = FastAPI()

# Директория, в которой будут сканироваться подложки
LAYOUTS_DIR = os.getenv("LAYOUTS_DIR", "/layouts")

# Хранение статусов задач
task_status = {}

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

@app.get("/layouts/")
async def get_layouts():
    return layouts

@app.post("/repair_pixels/")
async def repair_pixels(file: UploadFile = File(...)):
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

def run_main_process(layout_name: str, input_file_path: str, taskid: str):
    layout_name = os.path.join(LAYOUTS_DIR, layout_name)
    main_process_func(layout_name, input_file_path, taskid)
    task_status[taskid] = "completed"

@app.post("/process_image_api/{layout_name}")
async def main_process(layout_name: str, file: UploadFile, background_tasks: BackgroundTasks):
    input_file_path = file.filename
    
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    timestamp = datetime.now(timezone.utc).timestamp()
    taskid = f"{int(timestamp * 1_000_000)}"
    
    # Изначально устанавливаем статус задачи как "in_progress"
    task_status[taskid] = "in_progress"
    background_tasks.add_task(run_main_process, layout_name, input_file_path, taskid)

    return {
        "message": "Task received, processing in background",
        "taskid": taskid
    }

@app.get("/task_status/{taskid}")
async def get_task_status(taskid: str):
    status = task_status.get(taskid)
    if status:
        return {"taskid": taskid, "status": status}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/download_coords/{taskid}")
async def download_coords(taskid: str):
    # filename = 'ccoords_' + taskid + '.csv'
    filename = taskid
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            row = next(reader, None)
            if row:
                # Возвращаем данные как JSON
                return JSONResponse(content=row)
            else:
                raise HTTPException(status_code=404, detail="No data found in the file")
    else:
        raise HTTPException(status_code=404, detail="File not found")
