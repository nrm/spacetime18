import subprocess
import argparse
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import csv
import tempfile
from datetime import datetime, timezone
# import uuid
from pixel_repair_report import process_image_file

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
    with tempfile.NamedTemporaryFile(delete=False) as input_file:
        input_file_path = input_file.name
        input_file.write(await file.read())
    
    output_file_path = f"fixed_{file.filename}"
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

def generate_task_id():
    timestamp = datetime.now(timezone.utc).timestamp()
    # unique_component = uuid.uuid4().hex  # Генерируем уникальный UUID
    task_id = f"{int(timestamp * 1_000_000)}"  # _{unique_component}"  # Умножаем на 1_000_000 для получения микросекундной точности и добавляем уникальный компонент
    return task_id

def run_main_process(layout_name: str, input_file_path: str, taskid: str):
    try:
        subprocess.run(["python", "main.py", "--crop_name", input_file_path, "--layout_name", layout_name], check=True)
        task_status[taskid] = "completed"
    except subprocess.CalledProcessError as e:
        task_status[taskid] = "failed"
        print(f"Error processing {taskid}: {e}")
    finally:
        os.remove(input_file_path)

@app.post("/process_image_api/{layout_name}")
async def main_process(layout_name: str, file: UploadFile, background_tasks: BackgroundTasks):
    with tempfile.NamedTemporaryFile(delete=False) as input_file:
        input_file_path = input_file.name
        input_file.write(await file.read())
    
    layout_name = os.path.join(LAYOUTS_DIR, layout_name)
    taskid = generate_task_id()
    
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
    # filename = 'coords_' + taskid + '.csv'
    filename = "coords.csv"
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            row = next(reader, None)
            if row:
                return JSONResponse(content=row)
            else:
                raise HTTPException(status_code=404, detail="No data found in the file")
    else:
        raise HTTPException(status_code=404, detail="File not found")
