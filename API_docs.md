# Документация API

Оглавление

- [Документация API](#документация-api)
  - [Endpoints](#endpoints)
    - [Get Layouts](#get-layouts)
    - [Repair Pixels](#repair-pixels)
    - [Download Fixed Image](#download-fixed-image)
    - [Download Report](#download-report)
    - [Process Image with Layout](#process-image-with-layout)
    - [Get Task Status](#get-task-status)
    - [Download Coordinates](#download-coordinates)


Этот API предоставляет Endpoints для сканирования директорий на наличие файлов подложек, исправления пикселей в загруженных изображениях, обработки изображений с использованием заданных подложек и управления статусами задач. Ниже приведены детали каждой Endpoint вместе с примерами использования в curl и Postman.

## Endpoints

---
### Get Layouts

    Endpoint: /layouts/
    Method: GET
    Описание: Возвращает список файлов подложек, доступных в указанной директории.

Пример curl

```sh
curl -X GET "http://127.0.0.1:8000/layouts/"

[
  {
    "name": "layout_2021-08-16.tif",
    "description": "Description of layout_2021-08-16.tif"
  },
  {
    "name": "layout_2021-06-15.tif",
    "description": "Description of layout_2021-06-15.tif"
  },
  {
    "name": "layout_2021-10-10.tif",
    "description": "Description of layout_2021-10-10.tif"
  },
  {
    "name": "layout_2022-03-17.tif",
    "description": "Description of layout_2022-03-17.tif"
  }
]
```

Пример Postman

    Создайте новый GET Request.
    Установите URL на http://127.0.0.1:8000/layouts/.
    Отправьте Request.

---
### Repair Pixels

    Endpoint: /repair_pixels/
    Method: POST
    Описание: Загружает файл изображения, исправляет его пиксели и возвращает пути к исправленному изображению и отчету.

Пример curl

```sh
curl -X POST "http://127.0.0.1:8000/repair_pixels/" -F "file=@path/to/your/crop_0_0_0000.tif"

{
  "fixed_image": "fixed_crop_0_0_0000.tif",
  "report": "report_crop_0_0_0000.tif.txt"
}
```

Пример Postman

    Создайте новый POST Request.
    Установите URL на http://127.0.0.1:8000/repair_pixels/.
    Вкладка "Body", выберите "form-data".
    Добавьте новый ключ "file", выберите тип "File" и выберите ваш файл изображения.
    Отправьте Request.

---
### Download Fixed Image

    Endpoint: /download_fixed_image/{filename}
    Method: GET
    Описание: Скачивает исправленный файл изображения по имени файла.

Пример curl

```sh
curl -X GET "http://127.0.0.1:8000/download_fixed_image/fixed_image.tif" -o fixed_image.tif
```

Пример Postman

    Создайте новый GET Request.
    Установите URL на http://127.0.0.1:8000/download_fixed_image/fixed_image.tif.
    Отправьте Request.
    Файл будет загружен.

---
### Download Report

    Endpoint: /download_report/{filename}
    Method: GET
    Описание: Скачивает файл отчета по имени файла.

Пример curl

```sh
curl -X GET "http://127.0.0.1:8000/download_report/report_image.txt" -o report_image.txt
```

Пример Postman

    Создайте новый GET Request.
    Установите URL на http://127.0.0.1:8000/download_report/report_image.txt.
    Отправьте Request.
    Файл будет загружен.

---
### Process Image with Layout

    Endpoint: /process_image_api/{layout_name}
    Method: POST
    Описание: Загружает файл изображения для обработки с использованием указанной подложки. Обработка выполняется в фоновом режиме.

Пример curl

```sh
curl -X POST "http://127.0.0.1:8000/process_image_api/layout_name" -F "file=@path/to/your/image.tif"

{
  "message": "Task received, processing in background",
  "taskid": "1718564451849337"
}
```

Пример Postman

    Создайте новый POST Request.
    Установите URL на http://127.0.0.1:8000/process_image_api/layout_name.
    Вкладка "Body", выберите "form-data".
    Добавьте новый ключ "file", выберите тип "File" и выберите ваш файл изображения.
    Отправьте Request.

---
### Get Task Status

    Endpoint: /task_status/{taskid}
    Method: GET
    Описание: Возвращает статус фоновой задачи по идентификатору задачи.

Пример curl

```sh
curl -X GET "http://127.0.0.1:8000/task_status/123456789"

{
  "taskid": "1718564451849337",
  "status": "in_progress"
}
```

Пример Postman

    Создайте новый GET Request.
    Установите URL на http://127.0.0.1:8000/task_status/123456789.
    Отправьте Request.

---
### Download Coordinates

    Endpoint: /download_coords/{taskid}
    Method: GET
    Описание: Скачивает файл CSV с координатами по идентификатору задачи и возвращает его содержимое в формате JSON.

Пример curl

```sh
curl -X GET "http://127.0.0.1:8000/download_coords/123456789"

{
  "layout_name": "layout_2021-10-10.tif",
  "crop_name": "crop_0_2_0000.tif",
  "ul": "427102.709_5796443.207",
  "ur": "438664.31_5796165.846",
  "br": "438514.184_5777348.5",
  "bl": "426952.583_5777625.86",
  "crs": "EPSG:32637",
  "start": "2024-06-16T19:00:51",
  "end": "2024-06-16T19:02:59"
}
```

Пример Postman

    Создайте новый GET Request.
    Установите URL на http://127.0.0.1:8000/download_coords/123456789.
    Отправьте Request.

Эта документация охватывает основные Endpoints API с примерами для использования в curl и Postman, что облегчает тестирование и интеграцию.