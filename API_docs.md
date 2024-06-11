# Документация API

## Оглавление

- [Документация API](#документация-api)
  - [Оглавление](#оглавление)
  - [1. Обработать изображение](#1-обработать-изображение)
  - [2. Скачать исправленное изображение](#2-скачать-исправленное-изображение)
  - [3. Скачать отчет](#3-скачать-отчет)
  - [5. Назначить активную подложку](#5-назначить-активную-подложку)
  - [6. Получить активную подложку](#6-получить-активную-подложку)
  - [Пример использования всех эндпоинтов](#пример-использования-всех-эндпоинтов)


## 1. Обработать изображение

- Endpoint: /process_image/
- Method: POST
- Description: Загружает изображение и возвращает исправленное изображение и отчет о битых пикселях.

Request:

    file: Файл изображения (формат UploadFile).

Response:

    fixed_image: Путь к исправленному изображению.
    report: Путь к отчету о битых пикселях.

Пример вызова с помощью curl:

```sh
curl -X POST "http://localhost:8000/process_image/" -F "file=@/path/to/your/image.tif"
```

## 2. Скачать исправленное изображение

- Endpoint: /download_fixed_image/{filename}
- Method: GET
- Description: Возвращает исправленное изображение по имени файла.

Request Parameters:

    filename: Имя файла исправленного изображения.

Response:

    Файл исправленного изображения.

Пример вызова с помощью curl:

```sh
curl -O "http://localhost:8000/download_fixed_image/fixed_image.tif"
```

## 3. Скачать отчет

- Endpoint: /download_report/{filename}
- Method: GET
- Description: Возвращает отчет о битых пикселях по имени файла.

Request Parameters:

    filename: Имя файла отчета.

Response:

    Файл отчета о битых пикселях.

Пример вызова с помощью curl:

```sh
curl -O "http://localhost:8000/download_report/report_image.tif.txt"
```

4. Получить список подложек

- Endpoint: /layouts/
- Method: GET
- Description: Возвращает список доступных подложек.

Request: None

Response:

    Список подложек (массив объектов, каждый из которых содержит name и description).

Пример вызова с помощью curl:

```sh
curl "http://localhost:8000/layouts/"
```

## 5. Назначить активную подложку

- Endpoint: /layouts/active/{layout_name}
- Method: PUT
- Description: Назначает подложку активной по имени.

Request Parameters:

    layout_name: Имя подложки.

Response:

    Сообщение о том, что подложка назначена активной.

Пример вызова с помощью curl:

```sh
curl -X PUT "http://localhost:8000/layouts/active/layout1"
```

## 6. Получить активную подложку

- Endpoint: /layouts/active/
- Method: GET
- Description: Возвращает текущую активную подложку.

Request: None

Response:

    Активная подложка (объект, содержащий name и description).

Пример вызова с помощью curl:

```sh
curl "http://localhost:8000/layouts/active/"
```

## Пример использования всех эндпоинтов

Загрузить изображение для обработки:

```sh
curl -X POST "http://localhost:8000/process_image/" -F "file=@/path/to/your/image.tif"
```

Скачать исправленное изображение:

```sh
curl -O "http://localhost:8000/download_fixed_image/fixed_image.tif"
```

Скачать отчет о битых пикселях:

```sh
curl -O "http://localhost:8000/download_report/report_image.tif.txt"
```

Получить список подложек:

```sh
curl "http://localhost:8000/layouts/"
```

Назначить активную подложку:

```sh
curl -X PUT "http://localhost:8000/layouts/active/layout1"
```

Получить текущую активную подложку:

```sh
curl "http://localhost:8000/layouts/active/"
```

