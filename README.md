# images

## Пример работы с API

```sh
# Отправка файла на корректировку и анализ. Ответ в формате JSON
curl -X POST "http://172.20.180.215:9000/process_image/" -F "file=@crop_0_1_0000.tif"
{"fixed_image":"fixed_crop_0_1_0000.tif","report":"report_crop_0_1_0000.tif.txt"}

# Получение исправленного файла, без битых пикселей
curl -O "http://172.20.180.215:9000/download_fixed_image/fixed_crop_0_1_0000.tif"

# Получение отчета по исправленному файлу
curl -O "http://172.20.180.215:9000/download_fixed_image/report_crop_0_1_0000.tif.txt"
```

