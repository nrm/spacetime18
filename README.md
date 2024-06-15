# images

## Переменные окружения

В Dockerfile и docker-compose.yml, мы устанавливаем переменную окружения LAYOUTS_DIR, которая предоставляет путь к директории с подложками и используется приложением внутри контейнера.

А так же монтируем внутрь контейнера папку с подложками.

```yaml
...
volumes:
    - ./path/2/layouts:/layouts
environment:
    - LAYOUTS_DIR=/layouts
```

## Запуск приложения с Docker Compose

Соберите и запустите контейнер:

```sh
docker compose up --build
```

Приложение будет доступно по адресу http://localhost:8000.

## Пример работы с API

 Описан в файле API_docs.md


 ## Add gitlab CI