import argparse
from process_image import main_process_func
from datetime import datetime, timezone

# Получение текущего времени в формате unix time
# unix_time = datetime.now(timezone.utc).timestamp() * 1000
# taskid = str(unix_time) + '.csv'

# timestamp = datetime.now(timezone.utc).timestamp()
# taskid = f"{int(timestamp * 1_000_000)}"
taskid = "result.csv"

# Создание парсера аргументов
parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--layout', type=str, required=True, help='Path to the layout file')
parser.add_argument('--crop', type=str, required=True, help='Path to the input file')

# Парсинг аргументов
args = parser.parse_args()

# Вызов основной функции с аргументами
main_process_func(args.layout, args.crop, taskid)
