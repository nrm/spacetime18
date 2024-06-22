import argparse
from process_image import main_process_func
from datetime import datetime, timezone


# Создание парсера аргументов
parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--layout', type=str, required=True, help='Path to the layout file')
parser.add_argument('--crop', type=str, required=True, help='Path to the input file')
parser.add_argument('--taskid', type=str, required=True, help='taskid, name result files with extension')

# Парсинг аргументов
args = parser.parse_args()

# taskid = "result.csv"

# Вызов основной функции с аргументами
main_process_func(args.layout, args.crop, args.taskid)
