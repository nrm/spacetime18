from process_image import main_process_func
from datetime import datetime, timezone

unix_time = datetime.now(timezone.utc).timestamp()*1000
taskid =str(unix_time) + '.csv'

layout_name = '../18/layouts/layout_2021-10-10.tif'
input_file_path = '../18/1_20/crop_0_2_0000.tif'

main_process_func(layout_name, input_file_path, taskid)