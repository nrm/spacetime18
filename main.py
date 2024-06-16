import argparse
from process_image import main_process_func
from datetime import datetime, timezone

def main(crop_name, layout_name):
    unix_time = datetime.now(timezone.utc).timestamp() * 1000
    taskid = f"{int(unix_time)}"

    main_process_func(layout_name, crop_name, taskid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main process function with crop and layout files")
    parser.add_argument("--crop_name", type=str, required=True, help="Path to the crop file")
    parser.add_argument("--layout_name", type=str, required=True, help="Path to the layout file")

    args = parser.parse_args()
    main(args.crop_name, args.layout_name)
