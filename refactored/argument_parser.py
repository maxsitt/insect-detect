import argparse

def parse_arguments():
    """
    Parses command-line arguments for the application.
    """
    parser = argparse.ArgumentParser(description="Run object detection and tracking.")
    parser.add_argument("-4k", "--four_k_resolution", action="store_true",
                        help="crop detections from (+ save HQ frames in) 4K resolution; default = 1080p")
    parser.add_argument("-crop", "--crop_bbox", choices=["square", "tight"], default="square", type=str,
                        help="save cropped detections with aspect ratio 1:1 ('-crop square') or \
                              keep original bbox size with variable aspect ratio ('-crop tight')")
    parser.add_argument("-raw", "--save_raw_frames", action="store_true",
                        help="additionally save full raw HQ frames in separate folder (e.g., for training data)")
    parser.add_argument("-overlay", "--save_overlay_frames", action="store_true",
                        help="additionally save full HQ frames with overlay (bbox + info) in separate folder")
    parser.add_argument("-log", "--save_logs", action="store_true",
                        help="save RPi CPU + OAK chip temperature, RPi available memory (MB) + \
                              CPU utilization (%) and battery info to .csv file")

    args = parser.parse_args()
    if args.save_logs:
        from apscheduler.schedulers.background import BackgroundScheduler
        from gpiozero import CPUTemperature
    return args
