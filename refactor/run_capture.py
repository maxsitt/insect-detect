#!/usr/bin/env python3

from logging_setup import setup_logging
from argument_parser import parse_arguments
from power_management import check_system_resources
from setup_pipeline import create_pipeline
from setup_directories import setup_directories
from data_management import store_data
from rest import run

# Other imports remain the same
# import csv, json, subprocess, sys, time, traceback, etc.

latest_images = {}
image_count = {}  # Dictionary to keep track of image count for each track.id
webhook_url = "https://nytelyfe-402203.uc.r.appspot.com/upload" # Webhook URL

def capture(args):
    
    # Setup logging first
    logger = setup_logging()

    # Check system resources and manage power
    pijuice, chargelevel_start = check_system_resources(logger)
    
    # Create the DepthAI pipeline
    pipeline, labels = create_pipeline(args.four_k_resolution)
    print(f"Labels type: {type(labels)}")
    print(f"Labels content: {labels}")

    # Set up data directories
    save_path, rec_id, rec_start = setup_directories(labels, args.save_raw_frames, args.save_overlay_frames)

    run(args.save_logs, args.save_raw_frames, args.save_overlay_frames, args.crop_bbox, args.four_k_resolution, webhook_url, latest_images, image_count, labels, pijuice, chargelevel_start, logger, pipeline, rec_id, rec_start, save_path)

    # Store and send data
    #store_data(frame, tracks, rec_id, rec_start, save_path, labels, args.save_raw_frames, args.save_overlay_frames, args.crop_bbox, args.four_k_resolution, webhook_url, latest_images, image_count)
    #store_data(frame, tracks, rec_id, rec_start, save_path, labels, args.save_raw_frames, args.save_overlay_frames, args.crop_bbox, args.four_k_resolution, webhook_url, latest_images, image_count)

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_arguments()
    # Pass the parsed arguments to the capture function
    capture(args)
