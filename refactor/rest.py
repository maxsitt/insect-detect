import time
import logging
import depthai as dai
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import traceback
from data_log import save_logs, record_log
from data_management import store_data
# Ensure that save_logs and record_log functions are imported or defined here
# from your_logging_module import save_logs, record_log
# from your_data_handling_module import store_data
# Ensure the args variable is accessible if needed, consider passing it as a parameter
def run(save_logs, save_raw_frames, save_overlay_frames, crop_bbox, four_k_resolution, webhook_url, latest_images, image_count, labels, pijuice, chargelevel_start, logger, pipeline, rec_id, rec_start, save_path):
    # Connect to OAK device and start pipeline in USB2 mode


    with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

        # Write RPi + OAK + battery info to .csv log file at specified interval
        if save_logs:
            logging.getLogger("apscheduler").setLevel(logging.WARNING)
            scheduler = BackgroundScheduler()
            scheduler.add_job(save_logs, "interval", seconds=30, id="log")
            scheduler.start()

        # Create empty list to save charge level (if < 10) and set charge level
        lst_chargelevel = []
        chargelevel = chargelevel_start

        # Set recording time conditional on PiJuice battery charge level
        if chargelevel >= 70:
            rec_time = 60 * 40
        elif 50 <= chargelevel < 70:
            rec_time = 60 * 30
        elif 30 <= chargelevel < 50:
            rec_time = 60 * 20
        elif 15 <= chargelevel < 30:
            rec_time = 60 * 10
        else:
            rec_time = 60 * 5

        # Write info on start of recording to log file
        logger.info(f"Rec ID: {rec_id} | Rec time: {int(rec_time / 60)} min | Charge level: {chargelevel}%")

        # Create output queues to get the frames and tracklets + detections from the outputs defined above
        q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
        q_track = device.getOutputQueue(name="track", maxSize=4, blocking=False)

        # Set start time of recording
        start_time = time.monotonic()

        try:
            # Record until recording time is finished or charge level dropped below threshold for 10 times
            while time.monotonic() < start_time + rec_time and len(lst_chargelevel) < 10:

                # Update charge level (return "99" if not readable and write to list if < 10)
                chargelevel = pijuice.status.GetChargeLevel().get("data", 99)
                if chargelevel < 10:
                    lst_chargelevel.append(chargelevel)

                # Get synchronized HQ frames + tracker output (passthrough detections)
                if q_frame.has():
                    frame = q_frame.get().getCvFrame()

                    if q_track.has():
                        tracks = q_track.get().tracklets

                        # Save cropped detections (slower if saving additional HQ frames)
                        store_data(frame, tracks, rec_id, rec_start, save_path, labels, save_raw_frames, save_overlay_frames, crop_bbox, four_k_resolution, webhook_url, latest_images, image_count)

                # Wait for 1 second
                time.sleep(1)

            # Write info on end of recording to log file and write record logs to .csv
            logger.info(f"Recording {rec_id} finished | Charge level: {chargelevel}%\n")
            record_log(rec_id, rec_start, save_path, chargelevel_start, chargelevel, start_time)

            # Enable charging of PiJuice battery if charge level is lower than threshold
            if chargelevel < 80:
                pijuice.config.SetChargingConfig({"charging_enabled": True})

            # Shutdown Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

        # Write info on error during recording to log file and write record logs to .csv
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Error during recording {rec_id} | Charge level: {chargelevel}%\n")
            record_log(rec_id, rec_start, save_path, chargelevel_start, chargelevel, start_time)

            # Enable charging of PiJuice battery if charge level is lower than threshold
            if chargelevel < 80:
                pijuice.config.SetChargingConfig({"charging_enabled": True})

            # Shutdown Raspberry Pi
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
    return frame, tracks