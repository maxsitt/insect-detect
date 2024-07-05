import time
import subprocess
import threading
import logging
import signal

def run_led_script():
    try:
        import LED
        logging.info("LED script running...")
        LED.run_LEDS()
        logging.info("LED script finished.")
    except Exception as e:
        logging.error(f"Error running LED script: {e}")
        raise

def capture_script():
    try:
        logging.info("Capture script starting...")
        process = subprocess.Popen(["python3", "insect-detect/yolo_tracker_save_hqsync.py --4k"])
        return process
    except Exception as e:
        logging.error(f"Error running capture script: {e}")
        raise

def terminate_process(process):
    try:
        process.terminate()
        process.wait(timeout=5)
        logging.info("Capture script terminated.")
    except Exception as e:
        logging.error(f"Error terminating capture script: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    duration = 2 * 60 * 60  # 2 hours in seconds

    while time.time() - start_time < duration:
        # Start LED script in a separate thread
        led_thread = threading.Thread(target=run_led_script)
        logging.info("Starting LED script")
        led_thread.start()

        # Wait for 5 minutes before starting the capture script
        time.sleep(5 * 60)

        # Start capture script in a separate process
        logging.info("Starting capture script")
        capture_process = capture_script()

        # Run capture script for 15-20 minutes
        capture_duration = 15 * 60  # 15 minutes in seconds
        time.sleep(capture_duration)

        # Terminate the capture script
        logging.info("Terminating capture script")
        terminate_process(capture_process)

        # Wait for the LED script to finish if it's still running
        led_thread.join()

        # Wait for 10 minutes break before the next cycle
        time.sleep(10 * 60)

    logging.info("Main script execution complete after 2 hours")

if __name__ == "__main__":
    main()
