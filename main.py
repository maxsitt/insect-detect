import time
import subprocess
import threading
import logging

def run_led_script():
    try:
        import LEDcopy
        logging.info("LED script running...")
        LEDcopy.run_LEDS()
        logging.info("LED script finished.")
    except Exception as e:
        logging.error(f"Error running LED script: {e}")
        raise

def run_capture_script():
    try:
        logging.info("Capture script starting...")
        subprocess.run(["python3", "insect-detect/run_capture.py"])
        logging.info("Capture script finished.")
    except Exception as e:
        logging.error(f"Error running capture script: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO)

    # Start LED script in a separate thread
    led_thread = threading.Thread(target=run_led_script)
    logging.info("Starting LED script")
    led_thread.start()

    # Wait for 1 minute before starting the capture script
    time.sleep(3)

    # Start capture script and wait for it to complete
    logging.info("Starting capture script")
    run_capture_script()

    # Wait for the LED script to finish if it's still running
    led_thread.join()

    logging.info("Main script execution complete")

if __name__ == "__main__":
    main()
