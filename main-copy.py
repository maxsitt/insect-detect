import time
import subprocess
import threading
import logging

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
        subprocess.run(["python3", "insect-detect/capture.py"])
        logging.info("Capture script finished.")
    except Exception as e:
        logging.error(f"Error running capture script: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize and start LED thread
    led_thread = threading.Thread(target=run_led_script)
    logging.info("Starting LED script")
    led_thread.start()

    # Initialize capture thread and start it after a delay
    time.sleep(3)  # Keep this if you need a delay before starting capture
    capture_thread = threading.Thread(target=capture_script)
    logging.info("Starting capture script")
    capture_thread.start()

    # Wait for both threads to complete
    led_thread.join()
    capture_thread.join()

    logging.info("Main script execution complete")

if __name__ == "__main__":
    main()