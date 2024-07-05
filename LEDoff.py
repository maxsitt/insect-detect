import RPi.GPIO as GPIO
import time

def run_LEDS():
    # Set the GPIO mode
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.cleanup()

    # Set the GPIO pins for the LEDs
    UV_LED_PIN = 22  # UV light
    WHITE_LED_PIN = 17  # White light


    # Set the GPIO pins as outputs
    GPIO.setup(UV_LED_PIN, GPIO.OUT)
    GPIO.setup(WHITE_LED_PIN, GPIO.OUT)

    # Turn off the LEDs
    GPIO.output(UV_LED_PIN, GPIO.LOW)
    GPIO.output(WHITE_LED_PIN, GPIO.LOW)  # Turn off the new LED
    print("LEDs OFF")

    time.sleep(5)

if __name__ == "__main__":
    run_LEDS()