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

    time.sleep(1)

    # Turn on the UV LED
    GPIO.output(UV_LED_PIN, GPIO.HIGH)
    print("UV LED ON - Attracting bugs")

    # Wait for 5 minutes before turning on the white LED
    time.sleep(5*60)  # 300 seconds = 5 minutes

    # Turn on the white LED
    GPIO.output(WHITE_LED_PIN, GPIO.HIGH)
    print("White LED ON")

    time.sleep(600)

    # Turn off the LEDs
    GPIO.output(UV_LED_PIN, GPIO.LOW)
    GPIO.output(WHITE_LED_PIN, GPIO.LOW)  # Turn off the new LED
    print("LEDs OFF")

if __name__ == "__main__":
    run_LEDS()