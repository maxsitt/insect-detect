import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Set the GPIO pins
LED_PIN_1 = 17
LED_PIN_2 = 22  # The new pin

# Set the GPIO pins as outputs
GPIO.setup(LED_PIN_1, GPIO.OUT)
GPIO.setup(LED_PIN_2, GPIO.OUT)  # Setup for the new pin

while True:
    # Turn on the LEDs
    GPIO.output(LED_PIN_1, GPIO.HIGH)
    GPIO.output(LED_PIN_2, GPIO.HIGH)  # Turn on the new LED
    print("LEDs ON")
    
    # Keep the LEDs on for 1 second
    time.sleep(1)
    
    # Turn off the LEDs
    GPIO.output(LED_PIN_1, GPIO.LOW)
    GPIO.output(LED_PIN_2, GPIO.LOW)  # Turn off the new LED
    print("LEDs OFF")
    
    time.sleep(1)

# Cleanup
GPIO.cleanup()  # This will only run if the loop is exited, which doesn't happen in this script
