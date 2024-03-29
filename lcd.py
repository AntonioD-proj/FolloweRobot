import lcddriver
import time
from RPLCD.i2c import CharLCD

mylcd = lcddriver.lcd()


import RPi.GPIO as GPIO
import time

try:
    while True:
        
    
        GPIO.setmode(GPIO.BOARD)

        PIN_TRIGGER = 18
        PIN_ECHO = 19

        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        GPIO.setup(PIN_ECHO, GPIO.IN)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        GPIO.output(PIN_TRIGGER, GPIO.HIGH)

        time.sleep(0.00001)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        while GPIO.input(PIN_ECHO)==0:
            pulse_start_time = time.time()
        while GPIO.input(PIN_ECHO)==1:
            pulse_end_time = time.time()

        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        mylcd.lcd_display_string("Distance: ",distance)
        print ("Distance:",distance,"cm")

finally:
      GPIO.cleanup()



mylcd.lcd_display_string(u"Hello world!")
