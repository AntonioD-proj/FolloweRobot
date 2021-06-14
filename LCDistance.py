import RPi.GPIO as GPIO
import lcddriver
display = lcddriver.lcd()
import RPi.GPIO as GPIO
import time
import curses
from time import sleep

PIN_TRIGGER = 18
PIN_ECHO = 19


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


GPIO.setup(PIN_TRIGGER, GPIO.OUT)
GPIO.setup(PIN_ECHO, GPIO.IN)


def Distance():
    display.lcd_clear()
    display.lcd_display_string('Distance is: ', 1)

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
    display.lcd_display_string(str(distance), 2)
    print("Distance is:", str(distance), "cm\t")
    
    if (distance<10):
            display.lcd_clear()
            display.lcd_display_string("Object too close", 1)
            display.lcd_display_string(str(distance), 2)
    elif(distance>300):
            display.lcd_clear()
            display.lcd_display_string("Nothing found", 1)
            display.lcd_display_string("in next 3 meters", 2)
              
        
  
while True:
    Distance()
    time.sleep(1)
             
          



