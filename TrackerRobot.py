import cv2
import RPi.GPIO as GPIO
import lcddriver
display = lcddriver.lcd()
import time
from time import sleep

#HC-SR04 ultrasonic distance sensor
PIN_TRIGGER = 18
PIN_ECHO = 19

#DxMx - Driver 1/2 Motor 1/2 Forward/Backward
D2M1F = 21
D2M1B = 22
D2M2F = 23
D2M2B = 24
D1M1F = 31
D1M1B = 29
D1M2F = 35
D1M2B = 33

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

GPIO.setup(PIN_TRIGGER, GPIO.OUT)
GPIO.setup(PIN_ECHO, GPIO.IN)

#Motors
GPIO.setup(D1M1F,GPIO.OUT)
GPIO.setup(D1M1B,GPIO.OUT)
GPIO.setup(D1M2F,GPIO.OUT)
GPIO.setup(D1M2B,GPIO.OUT)
GPIO.setup(D2M1F,GPIO.OUT)
GPIO.setup(D2M1B,GPIO.OUT)
GPIO.setup(D2M2F,GPIO.OUT)
GPIO.setup(D2M2B,GPIO.OUT)

print("If the invironment is dark pres '0' and if it is well lighted press '1'")
x = int (input("Chose between 0 and 1: "))

cap = cv2.VideoCapture(x)
#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerMedianFlow_create()
success, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)



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
    
    if (distance<5):
            display.lcd_clear()
            display.lcd_display_string("The object is", 1)
            display.lcd_display_string("too close", 2)
    elif(distance>200):
            display.lcd_clear()
            display.lcd_display_string("The object is", 1)
            display.lcd_display_string("too far", 2)
              
        
    
def forward(): #all the four mottros works together to move forward
    GPIO.output(D1M1F,True)
    GPIO.output(D1M1B,False)
    GPIO.output(D1M2F,True)
    GPIO.output(D1M2B,False)
    GPIO.output(D2M1F,True)
    GPIO.output(D2M1B,False)
    GPIO.output(D2M2F,True)
    GPIO.output(D2M2B,False)

def backward(): #same mottors but moving backward
    GPIO.output(D1M1F,False)
    GPIO.output(D1M1B,True)
    GPIO.output(D1M2F,False)
    GPIO.output(D1M2B,True)
    GPIO.output(D2M1F,False)
    GPIO.output(D2M1B,True)
    GPIO.output(D2M2F,False)
    GPIO.output(D2M2B,True)

def right(): #right sided motors moves backward and left sided motors move forward
    GPIO.output(D1M1F,False)
    GPIO.output(D1M1B,True)
    GPIO.output(D1M2F,True)
    GPIO.output(D1M2B,False)
    GPIO.output(D2M1F,False)
    GPIO.output(D2M1B,True)
    GPIO.output(D2M2F,True)
    GPIO.output(D2M2B,False)

def left(): # oposite do the right direction 
    GPIO.output(D1M1F,True)
    GPIO.output(D1M1B,False)
    GPIO.output(D1M2F,False)
    GPIO.output(D1M2B,True)
    GPIO.output(D2M1F,True)
    GPIO.output(D2M1B,False)
    GPIO.output(D2M2F,False)
    GPIO.output(D2M2B,True)

def stop(): #all the motors is turned off
    GPIO.output(D1M1F,False)
    GPIO.output(D1M1B,False)
    GPIO.output(D1M2F,False)
    GPIO.output(D1M2B,False)
    GPIO.output(D2M1F,False)
    GPIO.output(D2M1B,False)
    GPIO.output(D2M2F,False)
    GPIO.output(D2M2B,False)
    
while True:
    #Distance()
    timer = cv2.getTickCount()
    success, frame = cap.read()
    success, bbox =  tracker.update(frame)
    
    print(bbox)
    
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
      
    
    coord = [x, y]
    shape = [w, h]
    min_shape = [60, 140]
    max_shape = [100, 220]
    mid_shape1 = [80, 180]
    coord_left = [240, y]
    coord_right = [400, y]
    
    #if coord: 
    if (shape[0] > min_shape[0]) and (shape[0] < max_shape[0]):
        cv2.imshow("Tracking", bbox)
            
        if(coord[0] > coord_right[0]):
            print("Turning Right")
            right()
        elif(coord[0] < coord_left[0]):
            print("Turning Left")
            left()
        else:
            print("Forward")
            forward()
    elif(shape[0] > min_shape[0]) and (shape[0] > max_shape[0]):
        print("Stop")
        stop()
    else:
        print("Stop")
        stop()
#     else:
#         cv2.putText(frame, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
#         print("Object Not Found")

    cv2.imshow("Tracking", bbox)        
        