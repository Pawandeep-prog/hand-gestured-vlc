import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pynput.keyboard import Controller, Key
import time
cont= Controller()
flag = True

model = load_model('finger.hdf')

cap = cv2.VideoCapture(0)

start = time.time()


while True:
    wind = np.zeros((200,200,3))
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    show = frame[50:200, 400:550]
    frame = cv2.blur(frame, (2,2))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = gray[50:200, 400:550]
    
    _, mask = cv2.threshold(gray,120 ,255 ,cv2.THRESH_BINARY_INV)
    mask = mask / 255.0
    mask = cv2.resize(mask, (128,128))
    mask = mask.reshape(-1,128,128,1)    
    
    ############################
 
    result=model.predict(mask)
    res = np.argmax(result)
    #print(res)

    cv2.putText(wind, "{}".format(res),(50,125), cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

    
    if flag:
        if res == 0:
            cont.press(Key.space)
            cont.release(Key.space)
            flag = False
        elif res == 1:
            cont.press(Key.up)
            cont.release(Key.up)
            flag = False            
        elif res == 2: 
            cont.press(Key.down)
            cont.release(Key.down) 
            flag = False
        elif res == 3:
             cont.press(Key.left)
             cont.release(Key.left)  
             flag = False
        elif res == 4:
             cont.press(Key.right)
             cont.release(Key.right)   
             flag = False
    ############################
 
    cv2.imshow("main", show)
    cv2.imshow("result", mask.reshape(128,128))
    cv2.imshow("", wind)

    end = time.time()
    if (end - start) > 2:
        start = end
        flag = True
 
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


