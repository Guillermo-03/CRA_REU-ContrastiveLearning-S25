import cv2

cv2.namedWindow("frame")
vc = cv2.VideoCapture(0) # 0: first camera / 1: second camera 

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20.0, (640,  480))
 

if not vc.isOpened(): # try to get the first frame
    print("Cannot open camera. Exiting...")
    exit()

rval, frame = vc.read() # rval True if first frame returned
 
if not rval:
    print('Frame not received. Exiting...')

while rval:
    cv2.imshow("frame", frame)
    rval, frame = vc.read() 
    frame = cv2.flip(frame, 0)
    out.write(frame)

    if cv2.waitKey(20) == 27: # waits 20ms and exit on ESC
        break

vc.release()
out.release()
cv2.destroyAllWindows()
