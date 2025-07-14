import cv2
import geocoder

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 3.5
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def run_video():
    cv2.namedWindow("frame")
    vc = cv2.VideoCapture(0) # 0: first camera / 1: second camera 

    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20.0, (640,  480))

    if not vc.isOpened(): # try to get the first frame
        print("Cannot open camera. Exiting...")
        exit()

    rval, frame = vc.read() # rval True if first frame returned
    
    if not rval:
        print('Frame not received. Exiting...')

    g = geocoder.ip('me')

    while rval:

        __draw_label(frame, str(g.latlng), (100,100), (255,255,255))
        cv2.imshow("frame", frame)
        rval, frame = vc.read() 
        # frame = cv2.flip(frame, 0)
        # out.write(frame)

        if cv2.waitKey(20) == 27: # waits 20ms and exit on ESC
            break

    vc.release()
    # out.release()
    cv2.destroyAllWindows()

def main():
    run_video()

main()