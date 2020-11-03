import cv2
# check source index
def show_webcam(mirror,index):
    cam = cv2.VideoCapture(index)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

for i in range(5):
    show_webcam(False, i)