import cv2
import numpy as np
import pandas as pd
import time
from imutils.video import FPS, WebcamVideoStream
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#   grad_fn=<StackBackward>), 'labels': tensor([2, 3, 1], device='cuda:0'), 'scores': tensor([0.9774, 0.0604, 0.0529]
#   grad_fn=<StackBackward>), 'labels': tensor([1, 3], device='cuda:0'), 'scores': tensor([0.9533, 0.1024]
#   grad_fn=<StackBackward>), 'labels': tensor([1, 2, 3, 1], device='cuda:0'), 'scores': tensor([0.8836, 0.8032, 0.2569, 0.2145]
import threading

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
frame = None
annotation = None
active = True

print("GPU CUDA :", torch.cuda.is_available())
def startWebCam(model):
    

    def draw_predictions():
        global frame, annotation

        if annotation is None:
            return
        
        for box in annotation:
            if(len(box['boxes'])<1):
                break

            for i,cord in enumerate(box['boxes']):
                xmin, ymin, xmax, ymax = cord
                color = None
                label = None

                # Create a Rectangle patch
                if (box['labels'][i]==1):
                    color = COLORS[1]
                    label = "Without Mask"
                elif (box['labels'][i]==2):
                    color = COLORS[0]
                    label = "With Mask"
                elif (box['labels'][i]==3):
                    color = COLORS[2]
                    label = "With Mask Incorrect"

                cv2.rectangle(frame,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)),
                                color, 1)
                cv2.putText(frame, label, (int(xmin), int(ymin)),
                            FONT, 0.4, color, 1, cv2.LINE_AA)

    def predict():
        global frame, annotation, active
        time.sleep(4.0)
        while active:
            #height, width = frame.shape[:2]
            #print(frame)
            x = torch.from_numpy(np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)).float()
            annotation = model(x) # forward pass
            # print(y)
            # draw_predictions(y)

    def update_stream():
        global frame, active
        stream = WebcamVideoStream(src=0).start()  # default camera
        time.sleep(1.0)
        # start fps timer
        # loop over frames from the video file stream
        while True:
            # grab next frame
            frame = stream.read()
            frame = frame/255
            key = cv2.waitKey(1) & 0xFF

            # update FPS counter
            #fps.update()
            draw_predictions()

            # keybindings for display
            if key == ord('p'):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', frame)
                    if key2 == ord('p'):  # resume
                        break
            cv2.imshow('frame', frame)
            if key == 27:  # exit
                active = False
                break

    t1=threading.Thread(target=update_stream)
    t2=threading.Thread(target=predict)

    t1.start()
    t2.start()

def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

if __name__ == '__main__':
    model = get_model_instance_segmentation(4)
    model.load_state_dict(torch.load("../model/weights/0511-30.pth"))

#    model.load_state_dict(torch.load("../model/weights/model_with_no_mask.pth"))
    model.eval()
    #model.to(device)
    torch.backends.cudnn.benchmark = True

    print("LOG : Load Weight Successfully")
    #fps = FPS().start()
    startWebCam(model)
    # stop the timer and display FPS information
    #fps.stop()

    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. Prediction per second: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    