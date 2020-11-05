import cv2
import numpy as np
import pandas as pd
import time
from imutils.video import FPS, WebcamVideoStream
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

print("GPU CUDA :", torch.cuda.is_available())
def startWebCam(model):
    def draw_predictions(annotation):
        for box in annotation:
            if(len(box['boxes'])<1):
                break

            for i,cord in enumerate(box['boxes']):
                xmin, ymin, xmax, ymax = cord
                color = None
                label = None

                # Create a Rectangle patch
                if (box['labels'][i]==1):
                    color = COLORS[0]
                    label = "With Mask"
                elif (box['labels'][i]==2):
                    color = COLORS[1]
                    label = "With Mask Incorrect"

                cv2.rectangle(frame,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)),
                                color, 1)
                cv2.putText(frame, label, (int(xmin), int(ymin)),
                            FONT, 0.4, color, 1, cv2.LINE_AA)

    def predict(frame):
        #height, width = frame.shape[:2]
        x = torch.from_numpy(np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)).float().cuda()
        y = model(x) # forward pass
        print(y)
        draw_predictions(y)


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
        fps.update()
        predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break

def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

if __name__ == '__main__':
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load("../model/weights/0311.pth"))
    model.eval()
    model.to(device)
    torch.backends.cudnn.benchmark = True

    print("LOG : Load Weight Successfully")
    fps = FPS().start()
    startWebCam(model)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. Prediction per second: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    