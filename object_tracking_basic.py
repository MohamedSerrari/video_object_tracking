# imports
import os 
import argparse
import cv2
from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess


from utils.object_tracking_utils import *


aspect_ratios = [[1,1], [0.7,0.7], [1.2, 1.2]]
steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# loading trained vgg model
model = load_model('weights/vgg16_based__9.h5')
# model = load_model('projet_analyse_video/weights/vgg16_based__9.h5')




if __name__ == "__main__":
    
    # video and boxes directory 
    videos_dir = 'VIDEOS/'
    boxes_dir =  'GT/'

    # default video displayed if no arguments are passed
    index_video = randint(0,len(os.listdir(videos_dir))-1)
    video_path = os.listdir(videos_dir)[index_video] 
    gt_path = os.listdir(boxes_dir)[index_video]

    # choose name and gt of displayed video
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", default = videos_dir + str(video_path))
    ap.add_argument("--gt_path", default = boxes_dir + str(gt_path))
    video_path = ap.parse_args().video_path
    gt_path = ap.parse_args().gt_path

    # read video
    predictions = {}
    labelIndex, bboxes = read_annotation_file(gt_path)
    frame_number = 1 
    cap = cv2.VideoCapture(video_path)
    tracking = False

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False : 
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if tracking or frame_number in bboxes : 
            if not tracking : 
                assert(frame_number in bboxes)
                current_bbox = bboxes[frame_number]
                tracking = True 
            else : 

                # get candidate patches
                candidate_bboxes = get_candidate_boxes(current_bbox, steps, aspect_ratios, fw, fh)
                candidate_bboxes_patches = get_candidate_patches(frame, candidate_bboxes)

                # select best candidate box
                candidate_bboxes_predictions = model.predict(preprocess(candidate_bboxes_patches))
                best_prediction_index = np.argmax(candidate_bboxes_predictions[:, labelIndex])


                print(f"Frame {frame_number}, index : {best_prediction_index}, Confidence: {candidate_bboxes_predictions[best_prediction_index, labelIndex]}")
                print(f"Number of very good  > 0.9 {np.count_nonzero(candidate_bboxes_predictions[:, labelIndex] > 0.9)}")
                
                best_bbox = candidate_bboxes[best_prediction_index]

                # Update current box
                current_bbox = best_bbox
                predictions[frame_number] = best_bbox

            x, y, w, h = current_bbox
            cv2.putText(frame, class_names[labelIndex], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break 
        frame_number += 1

    cap.release()
