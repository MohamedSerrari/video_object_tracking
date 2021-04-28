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

# 
aspect_ratios = [[1,1], [0.7,0.7], [1.2, 1.2]]
steps =  [0, 0.5, 1, 1.5]

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

    previous_frame = None

    tr_ratio = 1

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False : 
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        confidence = 0

        if tracking or frame_number in bboxes : 
            if not tracking : 
                assert(frame_number in bboxes)
                original_bbox = squared_boxes(*bboxes[frame_number])
                current_bbox = original_bbox
                tracking = True 

                color = (255, 0, 0)
                title = f'Tracking object {class_names[labelIndex]}'
            else : 

                # calculate translation ratio using optical flow
                if previous_frame is not None:
                    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    angles, transl, lines = calc_angl_n_transl(prev_gray, flow)
                    ang_mode, transl_mode, tr_ratio, steady = estimate_motion(angles, transl)

                # get candidate patches
                scaled_steps = [s * 2 * tr_ratio for s in steps] # scale steps depending on translation ratio of optical flow


                # avoid recursive rescaling which leads values to diverge:
                if True:
                    current_bbox = current_bbox[0], current_bbox[1], original_bbox[2], original_bbox[3]

                candidate_bboxes = get_candidate_boxes(current_bbox, scaled_steps, aspect_ratios, fw, fh)
                candidate_bboxes_patches = get_candidate_patches(frame, candidate_bboxes)

                # select best candidate box
                candidate_bboxes_predictions = model.predict(preprocess(candidate_bboxes_patches))
                best_prediction_index = np.argmax(candidate_bboxes_predictions[:, labelIndex])
                
                # Store best bounding box confidence
                confidence = candidate_bboxes_predictions[best_prediction_index, labelIndex]

                # Since sometimes there are multiple boxes that have score 1
                # Taking the first one doesn't insure we selected the best one
                # That is why we average the boxes that have score > 0.9
                # Select candidate boxes with score > 0.5
                best_indexes = candidate_bboxes_predictions[:, labelIndex] > 0.5
                if np.count_nonzero(best_indexes) > 0:
                    # select boxes and average them
                    #TODO A better average would be to go to cxcywh coords => average => xywh coords
                    best_bboxes = candidate_bboxes[best_indexes]
                    avg_bboxes = np.average(best_bboxes, axis=0)

                    # Update current box
                    current_bbox = squared_boxes(*avg_bboxes)
                    predictions[frame_number] = current_bbox

                if confidence > 0.5: # object still tracked
                    color = (0, 255, 0)
                    title = f'{class_names[labelIndex]} : {confidence:0.2f}'
                else: # object lost
                    color = (0, 0, 255)
                    title = f'Object {class_names[labelIndex]} lost : {confidence:0.2f}'

            x, y, w, h = [int(e) for e in current_bbox]

            cv2.putText(frame, title, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break 
        frame_number += 1
        previous_frame = frame.copy()

    cap.release()

