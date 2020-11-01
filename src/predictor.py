import os
import cv2
import sys
import math
import time
import glob
import random
import pickle
import warnings
import importlib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

# path
CENTERTRACK_PATH = 'CenterTrack/src/lib'
sys.path.insert(0, CENTERTRACK_PATH)

if sys.argv:
    del sys.argv[1:]

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method
 
        Args:
            model_path (str): Path to the trained model directory.
 
        Returns:
            bool: The return value. True for success, False otherwise.
        
        Note:
            - You cannot connect to external network during the prediction,
              so do not include such process as using urllib.request.
 
        """
        opt = opts().init()
        cls.model = Detector(opt)
        return True

    @classmethod
    def predict(cls, input):
        """Predict method
 
        Args:
            input (str): path to the video file you want to make inference from
 
        Returns:
            dict: Inference for the given input.
                format:
                    - filename []:
                        - category_1 []:
                            - id: int
                            - box2d: [left, top, right, bottom]
                        ...
        Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
        """
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)
        cls.min_score_thresh_ped = 0.4
        cls.min_score_thresh_car = 0.4
        
        frame_cnt = 0
        term = 0
        prev_time_model = time.time()
        
        predictions = []

        while True:
            ret, frame = cap.read()
            term = term + 1
            
            if not ret:
                break
                
            if cls.model is not None:
                # predict
                output_dict = cls.model.run(frame)['results']

                prediction = {}
                ped_list = []
                car_list = []
                for num in range(len(output_dict)):
                    if output_dict[num]['class'] == 1: # pedestrian
                        bbox = output_dict[num]['bbox']
                        if (output_dict[num]['score'] < cls.min_score_thresh_ped) or (cls.is_bbox_large_enough(bbox) == False):
                            continue
                        else:
                            ped_list.append({"id": output_dict[num]['tracking_id'], "box2d": [int(bbox[0]) ,int(bbox[1]), int(bbox[2]), int(bbox[3])]})

                    if output_dict[num]['class'] == 3: # car
                        bbox = output_dict[num]['bbox']
                        if (output_dict[num]['score'] < cls.min_score_thresh_car) or (cls.is_bbox_large_enough(bbox) == False):
                            continue
                        else:
                            car_list.append({"id": output_dict[num]['tracking_id'], "box2d": [int(bbox[0]) ,int(bbox[1]), int(bbox[2]), int(bbox[3])]})
                prediction["Pedestrian"] = ped_list
                prediction["Car"] = car_list


                frame_cnt += 1
                if term == 1:
                    prev_time = time.time()
            else:
                prediction = {"Car": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}],
                                "Pedestrian": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}]}

            predictions.append(prediction)
        cap.release()

        # Extracts only the frames that appear in more than 3 frames.
        predictions_cut = predictions.copy()
        ped_ids = []
        car_ids = []
        for frm in range(len(predictions)):
            for key in predictions[frm].keys():
                ids = predictions[frm][key]
                for j in range(len(ids)):
                    if key == 'Pedestrian':
                        ped_ids.append(ids[j]['id'])
                    if key == 'Car':
                        car_ids.append(ids[j]['id'])
    
        ped_ids_3f = []
        car_ids_3f = []
        if len(ped_ids) > 0:
            ped_count = pd.DataFrame(ped_ids)[0].value_counts()
            ped_ids_3f = list(ped_count[ped_count >= 3].index)
        if len(car_ids) > 0:
            car_count = pd.DataFrame(car_ids)[0].value_counts()
            car_ids_3f = list(car_count[car_count >= 3].index)
        
        for frm in range(len(predictions)):
            for key in predictions[frm].keys():
                ids = predictions[frm][key]
                cut_ids = []
                for j in range(len(ids)):
                    if key == 'Pedestrian':
                        if ids[j]['id'] in ped_ids_3f:
                            cut_ids.append(ids[j])
                    if key == 'Car':
                        if ids[j]['id'] in car_ids_3f:
                            cut_ids.append(ids[j])
                predictions_cut[frm][key] = cut_ids.copy()
        
        curr_time = time.time()
        exec_time_model = curr_time - prev_time_model        
        return {fname: predictions_cut}
    
    @classmethod
    def is_bbox_large_enough(cls,bbox):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        S = abs(xmax - xmin) * abs(ymax - ymin)
        if S >= 1024:
            return True
        else:
            return False
