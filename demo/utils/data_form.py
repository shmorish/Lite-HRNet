import json
import re

from cv2 import pointPolygonTest


class Form():
    def data_form(dict, i , keypoint):
        
        if i == 0:
            dict["nose"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 1:
            dict["left_eye"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 2:
            dict["right_eye"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 3:
            dict["lft_ear"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 4:
            dict["right_ear"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 5:
            dict["left_shoulder"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 6:
            dict["right_shoulder"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 7:
            dict["left_elbow"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 8:
            dict["right_elbow"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 9:
            dict["left_wrist"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 10:
            dict["right_wrist"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 11:
            dict["left_hip"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 12:
            dict["right_hip"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 13:
            dict["left_knee"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 14:
            dict["right_knee"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 15:
            dict["left_ankle"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        elif i == 16:
            dict["right_ankle"] = [float(keypoint[0]),float(keypoint[1]), float(keypoint[2])]
        return dict

    def make_dic(frame, bbox, keypoint_dic):
        

        data = dict()
        data["frame"] = frame
        data["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])]
        data["keypoints"] = keypoint_dic
        
        
        return data