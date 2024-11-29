
# 1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
# 2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
#    1. `frame_dir` (str): The identifier of the corresponding video.
#    2. `total_frames` (int): The number of frames in this video.
#    3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
#    4. `original_shape` (tuple[int]): Same as `img_shape`.
#    5. `label` (int): The action label.
#    6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
#    7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

import json
import glob
from tqdm import tqdm
import numpy as np
from mmcv import dump

np.set_printoptions(threshold=np.inf)

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def main():
    json_files = glob.glob('result/*.json')
    frame_dirs = []
    annotations = []
    print('Processing json files...')
    print('Total files:', len(json_files))
    if len(json_files) == 0:
        print('No json files found!')
        return
    for json_file in tqdm(json_files):
        frame_dir = json_file.split('.')[0].split('_')[0]
        frame_dirs.append(frame_dir)
        label = int(frame_dir[-3:]) - 1
        data = load_json(json_file)

        keypoint = []
        keypoint_score = []
        image_shape = (1080, 1920)
        for key in data.keys():
            keypoints = data[key]['keypoints']
            total_frames = data[key]['frame']

            nose = keypoints['nose'][:2]
            nose_score = keypoints['nose'][2]

            left_eye = keypoints['left_eye'][:2]
            left_eye_score = keypoints['left_eye'][2]

            right_eye = keypoints['right_eye'][:2]
            right_eye_score = keypoints['right_eye'][2]

            lft_ear = keypoints['lft_ear'][:2]
            lft_ear_score = keypoints['lft_ear'][2]

            right_ear = keypoints['right_ear'][:2]
            right_ear_score = keypoints['right_ear'][2]

            left_shoulder = keypoints['left_shoulder'][:2]
            left_shoulder_score = keypoints['left_shoulder'][2]

            right_shoulder = keypoints['right_shoulder'][:2]
            right_shoulder_score = keypoints['right_shoulder'][2]

            left_elbow = keypoints['left_elbow'][:2]
            left_elbow_score = keypoints['left_elbow'][2]

            right_elbow = keypoints['right_elbow'][:2]
            right_elbow_score = keypoints['right_elbow'][2]

            left_wrist = keypoints['left_wrist'][:2]
            left_wrist_score = keypoints['left_wrist'][2]

            right_wrist = keypoints['right_wrist'][:2]
            right_wrist_score = keypoints['right_wrist'][2]

            left_hip = keypoints['left_hip'][:2]
            left_hip_score = keypoints['left_hip'][2]

            right_hip = keypoints['right_hip'][:2]
            right_hip_score = keypoints['right_hip'][2]

            left_knee = keypoints['left_knee'][:2]
            left_knee_score = keypoints['left_knee'][2]

            right_knee = keypoints['right_knee'][:2]
            right_knee_score = keypoints['right_knee'][2]
            
            left_ankle = keypoints['left_ankle'][:2]
            left_ankle_score = keypoints['left_ankle'][2]
            
            right_ankle = keypoints['right_ankle'][:2]
            right_ankle_score = keypoints['right_ankle'][2]

            single_keypoint = [nose, left_eye, right_eye, lft_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
            single_keypoint_score = [nose_score, left_eye_score, right_eye_score, lft_ear_score, right_ear_score, left_shoulder_score, right_shoulder_score, left_elbow_score, right_elbow_score, left_wrist_score, right_wrist_score, left_hip_score, right_hip_score, left_knee_score, right_knee_score, left_ankle_score, right_ankle_score]

            keypoint.append(single_keypoint)
            keypoint_score.append(single_keypoint_score)

        annotation_dict = {
            'frame_dir': frame_dir,
            'label': label,
            'img_shape': image_shape,
            'original_shape': image_shape,
            'total_frames': total_frames,
            'keypoint': np.array([keypoint]),
            'keypoint_score': np.array([keypoint_score])
        }
        annotations.append(annotation_dict)
    names = [name for name in frame_dirs if int(name.split('A')[-1]) <= 60]
    xsub_train = [name for name in names if int(name.split('P')[1][:3]) in training_subjects]
    xsub_val = [name for name in names if int(name.split('P')[1][:3]) not in training_subjects]
    xveiw_train = [name for name in names if 'C001' not in name]
    xview_val = [name for name in names if 'C001' in name]
    split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xview_train=xveiw_train, xview_val=xview_val)
    dict_data = dict(split=split, annotations=annotations)
    dump(dict_data, 'ntu60_lite_hrnet.pkl')

if __name__ == '__main__':
    main()