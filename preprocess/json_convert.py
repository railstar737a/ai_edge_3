import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# to xywh
def conv_xywh(bbox):
    x_min = bbox[0]
    y_min = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    return [int(x_min), int(y_min) ,int(w) ,int(h)]

if __name__ == "__main__":
    # The directory containing the annotation files
    path_labels  = './train_annotations'

    # Traget class definition
    classes = ['Pedestrian', 'Truck','Car', 
            'Signal', 'Signs', 'Bicycle',
            'Motorbike', 'Bus', 'Svehicle', 'Train']

    ret_id = lambda arg : classes.index(arg) + 1

    annotations = os.listdir(path_labels)

    j_file = {}
    j_file['videos'] = []
    j_file['categories'] = [{'id': 1, 'name': 'Pedestrian'},
                            {'id': 2, 'name': 'Truck'},
                            {'id': 3, 'name': 'Car'},
                            {'id': 4, 'name': 'Signal'},
                            {'id': 5, 'name': 'Signs'},
                            {'id': 6, 'name': 'Bicycle'},
                            {'id': 7, 'name': 'Motorbike'},
                            {'id': 8, 'name': 'Bus'},
                            {'id': 9, 'name': 'Svehicle'},
                            {'id': 10, 'name': 'Train'}]
    j_file['images'] = []
    j_file['annotations'] = []

    idd = 0
    idd_box = 0
    for i in tqdm(range (0, len(annotations))):
        video_name = annotations[i].split('/')[-1].split('\\')[-1].split('.')[0]
        data = json.load(open(os.path.join(path_labels, annotations[i])))
        j_file['videos'].append({'file_name': video_name, 'id': i + 1})
        for v in range (0, 600):
            idd += 1
            labels = data['sequence'][v]
            # images
            if (v == 0):# 先頭/最終フレームのみ処理を分ける
                j_file['images'].append({'file_name': video_name + '/' + video_name  + '_images/' + video_name +'_' + str(v) + '.jpg',
                'frame_id': v + 1,
                'id': idd,
                'next_image_id': idd + 1,
                'prev_image_id': -1,
                'video_id': i + 1
                })
            elif (v == 599):
                j_file['images'].append({'file_name': video_name + '/' + video_name  + '_images/' + video_name +'_' + str(v) + '.jpg',
                'frame_id': v + 1,
                'id': idd,
                'next_image_id': -1,
                'prev_image_id': idd - 1,
                'video_id': i + 1
                })
            else:
                j_file['images'].append({'file_name': video_name + '/' + video_name  + '_images/' + video_name +'_' + str(v) + '.jpg',
                'frame_id': v + 1,
                'id': idd,
                'next_image_id': idd + 1,
                'prev_image_id': idd - 1,
                'video_id': i + 1
                })

            for key in data['sequence'][v].keys():
                for k in range(len(data['sequence'][v][key])):
                    idd_box += 1
                    j_file['annotations'].append(
                        {'bbox': conv_xywh(data['sequence'][v][key][k]['box2d']),
                        'category_id': ret_id(key),
                        'conf': 1.0,
                        'id': int(idd_box),
                        'image_id': idd,
                        'track_id': data['sequence'][v][key][k]['id']})

    # Export a single file from multiple files
    with open('mot17_signate/annotations/full_train.json', mode = 'w') as fw:
        json.dump(j_file,fw)