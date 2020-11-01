# The 3rd AI Edge Contest
This repository shows the solution by RailStar737A.

## Based environment:
https://github.com/xingyizhou/CenterTrack



## How to get the submitted model

### (1) Create the container (execute the following command at the command prompt)
```
sudo docker run -it --name center_track3 --shm-size=32g -v $(pwd):/opt/ml --gpus all signate/runtime gpu:latest bash
```



### (2) Install the libraries
```
cd opt/ml/centertrack/
pip install -r requirements.txt
```



### (3) Convert video to image data
・Store train_00.mp4 ~ train_24.mp4 in centertrack/preprocess/train_videos, and train_00.json ~ train_24.json in centertrack/preprocess/train_annotations.
・Run videos2images.py, and the image data is divided into the folders under centertrack/preprocess/mot17_signate/train/ 
・Run centertrack/preprocess/convert_json.py, and centertrack/preprocess/mot17_signate/annotations/full_train.json is generated (this becomes annotation data with *ID* of centertrack specification).



### (4) Training
> We interrupted the training at 11epoch. After that, the training was restarted at 11 epochs and after 44 epochs of training, the training was completed and the model was used as the final submission (as a result, 11 + 44 = 55 epochs of training was done). (As a result, 11 + 44 = 55epoch). In the following, the procedure is a little redundant because the training is stopped and restarted at 11epoch in order to reproduce the model of the final submission strictly.

・Move mot17_signate, the set of image data and annotation data generated in (3) and (4) above, to"centertrack/src/CenterTrack/data/".

・Trained model "kitti_fulltrain.pth" from the kitti dataset in the model zoo on github (https://github.com/xingyizhou/CenterTrack/blob/master/readme/MODEL_) Download from ( ZOO.md ) (download link https://drive.google.com/open?id=1kBX4AgQj7R7HvgMdbgBcwvIac-IFp95h)

・Place kitti_fulltrain.pth in centertrack/src/CenterTrack/models/ and Move to centertrack/src/CenterTrack/src/.

```
python main.py tracking --exp_id signate_mot_kitti_finetune --dataset custom --custom_dataset_ann_path ... /data/mot17_signate/annotations/full_train.json --custom_dataset_img_path . /data/mot17_signate/train/ --input_h 1216 --input_w 1216 --num_classes 10 --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --batch_size 4 --load_model ... /models/kitti_fulltrain.pth --lr 1.25e-4
```

・Stop learning when 11epoch is completed on the command prompt.

・Since the training log and the learned data model_last.pth should be saved under centertrack/src/CenterTrack/exp/tracking/signate_mot_kitti_finetune, model_last.pth to model_last_kittift_1216_11epoch.pth and place it in centertrack/src/CenterTrack/models/.

Run the following command at the command prompt (11epoch)

```
python main.py tracking --exp_id signate_mot_kitti_finetune --dataset custom --custom_dataset_ann_path ... /data/mot17_signate/annotations/full_train.json --custom_dataset_img_path . /data/mot17_signate/train/ --input_h 1216 --input_w 1216 --num_classes 10 --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --batch_size 4 --load_model ... /models/model_last_kittift_1216_11epoch.pth --lr 1.25e-4
```

・ Stop the training when 44epoch is completed (actually, 55epoch training is in progress).

・You should find the training log and learned data model_last.pth under centertrack/src/CenterTrack/exp/tracking/signate_mot_kitti_finetune.

## Others

predictor.py is a file to run on the signate runtime.