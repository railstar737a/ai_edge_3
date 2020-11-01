import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    #define paths
    path_to_videos    = 'train_videos'
    path_out  = 'mot17_signate/train/'

    #browse all videos, decode them, and save into folders
    list_of_videos =[os.path.join(root, name) for root, dirs, files in os.walk(path_to_videos) for name in files]

    for v in tqdm(range (0, len(list_of_videos))):
        video_name = list_of_videos[v].split('/')[-1].split('\\')[-1].split('.')[0]
        
        try:
            os.makedirs(path_out + video_name + '/' + video_name + '_images',  exist_ok=True)
        except OSError:
            print ('cannot create directory')
            exit()

        stream = cv2.VideoCapture(list_of_videos[v])
        for i in range (0, 10000):
            (grabbed, frame) = stream.read()
            if not grabbed: break

            cv2.imwrite(path_out + video_name + '/' + video_name + '_images' + '/' +video_name + '_' + str(i) +'.jpg', frame)