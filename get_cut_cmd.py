import argparse
from datetime import datetime, timedelta
import os
import time

import cv2
import numpy as np
import pandas as pd



start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('movie_path')
parser.add_argument('output_dir')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

video = cv2.VideoCapture(args.movie_path)
num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

bkout_frames = []
was_bkout = False

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    is_blackout = np.all(frame[50:600, 50:600, :] == 0)
    
    if is_blackout:
        bkout_frames.append(0)
        was_bkout = True
    else:
        if was_bkout:
            bkout_frames.append(1)
        else:
            bkout_frames.append(0)
        was_bkout = False

video.release()
cv2.destroyAllWindows()

bkout_frames = np.array(bkout_frames)
bkout_time = np.where(bkout_frames==1)[0] / fps

diffs = []
games = []
for i in range(len(bkout_time) - 1):
    diff = bkout_time[i+1] - bkout_time[i]
    if diff > 30:
        games.append(np.array([bkout_time[i], bkout_time[i+1], diff]))

last_extreme = len(bkout_frames) / fps - bkout_time[-1]
if last_extreme > 30 and last_extreme < 400:
    games.append(np.array([bkout_time[-1], len(bkout_frames) / fps, last_extreme]))
        
games = np.array(games)

game_timestamps = []
for game in games:
    mins = np.uint8(game // 60)
    secs = np.uint8(game % 60)
    game_timestamp = ["{:02d}:{:02d}".format(_min, sec) for _min, sec in zip(mins, secs)]
    game_timestamp[2] = game[0]
    game_timestamp.append(game[2])
    game_timestamps.append(game_timestamp)

ts = args.movie_path.split('/')[-1].split('.')[0]
ts = datetime.strptime(ts, '%Y-%m-%d_%H-%M-%S')

for gts in game_timestamps:
    delta = timedelta(minutes=int(gts[0][:2]), seconds=int(gts[0][3:]))
    stts = ts + delta
    stts_str = stts.strftime('%Y-%m-%d_%H-%M-%S')
    gts.append(stts_str)

for gt in game_timestamps:
    cmd = 'ffmpeg -ss {}  -i {} -t {} -vcodec copy -acodec copy {}.mp4'.format(int(round(gt[2])),
                                                                               args.movie_path,
                                                                               int(round(gt[3])),
                                                                               os.path.join(args.output_dir, gt[4]))
    print(cmd)

elapsed_time = time.time() - start_time
print('Elapsed time: {:.1f}min'.format(elapsed_time / 60))
