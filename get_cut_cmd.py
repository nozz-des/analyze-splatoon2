import argparse
from datetime import datetime, timedelta
import os
import pdb
import time

import cv2
import numpy as np
import pandas as pd



start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('movie_path')
parser.add_argument('output_dir')
parser.add_argument('--game_duration_thr', type=int, default=50,
                    help='minimum seconds regard as one game')
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

    is_bkout = np.all(frame[50:600, 50:600, :] == 0)
    
    if is_bkout:
        bkout_frames.append(0)
        was_bkout = True
    else:
        if was_bkout:
            bkout_frames.append(1)
        else:
            bkout_frames.append(0)
        was_bkout = False

video.release()

# get frames after bkout_frames and judge the bunch of frames is a game of not

bkout_frames = np.array(bkout_frames)
bkout_sec = np.where(bkout_frames==1)[0] / fps

games = []
for i in range(len(bkout_sec) - 1):
    diff = bkout_sec[i+1] - bkout_sec[i]
    if diff > args.game_duration_thr:
        games.append([bkout_sec[i], diff])

margin_sec = len(bkout_frames) / fps - bkout_sec[-1]
if margin_sec > args.game_duration_thr and margin_sec < 400:
    games.append([bkout_sec[-1], margin_sec])
        
timestamp = args.movie_path.split('/')[-1].split('.')[0]
timestamp = datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S')

cmds = []
for game in games:
    mins = int(game[0] // 60)
    secs = int(game[0] % 60)
    delta = timedelta(minutes=mins, seconds=secs)
    start_timestamp = timestamp + delta
    start_timestamp = start_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    game.append(start_timestamp)
    
    ss = round(int(game[0]))
    duration = round(int(game[1]))

    output_path = os.path.join(args.output_dir, start_timestamp) + '.mp4'
    
    cmd = 'ffmpeg -ss {}  -i {} -t {} -vcodec copy -acodec copy {}'.format(ss,
                                                                           args.movie_path,
                                                                           duration,
                                                                           output_path)
    cmds.append(cmd)

with open('do.sh', 'w') as f:
    for cmd in cmds:
        f.write('{}\n'.format(cmd))


elapsed_time = time.time() - start_time
print('Elapsed time: {:.1f}min'.format(elapsed_time / 60))

# save as csv
df = pd.DataFrame(games, columns=['start_sec', 'duration', 'start_timestamp'])
df.to_csv(os.path.join(args.output_dir, 'log.csv'))
