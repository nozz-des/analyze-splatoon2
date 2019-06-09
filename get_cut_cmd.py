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
args = parser.parse_args()

game_duration_min = 50
game_duration_max = 540

os.makedirs(args.output_dir, exist_ok=True)

video = cv2.VideoCapture(args.movie_path)
num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

# 1. Detect blackout frames

bkout_frames = []
was_bkout = False
frame_count = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # judge from a part of each frame
    is_bkout = np.all(frame[50:600, 50:600, :] == 0)
    
    if is_bkout:
        was_bkout = True
    else:
        if was_bkout:
            bkout_frames.append(frame_count)
        was_bkout = False

    frame_count += 1

bkout_frames = np.array(bkout_frames)
bkout_sec = bkout_frames / fps


# 2. Filter by duration seconds (50-540)
games = []
for i in range(len(bkout_sec) - 1):
    duration = bkout_sec[i+1] - bkout_sec[i]
    if duration > game_duration_min and duration < game_duration_max:
        games.append([bkout_frames[i], bkout_sec[i], duration])

last_duration = (len(bkout_frames) / fps) - bkout_sec[-1]
if last_duration > game_duration_min and last_duration < game_duration_max:
    games.append([bkout_frames[i], bkout_sec[-1], last_duration])


# 3. Detect whiteout frame after each blackout frame
detected_games = []
for game in games:
    is_game = False
    start_frame = game[0]
    for i in np.arange(150, 200, 1):
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ret, frame = video.read()
        
        if frame is None:
            break
        
        num_bright_px = np.count_nonzero(frame > 250)
        ratio_bright_px = (num_bright_px / np.size(frame)) * 100
        
        if ratio_bright_px > 60:
            is_game = True

    if is_game:
        detected_games.append(game)


# 4. Make timestamp of each games and commands to cut movie
timestamp = args.movie_path.split('/')[-1].split('.')[0]
timestamp = datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S')

cmds = []
for game in detected_games:
    mins = int(game[0] // 60)
    secs = int(game[0] % 60)
    delta = timedelta(minutes=mins, seconds=secs)
    start_timestamp = timestamp + delta
    start_timestamp = start_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    game.append(start_timestamp)
    
    ss = round(int(game[1]))
    duration = round(int(game[2]))

    output_path = os.path.join(args.output_dir, start_timestamp) + '.mp4'
    
    cmd = 'ffmpeg -ss {}  -i {} -t {} -vcodec copy -acodec copy {}'.format(ss,
                                                                           args.movie_path,
                                                                           duration,
                                                                           output_path)
    cmds.append(cmd)

with open('do.sh', 'w') as f:
    for cmd in cmds:
        f.write('{}\n'.format(cmd))

df = pd.DataFrame(games, columns=['start_frame', 'start_sec', 'duration', 'start_timestamp'])
df.to_csv(os.path.join(args.output_dir, 'log.csv'))

elapsed_time = time.time() - start_time
print('Elapsed time: {:.1f}min'.format(elapsed_time / 60))

video.release()
