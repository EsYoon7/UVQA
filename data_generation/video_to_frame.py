import cv2
import os
import json

from tqdm import tqdm
from multiprocessing import Pool
from moviepy.editor import VideoFileClip

import argparse

# =============================================================================================================
parser = argparse.ArgumentParser(description='Extract frames from videos')
parser.add_argument('--dataset_name', choices=['MOMA-LRG', 'DiDeMo', 'Video-ChatGPT'], required=True, help='Name of the dataset')
parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the folder containing videos')
parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save extracted frames')
args = parser.parse_args()
# =============================================================================================================

# for MOMA
# video_folder = "~/MOMA-LRG/videos/raw"
# output_folder = "~/MOMA-LRG/videos/frames"

# for DiDeMo
# video_folder = "~/DiDeMo/video/YFCC100M_videos"
# output_folder = "~/DiDeMo/frames"

# for Video-ChatGPT
# video_folder = "~/Video-ChatGPT/activitynet_videos"
# video_folder = "~/Video-ChatGPT/all_test"
# output_folder = "~/Video-ChatGPT/frames"

video_folder = args.video_folder
output_folder = args.output_folder

video_list = os.listdir(video_folder)
# video_list = [video_name for video_name in video_list if video_name.endswith('.mp4')]
# video_list = [video_name for video_name in video_list if not video_name.endswith('mpg.mp4') and not video_name.endswith('avi.mp4')]
errored_videos = []
error_messages = []
# =============================================================================================================
# for multiprocessing
def process_video(video_name):
    video_path = os.path.join(video_folder, video_name)
    try:
        clip = VideoFileClip(video_path)
         # Get the total number of frames 
        total_frames = int(clip.fps * clip.duration)

        num_output_frames = 50

        # Calculate the interval between frames to be saved
        frame_interval = max(1, total_frames // num_output_frames)

        # try:
        #     success, image = vidcap.read()
        # except:
        #     print(f'Error reading frame from video: {video_name}')
        #     errored_videos.append(video_name)
        #     error_messages.append(f'Error reading frame from video: {video_name}')
        count = 0
        saved_count = 0

        if args.dataset_name in ['MOMA-LRG']:
            output_folder_video = os.path.join(output_folder, video_name.split('.')[0])
        elif args.dataset_name in ['DiDeMo']:
            output_folder_video = os.path.join(output_folder, '.'.join(video_name.split('.')[:-1])) # since the video name has its extension
        # output_folder_video = os.path.join(output_folder, '.'.join(video_name.split('.')[:-1])) # since the video name has its extension
        elif args.dataset_name in ['Video-ChatGPT']:
            output_folder_video = os.path.join(output_folder, '.'.join(video_name.split('.')[0]))

        if not os.path.exists(output_folder_video):
            os.makedirs(output_folder_video)

        for i, frame in enumerate(clip.iter_frames()):
            if i % frame_interval == 0 and i // frame_interval < num_output_frames:
                frame_filename = os.path.join(output_folder_video, f"{i // frame_interval:06d}.jpg")
                frame_image = frame[:, :, ::-1]  # Convert RGB to BGR for OpenCV compatibility
                cv2.imwrite(frame_filename, frame_image)
        clip.close()

    except:
        print(f'Error processing video: {video_name}')
        errored_videos.append(video_name)
        error_messages.append(f'Error processing video: {video_name}')

    print(f'Finished processing video: {video_name}')

if __name__ == "__main__":    

    # Use a pool of workers to process multiple videos in parallel
    with Pool(processes=os.cpu_count()) as pool:  # Use all available CPU cores
        pool.map(process_video, video_list)

    print("Finished processing all videos.")