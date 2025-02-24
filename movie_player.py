#!/usr/bin/env python3

import sys
import cv2
import numpy
import os
from typing import List

from common_ssd import ImageProc


def main(movie_fpath:str, play_fps:float, img_procs:List[ImageProc]):

    # 画像出力用フォルダ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = f"./test_data/{output_imgdir_name}"

    if os.path.isfile(movie_fpath) == False:
        print("Error: ", movie_fpath, " is nothing.")
    else:

        if os.path.isdir(output_imgdir_path) == False:
            os.makedirs(output_imgdir_path)

            
        # 入力動画読み込み
        cap = cv2.VideoCapture(movie_fpath)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fps   = cap.get(cv2.CAP_PROP_FPS)

        frame_play_step = int((cap_fps + 0.1) / play_fps)
        if frame_play_step < 1:
            frame_play_step = 1

        frame_no = 0

        while frame_no < num_frame:

            # frame読み込み
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_img:numpy.ndarray = None
            (_, frame_img) = cap.read()

            if frame_no % frame_play_step == 0:

                if not (frame_img is None):
                    
                    for img_proc in img_procs:
                        frame_img_cliped = img_proc.clip(frame_img)

                        # 保存
                        frame_img_fpath = f"{output_imgdir_path}/{output_imgdir_name}_{str(img_proc)}_F{frame_no:05}.jpg" 
                        cv2.imwrite(frame_img_fpath, frame_img_cliped)

                        print(f"Image saved: {frame_img_fpath}")

                        # ImageProc.draw_text(frame_img, "F{:05}".format(frame_no), (0, 30), 0.75, (255,255,255), 2, True)
                        
                #         # 表示
                #         cv2.imshow(output_imgdir_name, frame_img_cliped)

            # key:int = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break           

    return


if __name__ == "__main__":
    args = sys.argv

    # 切り出し領域
    img_procs = [ImageProc(180, 150, 530, 500), 
                 ImageProc(580, 200, 880, 500), 
                 ImageProc(930, 250, 1280, 600)]

    # 切り出ししない場合は以下を有効化
    # img_procs = [ImageProc()] 


    play_fps = -1.0

    if len(args) < 2:
        print("Usage: ", args[0], " [movie file path] ([play fps])")
    else:
        if len(args) >= 3:
            play_fps = float(args[2])

        main(args[1], play_fps, img_procs)
