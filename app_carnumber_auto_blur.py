#!/usr/bin/env python3

import sys
import cv2
import os
import time
from typing import List,Tuple,Any

import numpy as np
import torch

from common_ssd import ImageProc, DetResult, DrawPen, Logger
from predict_ssd import SSDModelDetector


def main_blur_movie(img_procs:List[ImageProc], ssd_model:SSDModelDetector, movie_fpath:str, play_fps:float, conf:float, overlap:float):

    # 画像出力用フォルダ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0] + ".blur"
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, output_imgdir_name)

    # 入力動画読み込み
    cap       = cv2.VideoCapture(movie_fpath)  
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fps   = cap.get(cv2.CAP_PROP_FPS)
    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_play_step = int((cap_fps + 0.1) / play_fps)
    if frame_play_step < 1:
        frame_play_step = 1

    if play_fps < 0.0:
        play_fps = cap_fps


    fourcc    = cv2.VideoWriter_fourcc("m","p","4","v")
    out_movie = cv2.VideoWriter(f"{output_imgdir_path}/{output_imgdir_name}.mp4", fourcc, play_fps, (frame_w, frame_h))

    # 動画再生
    frame_no = 0

    while frame_no < num_frame:

        # 画像読み込み
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        img_org:np.ndarray = None
        (_, img_org) = cap.read()

        if frame_no % frame_play_step == 0:

            if not (img_org is None):
                
                time_s = time.perf_counter()

                # SSD物体検出
                det_results = ssd_model.predict(img_procs, img_org, conf, overlap)

                # 検出位置にぼかしを入れる
                for det in det_results:
                    if det.class_name_ == "number":
                        s_roi = img_org[det.bbox_[1]: det.bbox_[3], det.bbox_[0]: det.bbox_[2]]
                        s_roi = cv2.blur(s_roi, (10, 10)) # ぼかし処理
                        img_org[det.bbox_[1]: det.bbox_[3], det.bbox_[0]: det.bbox_[2]] = s_roi


                time_e = time.perf_counter()

                # FPS等を描画
                img_org = img_procs[0].drawResultSummary(img_org, frame_no, num_frame, 
                                                     ssd_model.device_.type, 
                                                     (time_e - time_s),
                                                     DrawPen((255,255,255), 2, 0.6))

                # 保存
                frame_img_fpath = output_imgdir_path + "/F{:05}".format(frame_no) + ".jpg" 
                cv2.imwrite(frame_img_fpath, img_org)

                # 動画出力
                out_movie.write(img_org)
                
                # 表示
                cv2.imshow(output_imgdir_name, img_org)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("output: ", f"{output_imgdir_path}/{output_imgdir_name}.mp4")

    out_movie.release()

    return


def main(media_fpath:str, play_fps:float, weight_fpath:str, img_procs:List[ImageProc], conf:float, overlap:float):

    if (os.path.isfile(media_fpath) == False) and (os.path.isdir(media_fpath) == False):
        print("Error: ", media_fpath, " is nothing.")

    else:
        # SSDモデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print("使用デバイス：", device)

        ssd_model = SSDModelDetector(device, weight_fpath)

        media_fname:str = os.path.basename(media_fpath)
        if ".mp4" in media_fname:
            # ナンバープレートぼかし実行
            main_blur_movie(img_procs,ssd_model, media_fpath, play_fps, conf, overlap)

        else:
            print("No support ext [", media_fname, "]")
    
    return

if __name__ == "__main__":
    args = sys.argv

    weight_fpath = "./weights/ssd_best_od_cars.pth"

    # 検出範囲
    #   (1280x720を)300x300/350x350に切り出し
    # img_procs = [ImageProc(180, 250, 530, 600), 
    #              ImageProc(480, 200, 780, 500), 
    #              ImageProc(730, 200, 1030, 500), 
    #              ImageProc(930, 250, 1280, 600)] 
    img_procs = [ImageProc(480, 200, 780, 500), 
                 ImageProc(730, 200, 1030, 500), 
                 ImageProc(930, 250, 1280, 600)] 

    # 入力画像全域を検出範囲にする場合は以下を有効化
    # img_procs = [ImageProc()] 

    conf     = 0.5
    overlap  = 0.2
    play_fps = -1.0

    if len(args) < 2:
        print("Usage: ", args[0], " [movie file path] ([play fps])")
    else:
        if len(args) >= 3:
            play_fps = float(args[2])

        main(args[1], play_fps, weight_fpath, img_procs, conf, overlap)
