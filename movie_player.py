#!/usr/bin/env python3

import sys
import cv2
import os
import copy
from typing import List,Dict,Any
from tqdm import tqdm
import torch

from common_ssd import ImageProc, MovieLoader
from predict_ssd import SSDModelDetector

def mainExtractFixArea(movie_fpath:str, cfg:Dict[str,Any]):
    """ 固定領域を切り出し

    Args:
        movie_fpath (str)           : 入力動画パス
        cfg (Dict[str,Any])         : config
    """
    img_procs:List[ImageProc] = cfg["img_procs_fix"] 
    play_fps:float            = cfg["play_fps"]

    # 画像出力用ディレクトリ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = f"./output.data/{output_imgdir_name}"
    if os.path.isdir(output_imgdir_path) == False:
        os.makedirs(output_imgdir_path)

    # 入力動画読み込み
    movie_loader = MovieLoader(movie_fpath, play_fps, 1)
    num_frame    = movie_loader.getNumFrame()
    play_fps     = movie_loader.getPlayFps()

    with tqdm(movie_loader) as movie_iter:
        for batch_frame_nos, batch_imgs in movie_iter:

            if len(batch_imgs) > 0:
                for frame_no, frame_img in zip(batch_frame_nos, batch_imgs):
                    movie_iter.set_description(f"[{frame_no}/{num_frame}]") # 進捗表示

                    # 固定領域をカット＆保存
                    for img_proc in img_procs:
                        frame_img_cliped = img_proc.clip(frame_img)

                        # 保存
                        frame_img_fpath = f"{output_imgdir_path}/{output_imgdir_name}_{str(img_proc)}_F{frame_no:05}.jpg" 
                        cv2.imwrite(frame_img_fpath, frame_img_cliped)

                        # 表示
                        ImageProc.drawText(frame_img_cliped, 
                                            f"F{frame_no}/{num_frame} {img_proc}",
                                            (10, 15), 0.4, (255,255,255), 1, True)
                        cv2.imshow(output_imgdir_name, frame_img_cliped)

                        key:int = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

    return

def mainExtractDetArea(movie_fpath:str, cfg:Dict[str,Any]):
    """ 検出結果から切り出し

    Args:
        movie_fpath (str)           : 入力動画パス
        cfg (Dict[str,Any])         : config
    """
    play_fps:float            = cfg["play_fps"]
    net_type:str              = cfg["img_procs_det_net_type"]
    weight_fpath:str          = cfg["img_procs_det_weight_fpath"]
    img_procs:List[ImageProc] = cfg["img_procs_det_area"]
    num_batch:int             = cfg["img_procs_det_num_batch"]
    conf:float                = cfg["img_procs_det_conf_lower_th"]
    overlap:float             = cfg["img_procs_det_iou_th"]
    area_minsize:int          = cfg["img_procs_det_area_minsize"]
    area_class:str            = cfg["img_procs_det_area_class"]

    # SSDモデル作成
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    ssd_model = SSDModelDetector(device, net_type, weight_fpath)

    # 画像出力用ディレクトリ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = f"./output.data/{output_imgdir_name}_det"
    if os.path.isdir(output_imgdir_path) == False:
        os.makedirs(output_imgdir_path)

    # 入力動画読み込み
    num_batch_frame = int(num_batch / len(img_procs))
    if num_batch_frame < 1:
        num_batch_frame = 1

    movie_loader = MovieLoader(movie_fpath, play_fps, num_batch_frame)
    num_frame    = movie_loader.getNumFrame()
    play_fps     = movie_loader.getPlayFps()

    with tqdm(movie_loader) as movie_iter:
        for batch_frame_nos, batch_imgs in movie_iter:
            if len(batch_imgs) > 0:
                # SSD物体検出（複数フレーム分をまとめて検出）
                det_results = ssd_model.predict(img_procs, batch_imgs, conf, overlap)

                # フレーム毎の処理
                for frame_no, frame_img, det_result in zip(batch_frame_nos, batch_imgs, det_results):
                    movie_iter.set_description(f"[{frame_no}/{num_frame}]") # 進捗表示

                    # 検出結果から、切り出し領域を作成
                    clip_areas:List[ImageProc] = []

                    for det_obj in det_result:
                        if det_obj.class_name_ == area_class:
                            # area_classのみを採用
                            img_proc = ImageProc()
                            img_proc.initFromDet(frame_img, det_obj, area_minsize)
                            if img_proc.is_no_proc_ == False:
                                # サイズがarea_minsize以上の領域のみ採用
                                clip_areas.append(copy.deepcopy(img_proc))

                    # 検出領域をカット＆保存
                    for area_no, clip_area in enumerate(clip_areas):
                        frame_img_cliped = clip_area.clip(frame_img)

                        # 保存
                        frame_img_fpath = f"{output_imgdir_path}/{output_imgdir_name}_F{frame_no:05}_{area_class}{area_no}.jpg" 
                        cv2.imwrite(frame_img_fpath, frame_img_cliped)

                        # 表示
                        ImageProc.drawText(frame_img_cliped, 
                                            f"F{frame_no}/{num_frame}",
                                            (10, 15), 0.4, (255,255,255), 1, True)
                        cv2.imshow(output_imgdir_name, frame_img_cliped)

                        key:int = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

    return

if __name__ == "__main__":
    args = sys.argv

    cfg = {
        # 動画再生fps (負値＝入力動画のfpsそのまま)
        "play_fps"      : -1.0,

        # ※"img_procs_fix", "img_procs_det_xxx"は、どちらかのみ有効化
        #   （両方有効化するとimg_procs_fixで実行）

        # 切り出し領域＝固定
        # "img_procs_fix"     : [ImageProc(180, 150, 530, 500), 
        #                        ImageProc(580, 200, 880, 500), 
        #                        ImageProc(930, 250, 1280, 600)],
        # 切り出し領域＝固定（切り出ししない場合）
        # "img_procs_fix"   : [ImageProc()],

        # 切り出し領域＝検出結果
        "img_procs_det_area"          : [ImageProc(0, 250, 350, 600), 
                                         ImageProc(250, 200, 550, 500), 
                                         ImageProc(480, 200, 780, 500), 
                                         ImageProc(730, 200, 1030, 500), 
                                         ImageProc(930, 250, 1280, 600)],
        "img_procs_det_net_type"      : "mb2-ssd",
        "img_procs_det_weight_fpath"  : "./weights/mb2-ssd_best_od_cars.pth", 
        "img_procs_det_num_batch"     : 64,
        "img_procs_det_area_minsize"  : 100,
        "img_procs_det_area_class"    : "car",
        "img_procs_det_conf_lower_th" : 0.5,
        "img_procs_det_iou_th"        : 0.4,
    }

    if len(args) < 2:
        print(f"Usage: {args[0]} [movie file path] ([play fps])")
    else:
        if len(args) >= 3:
            cfg["play_fps"] = float(args[2])

        movie_fpath = args[1]

        if os.path.isfile(movie_fpath) == True:

            if "img_procs_fix" in cfg.keys():
                mainExtractFixArea(movie_fpath, cfg)
            elif "img_procs_det_area" in cfg.keys():
                mainExtractDetArea(movie_fpath, cfg)
            else:
                print("Error: config invalid!!")

        else:
            print(f"Error: {movie_fpath} is nothing.")
