#!/usr/bin/env python3

import sys
import cv2
import os
import copy
import math
import time
from typing import List,Dict,Any

import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

import torch

from utils.data_augumentation import jaccard_numpy
from common_ssd import ImageProc, MovieLoader, DetResult, DrawPen, Logger, KalmanFilter2D
from common_ssd import dumpDetResultsToCsvLine
from predict_ssd import SSDModelDetector

class DetNumberPlate:
    """ 検出されたナンバープレート1つ分

    （ナンバープレートと、ナンバープレートを包含する車のペアを保持）
    """

    def __init__(self, cfg:Dict[str,Any], frame_w:int, frame_h:int, fps:float):
        """ コンストラクタ

        Args:
            cfg (Dict[str,Any]) : config
            frame_w (int)       : 画像サイズ（width）[px]
            frame_h (int)       : 画像サイズ（height）[px]
            fps (float)         : 再生fps [cycle/sec]
        """

        self.id_         = 0
        self.is_valid_   = False
        self.frame_w_    = frame_w
        self.frame_h_    = frame_h
        self.accum_conf_ = 0.0   # 累積信頼度（検出した信頼度confを積算）

        self.obj_number_:DetResult            = None
        self.obj_numbers_sub_:List[DetResult] = [] # ナンバープレート誤検出対策：誤検出のほうが信頼度が高い場合に正検出のぼかしが解けてしまう現象の対策用
        self.obj_car_:DetResult               = None

        # 位置推定用カルマンフィルタ
        #   推定対象：ナンバープレート外接矩形の中心
        self.pos_estimator_ = KalmanFilter2D(fps)

        self.param_aconf_max_:float = cfg["ACCUM_CONF_MAX"]
        return
    
    def isValid(self) -> bool:
        if (self.obj_number_ is None) or (self.obj_car_ is None):
            self.is_valid_ = False
        return self.is_valid_
    
    def resetDetCurFlag(self):
        # 今周期検出済みフラグをOFF
        if self.isValid() == True:
            self.obj_number_.is_det_cur_ = False
            self.obj_car_.is_det_cur_    = False

            # ナンバープレート誤検出対策用の物体は、次周期に持ち越さない
            self.obj_numbers_sub_.clear()
        return
    
    def __eq__(self, other) -> bool:
        ret = False
        if (self.isValid() == True) and (other.isValid() == True):
            if self.id_ == other.id_:
                ret = True
        return ret

    def updateDetObject(self, obj_number:DetResult, obj_car:DetResult):
        """ 検出物体の更新（引数で指定された物体に更新）

        Args:
            obj_number (DetResult)  : ナンバープレート
            obj_car (DetResult)     : 車
        """
        if (obj_number is not None) and (obj_car is not None):
            self.is_valid_ = True

            if (self.isValid() == True) and (self.obj_number_.is_det_cur_ == True):
                # [今周期登録された物体の場合] ナンバープレートの信頼度が高ければ更新
                if self.obj_number_.score_ < obj_number.score_:
                    self.obj_number_ = copy.deepcopy(obj_number)
                    self.obj_car_    = copy.deepcopy(obj_car)
                else:
                    # ナンバープレート誤検出対策：信頼度低いほうもとっておき、ぼかしの対象に加える（ただし、時系列処理の対象外）
                    self.obj_numbers_sub_.append(obj_number)

            else:
                # [過去周期に登録された物体の場合] 更新
                self.obj_number_ = copy.deepcopy(obj_number)
                self.obj_car_    = copy.deepcopy(obj_car)

        return

    def updateDetCarObject(self, obj_car:DetResult):
        """ 検出物体（車のみ）の更新（引数で指定された物体に更新）

        Args:
            obj_car (DetResult): 車
        """        
        if obj_car is not None:
            self.is_valid_ = True

            if (self.isValid() == True) and (self.obj_car_.is_det_cur_ == True):
                # [今周期登録された車の場合] 信頼度が高ければ更新
                if self.obj_car_.score_ < obj_car.score_:
                    self.obj_car_ = copy.deepcopy(obj_car)
            else:
                # [過去周期に登録された車の場合] 更新
                self.obj_car_ = copy.deepcopy(obj_car)
        return

    def setDetObject(self, id:int, obj_number:DetResult, obj_car:DetResult):
        """ 検出物体の設定（引数で指定された物体に無条件で更新）

        Args:
            id (int)                : 物体ID ※現状、未参照
            obj_number (DetResult)  : ナンバープレート
            obj_car (DetResult)     : 車
        """        
        if (obj_number is not None) and (obj_car is not None):
            self.id_ = id
            self.is_valid_   = True
            self.obj_number_ = copy.deepcopy(obj_number)
            self.obj_car_    = copy.deepcopy(obj_car)
        return

    def updateAccumConf(self, conf_val:float):
        """ 信頼度の累積

        Args:
            conf_val (float): 信頼度conf
        """        
        # 累積信頼度の更新
        self.accum_conf_ += conf_val

        # 上限でクリップ
        if self.accum_conf_ > self.param_aconf_max_:
            self.accum_conf_ = self.param_aconf_max_

        if self.accum_conf_ < 0.0:
            # [下限（＝0）を下回った場合] 無効化
            self.is_valid_ = False

        return

    def updatePos(self):
        """ 位置更新（ナンバープレート外接矩形の中心座標の推定実行）
        """        
        if self.isValid() == True:

            self.pos_estimator_.predict() # 予測ステップ

            if self.obj_number_.is_det_cur_ == True:
                # [ナンバープレート検出時（観測値が得られた場合）] 
                measure = self.obj_number_.getBboxCenter().astype(float)
                self.pos_estimator_.update(measure) # 更新ステップ

                # 一般的ではないが、今回の場合、未検出のときだけカルマンFで外挿をかけたいので、
                # 観測値が得られた場合は推定値を観測値でリセット（位置成分のみ）
                self.pos_estimator_.resetPredict(measure)
        return

    def getEstimatedNumberBBox(self) -> np.ndarray:
        """ ナンバープレートの外接矩形（推定値）を返す

        Returns:
            np.ndarray: ナンバープレートの外接矩形（推定値）
        """        
        ret_bbox = np.zeros((4,)).astype(int)

        if self.isValid() == True:
            # カルマンフィルタ推定値（外接矩形の中点位置）を取得
            est_pt_center = self.pos_estimator_.getEstimatedValue()

            # 推定値（中点位置）のばらつきを取得
            #   3σ（小数切り上げ）
            est_3sigma    = np.ceil(self.pos_estimator_.getEstimatedStdev() * np.array([3.0, 3.0])) 

            # 推定値（中点位置）のばらつき（3σ）分だけ、外接矩形の幅／高さを増やす
            # （極端に大きくならないよう（増加量が元の幅／高さ以上にならないよう）クリップ）
            bbox_w_inc = int(est_3sigma[0] * 2.0) # 2.0：左右
            bbox_h_inc = int(est_3sigma[1] * 2.0) # 2.0：上下
            if bbox_w_inc > self.obj_number_.bbox_w_:
                bbox_w_inc = self.obj_number_.bbox_w_
            if bbox_h_inc > self.obj_number_.bbox_h_:
                bbox_h_inc = self.obj_number_.bbox_h_
            
            bbox_w = self.obj_number_.bbox_w_ + bbox_w_inc
            bbox_h = self.obj_number_.bbox_h_ + bbox_h_inc

            # 中点→左上座標に変換
            est_pt_leftup = (est_pt_center - np.array([float(bbox_w) / 2.0, float(bbox_h) / 2.0])).astype(int)

            # 画像の幅、高さでクリップ
            left_up_max_x = self.frame_w_ - 1 - bbox_w
            left_up_max_y = self.frame_h_ - 1 - bbox_h

            if est_pt_leftup[0] > left_up_max_x:
                est_pt_leftup[0] = left_up_max_x
            if est_pt_leftup[1] > left_up_max_y:
                est_pt_leftup[1] = left_up_max_y

            # 0でクリップ
            if est_pt_leftup[0] < 0:
                est_pt_leftup[0] = 0
            if est_pt_leftup[1] < 0:
                est_pt_leftup[1] = 0

            ret_bbox[0] = est_pt_leftup[0]
            ret_bbox[1] = est_pt_leftup[1]
            ret_bbox[2] = ret_bbox[0] + bbox_w
            ret_bbox[3] = ret_bbox[1] + bbox_h

        return ret_bbox
    
    @staticmethod
    def searchOwnerCar(obj_number:DetResult, det_objs:List[DetResult], own_car_rate_th:float) -> DetResult:
        """ ナンバープレート（obj_number）を所有（包含）する車の検出結果を探索

        - 「重なり矩形面積／ナンバープレートの面積」が閾値以上の場合に、「所有する（包含する）」と判定
        - 複数見つかった場合は、信頼度が一番大きな車を返す
        
        Args:
            obj_number (DetResult)      : ナンバープレート
            det_objs (List[DetResult])  : 検出結果（集合）
            own_car_rate_th (float)     : 判定閾値（「重なり矩形面積／ナンバープレートの面積」の閾値）

        Returns:
            DetResult: ナンバープレート（obj_number）を所有（包含）する車
        """        
        obj_car:DetResult = None
        
        bbox_n_area = float(obj_number.bbox_area_)

        if bbox_n_area > 0.0:

            for det_obj in det_objs:

                if det_obj.class_name_ == "car":

                    # 重なりを算出
                    own_rate = DetResult.calcOverlapAreaBBox(obj_number.bbox_, det_obj.bbox_) / bbox_n_area

                    if own_rate > own_car_rate_th:
                        # [車（det_obj）がナンバープレート(obj_number)を所有（包含）する、と判定された場合]

                        if obj_car is not None:
                            # [既に１つ以上車が見つかっている場合] 信頼度が大きい車を選択
                            if det_obj.score_ > obj_car.score_:
                                obj_car = copy.deepcopy(det_obj)
                        else:
                            # [まだ車が見つかっていない場合] det_objを選択
                            obj_car = copy.deepcopy(det_obj)

        return obj_car

class DetNumberPlateMng:
    """　検出されたナンバープレートを時系列管理
    """    

    def __init__(self, cfg:Dict[str,Any], frame_w:int, frame_h:int, fps:float):
        """ コンストラクタ

        Args:
            cfg (Dict[str,Any]) : config
            frame_w (int)       : 画像サイズ（width）[px]
            frame_h (int)       : 画像サイズ（height）[px]
            fps (float)         : 再生fps [cycle/sec]
        """        
        self.cfg_     = cfg
        self.frame_w_ = frame_w
        self.frame_h_ = frame_h
        self.fps_     = fps
        self.new_id_  = 1
        self.det_obj_buf_:List[DetNumberPlate] = [] # 検出された物体（ナンバープレート、車）
        return

    def initCycle(self):
        """　周期処理開始時の初期化（今周期検出済フラグをリセット）
        """
        for det_obj in self.det_obj_buf_:
            det_obj.resetDetCurFlag()
        return

    def addCurDetNumber(self, det_results:List[DetResult], same_cur_iou_th=0.5, own_car_rate_th=0.5):
        """ 今周期の検出結果を登録

        Args:
            det_results (List[DetResult])    : 検出結果
            same_cur_iou_th (float, optional): 同一の車と判定するIoU閾値. Defaults to 0.5.
            own_car_rate_th (float, optional): 判定閾値（「重なり矩形面積／ナンバープレートの面積」の閾値）. Defaults to 0.5.
        """        
        # 登録済みの車の外接矩形bboxを抽出
        regist_cars_bbox = [obj_reg.obj_car_.bbox_ 
                            # isValid()==Falseの場合、重なりなしと判定される外接矩形を入れておく
                            if obj_reg.isValid() == True else np.array([-1,-1,-1,-1])
                            for obj_reg in self.det_obj_buf_]
        
        # 登録済みの過去周期の車を抽出
        regist_past_cars = [obj_reg.obj_car_ 
                            for obj_reg in self.det_obj_buf_ 
                            if (obj_reg.isValid() == True) and (obj_reg.obj_car_.is_det_cur_ == False)]

        # 今周期検出された車の外接矩形bboxを抽出
        det_cars_bbox = [obj_det_car.bbox_
                         # 車ではない場合、重なりなしと判定される外接矩形を入れておく
                         if obj_det_car.class_name_ == "car" else np.array([-1,-1,-1,-1])
                         for obj_det_car in det_results]

        # 今周期の検出されたナンバープレートと、ナンバープレートを所有する車のペアを登録する
        #   - 既に登録されているナンバープレートと同一の場合は、差し替え
        #   - 同一ものがない場合は、新規登録
        for obj_det in det_results:
            if obj_det.class_name_ == "number":
                # ナンバープレートを所有する車のペアを探索
                obj_number = obj_det
                obj_car    = DetNumberPlate.searchOwnerCar(obj_number, det_results, own_car_rate_th)

                if obj_car is None:
                    # [所有する車が、今周期検出分から見つからなかった場合] 登録済みの過去周期の車から探す
                    obj_car = DetNumberPlate.searchOwnerCar(obj_number, regist_past_cars, own_car_rate_th)

                if obj_car is not None:
                    # [obj_car(車)が、obj_number(ナンバープレート)を所有している場合]

                    # 今回のobj_carと同一のものが、登録済みかどうかを確認（トラッキング）
                    buf_idx = self.trackingCar(obj_car, same_cur_iou_th, regist_cars_bbox)
                    if buf_idx >= 0:
                        # [登録されている場合] 差し替え
                        self.det_obj_buf_[buf_idx].updateDetObject(obj_number, obj_car)

                    else:
                        # [未登録の場合] 新規追加
                        new_det_obj = DetNumberPlate(self.cfg_, self.frame_w_, self.frame_h_, self.fps_)
                        new_det_obj.setDetObject(self.new_id_, obj_number, obj_car)
                        self.det_obj_buf_.append(new_det_obj)
                        self.new_id_ = self.new_id_ + 1

        # 今周期未検出のナンバープレートを所有する車を、今周期検出分に更新
        for obj_reg in self.det_obj_buf_:
            if (obj_reg.isValid() == True) and (obj_reg.obj_number_.is_det_cur_ == False):
                # [obj_regが今周期ナンバープレート未検出]

                # obj_regの車と同一のものが、今周期検出物体に入っているかどうかを確認（トラッキング）
                buf_idx = self.trackingCar(obj_reg.obj_car_, same_cur_iou_th, det_cars_bbox)
                if buf_idx >= 0:
                    # [今周期検出物体に入っている場合] 車を今周期検出分に差し替え
                    obj_reg.updateDetCarObject(det_results[buf_idx])

        return

    def trackingCar(self, obj_car:DetResult, same_cur_iou_th:float, obj_cars_bbox:List[np.ndarray]) -> int:
        """ 今周期の検出結果（車）が、登録済み（前周期）の検出結果に存在するかどうかを確認（トラッキング）

        - もし存在すれば、indexを返す
        - 複数存在する場合は、最も重なりの大きな車のindexを返す
        - ない場合は、-1を返す
        - 同一かどうかは、車同士の外接矩形の重なり(iou)＞閾値(same_cur_iou_th) で判定

        Args:
            obj_car (DetResult): 車
            same_cur_iou_th (float): 同一の車と判定するIoU閾値
            obj_cars_bbox (List[np.ndarray]): 登録済み（前周期）の検出結果（車の外接矩形（Boundingbox））

        Returns:
            int: 該当する車（検出結果のindex）
        """        
        ret_idx = -1

        if len(obj_cars_bbox) > 0:
            # 重なり(iou)を算出（全登録分の車との重なりを一括で算出）
            obj_car_ious = jaccard_numpy(np.array(obj_cars_bbox), obj_car.bbox_)

            # 外接矩形の重なり(iou)が閾値以上を抽出
            # 閾値以上が複数存在する場合は、重なり(iou)最大のものを抽出
            iou_max = same_cur_iou_th

            for buf_idx, obj_car_iou in enumerate(obj_car_ious):
                if obj_car_iou > iou_max:
                    ret_idx = buf_idx
                    iou_max = obj_car_iou

        return ret_idx

    def updateCycle(self):
        """ 個々の物体（ナンバープレート＆車）の時系列管理

        - 今周期にナンバープレート検出ありの場合は、累積信頼度を加算
        - 今周期にナンバープレート検出なしの場合は、累積信頼度を減算
        - 累積信頼度が高いものを残す（低くなった物体（検出されなくなったもの）を削除）
        """
        obj_reg_buf_updated: List[DetNumberPlate] = []

        for obj_reg in self.det_obj_buf_:

            if obj_reg.is_valid_ == True:

                if obj_reg.obj_number_.is_det_cur_ == True:
                    # [今周期にナンバープレート検出ありの場合] 
                    #  累積信頼度を加算。そのまま残す
                    obj_reg.updateAccumConf(obj_reg.obj_number_.score_)
                    obj_reg_buf_updated.append(obj_reg)

                else:
                    # [今周期にナンバープレート検出なしの場合] 
                    #  累積信頼度を減算。低くなったら無効化
                    dec_conf = -1.0
                    if obj_reg.obj_car_.is_det_cur_ == True:
                        # 所有する車の信頼度が高いほど、減少がゆるやかになるようにする。車が未検出の場合は減少を加速（-1.0）
                        dec_conf =  obj_reg.obj_car_.score_ - 1.1 # ※1.1: 減少量を0にはしない

                    obj_reg.updateAccumConf(dec_conf)

                    if obj_reg.isValid() == True:
                        #  [無効化されていない場合] 残す
                        obj_reg_buf_updated.append(obj_reg)

        self.det_obj_buf_ = obj_reg_buf_updated

        # 登録物体（残すと決まったもののみ）の位置推定
        for obj_reg in self.det_obj_buf_:
            obj_reg.updatePos()

        return

    def getNumberPlates(self) -> List[DetResult]:
        """ 登録されているナンバープレートを返す

        Returns:
            List[DetResult]: 登録されているナンバープレート
        """        
        ret_objs:List[DetResult] = []

        for det_obj in self.det_obj_buf_:
            if det_obj.isValid() == True:
                det_obj_new = copy.deepcopy(det_obj)

                # ナンバープレートの信頼度を、累積信頼度に書き換え
                det_obj_new.obj_number_.score_ = det_obj_new.accum_conf_

                # 今周期未検出だったナンバープレートの外接矩形は、過去周期結果から位置を推定した結果に書き換える
                if det_obj_new.obj_number_.is_det_cur_ == False:
                    det_obj_new.obj_number_.bbox_ = det_obj_new.getEstimatedNumberBBox()

                if DetResult.calcOverlapAreaBBox(det_obj_new.obj_number_.bbox_, det_obj.obj_car_.bbox_) > 0.0:
                    # ナンバープレートが車に含まれる場合のみ追加（位置推定で車の領域から外れた場合は除外）
                    ret_objs.append(det_obj_new.obj_number_)

                    # ナンバープレート誤検出対策用の物体（車の包含有無は確認済み）も、もしあれば加える
                    if len(det_obj_new.obj_numbers_sub_) > 0:
                        for obj_number_sub in det_obj_new.obj_numbers_sub_:
                            ret_objs.append(obj_number_sub)

        return ret_objs

    def getCars(self) -> List[DetResult]:
        """ 登録されている車を返す

        Returns:
            List[DetResult]: 登録されている車
        """        
        ret_objs = [det_obj.obj_car_ 
                    for det_obj in self.det_obj_buf_ 
                    if det_obj.isValid() == True]
        return ret_objs

def main_blur_movie(movie_fpath:str, ssd_model:SSDModelDetector, cfg:Dict[str,Any]):
    """ ナンバープレートぼかし処理メイン

    Args:
        movie_fpath (str)           : 入力動画パス
        ssd_model (SSDModelDetector): SSDモデル
        cfg (Dict[str,Any])         : config
    """
    play_fps:float                  = cfg["play_fps"]
    img_procs:List[ImageProc]       = cfg["img_procs"]
    img_erase_rects:List[ImageProc] = cfg["img_erase"]

    conf:float            = cfg["ssd_model_conf_lower_th"]
    overlap:float         = cfg["ssd_model_iou_th"]
    is_det_detail:bool    = cfg["ssd_model_is_det_detail"]
    detail_minsize:int    = cfg["ssd_model_detail_minsize"]
    blur_kernel_size:int  = cfg["blur_kernel_size"]
    is_blur:bool          = cfg["is_blur"]
    is_debug:bool         = cfg["is_debug"]
    is_output_movie:bool  = cfg["is_output_movie"]
    is_output_image:bool  = cfg["is_output_image"]
    is_disp_debug:bool    = cfg["is_disp_debug"]
    is_output_debug:bool  = cfg["is_output_debug"]
    same_cur_iou_th:float = cfg["same_cur_iou_th"]
    own_car_rate_th:float = cfg["own_car_rate_th"]
    num_batch:int         = cfg["ssd_model_num_batch"]

    num_batch_frame = int(num_batch / len(img_procs))
    if num_batch_frame < 1:
        num_batch_frame = 1

    num_batch_real = num_batch_frame * len(img_procs)

    # 画像出力用フォルダ作成
    movie_fname_base = os.path.splitext(os.path.basename(movie_fpath))[0]
    if (is_debug == True) and (is_output_debug == True):
        output_imgdir_name = f"{movie_fname_base}.dbg"
    else:
        output_imgdir_name = f"{movie_fname_base}.blur"
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, ssd_model.net_type_, output_imgdir_name)

    # 入力動画読み込み
    movie_loader       = MovieLoader(movie_fpath, play_fps, num_batch_frame)
    num_frame          = movie_loader.getNumFrame()
    play_fps           = movie_loader.getPlayFps()
    (frame_w, frame_h) = movie_loader.getFrameSize()

    det_numbers_mng = DetNumberPlateMng(cfg, frame_w, frame_h, play_fps)

    print(f"play_fps = {play_fps}")

    # 出力用動画を作成
    output_movie_fpath = f"{output_imgdir_path}/{output_imgdir_name}.mp4"
    out_movie:cv2.VideoWriter = None

    if is_output_movie == True:
        fourcc    = cv2.VideoWriter_fourcc("m","p","4","v")
        out_movie = cv2.VideoWriter(output_movie_fpath, fourcc, play_fps, (frame_w, frame_h))

    # 検出結果出力csvを作成
    output_detcsv_fpath = f"{output_imgdir_path}/{movie_fname_base}.csv"
    detcsv_fp = open(output_detcsv_fpath,"w")

    # 入力動画のフレーム読み込み ＆ 検出実行
    # 結果を出力動画に書き込み
    with tqdm(movie_loader) as movie_iter:
        for batch_frame_nos, batch_imgs in movie_iter:

            if len(batch_imgs) > 0:

                # SSD物体検出（複数フレーム分をまとめて検出）
                time_s = time.perf_counter()

                det_results = ssd_model.predict(img_procs, batch_imgs, conf, overlap)

                # 車の枠からナンバープレートを再検出
                if is_det_detail == True:
                    det_results = ssd_model.predictDetail(det_results, batch_imgs, detail_minsize, num_batch_real, "car", conf, overlap)

                time_e = time.perf_counter()

                time_per_batch = (time_e - time_s) / len(batch_imgs)

                # フレーム毎の処理
                for batch_frame_no, img_org, det_result in zip(batch_frame_nos, batch_imgs, det_results):

                    movie_iter.set_description(f"[{batch_frame_no}/{num_frame}]") # 進捗表示

                    # 今周期検出済みフラグをクリア
                    det_numbers_mng.initCycle() 

                    # 今周期の検出結果を追加
                    #   車に包含されたナンバープレートのみ有効化
                    #   過去物体（ナンバープレート）との紐づけ（トラッキング）
                    det_numbers_mng.addCurDetNumber(det_result, same_cur_iou_th, own_car_rate_th) 

                    # 検出結果の時系列フィルタリング
                    #   物体管理（検出状態を保持する期間を累積信頼度で管理）
                    #   未検出時の位置推定（カルマンフィルタ適用）
                    det_numbers_mng.updateCycle() 

                    # フィルタリング後の検出結果（ナンバープレート）を取得
                    det_numbers = det_numbers_mng.getNumberPlates()

                    if is_blur == True:
                        # ナンバープレート検出位置にぼかしを入れる
                        img_org = ImageProc.blurDetObject(img_org, det_numbers, blur_kernel_size)

                        # 固定領域を消去
                        for img_erase_rect in img_erase_rects:
                            img_org = img_erase_rect.eraseRectArea(img_org, True)

                    # Debug表示
                    img_debug = img_org.copy()

                    if is_debug == True:
                        # 検出結果（車）を追加取得
                        det_numbers += det_numbers_mng.getCars() 

                        # 検出範囲を描画
                        for img_proc in img_procs:
                            img_debug = img_proc.drawDetArea(img_debug, DrawPen((0,128,0), 1, 0.4))
                        # 消去範囲を描画
                        for img_erase_rect in img_erase_rects:
                            img_debug = img_erase_rect.drawDetArea(img_debug, DrawPen((0,128,128), 1, 0.4), "erase area")

                        # 検出結果を描画
                        img_debug = ImageProc.drawResultDet(img_debug, det_result,  DrawPen((255,255,255), 1, 0.4))
                        img_debug = ImageProc.drawResultDet(img_debug, det_numbers, DrawPen((0,255,0), 1, 0.4))

                        # FPS等を描画
                        img_debug = ImageProc.drawResultSummary(img_debug, batch_frame_no, num_frame, 
                                                                ssd_model.device_.type, 
                                                                ssd_model.net_type_,
                                                                time_per_batch,
                                                                DrawPen((255,255,255), 2, 0.6))
                        
                        if is_output_debug == False:
                            draw_pen = DrawPen((0,255,255), 2, 0.5)
                            ImageProc.drawText(img_debug, "Debug info is nothing in output(image/movie)..", (15, 35), draw_pen.char_size_, draw_pen.col_, draw_pen.thick_, True)

                    # 画像保存
                    if is_output_image == True:
                        frame_img_fpath = f"{output_imgdir_path}/F{batch_frame_no:05}.jpg" 
                        if is_output_debug == True:
                            cv2.imwrite(frame_img_fpath, img_debug)
                        else:
                            cv2.imwrite(frame_img_fpath, img_org)

                    # 動画出力
                    if out_movie is not None:
                        if is_output_debug == True:
                            out_movie.write(img_debug)
                        else:
                            out_movie.write(img_org)

                    # 検出結果出力
                    if detcsv_fp is not None:
                        outcsv_line = dumpDetResultsToCsvLine(batch_frame_no, det_result)
                        detcsv_fp.write(f"{outcsv_line}\n")

                    # 表示
                    if is_disp_debug == True:
                        cv2.imshow(output_imgdir_name, img_debug)
                    else:
                        cv2.imshow(output_imgdir_name, img_org)

                    # key = cv2.waitKey(int(1000.0 / play_fps)) & 0xFF
                    key = cv2.waitKey(int(100.0 / play_fps)) & 0xFF
                    if key == ord("q"):
                        break

    cv2.destroyAllWindows()
    print("output: ", f"{output_imgdir_path}/{output_imgdir_name}.mp4")

    if out_movie is not None:
        out_movie.release()
        out_movie = None

    if detcsv_fp is not None:
        detcsv_fp.close()

    return


def main(media_fpath:str, cfg:Dict[str,Any]):
    """ メイン（SSDモデル作成、ナンバープレートぼかし処理実行）

    Args:
        media_fpath (str)   : 入力動画パス
        cfg (Dict[str,Any]) : config
    """    
    net_type:str     = cfg["ssd_model_net_type"]
    weight_fpath:str = cfg["ssd_model_weight_fpath"]

    if (os.path.isfile(media_fpath) == False) and (os.path.isdir(media_fpath) == False):
        print("Error: ", media_fpath, " is nothing.")

    else:
        # SSDモデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(f"Using Device: {device}")

        ssd_model = SSDModelDetector(device, net_type, weight_fpath)

        media_fname:str = os.path.basename(media_fpath)
        if ".mp4" in media_fname:
            # ナンバープレートぼかし実行
            main_blur_movie(media_fpath, ssd_model, cfg)

        else:
            print("No support ext [", media_fname, "]")
    
    return

if __name__ == "__main__":
    args = sys.argv

    cfg = {
        # 動画再生fps (負値＝入力動画のfpsそのまま)
        "play_fps"     : -1.0,

        # 検出範囲(1280x720、真ん中or右車線走行シーン、駐車場シーン用)
        "img_procs"    : [ImageProc(0, 250, 350, 600), 
                          ImageProc(250, 200, 550, 500), 
                          ImageProc(480, 200, 780, 500), 
                          ImageProc(730, 200, 1030, 500), 
                          ImageProc(930, 250, 1280, 600)],

        # 検出範囲(1280x720、左車線走行シーン用)
        # "img_procs"    : [ImageProc(480, 200, 780, 500), 
        #                   ImageProc(730, 200, 1030, 500), 
        #                   ImageProc(930, 250, 1280, 600)], 

        # 固定の矩形領域を消去
        # "img_erase"    : [ImageProc(367, 33, 367+257, 33+100)],
        "img_erase"    : [ImageProc()], # 消去なし

        # ぼかし強度(カーネルサイズ)
        "blur_kernel_size" : 10,

        "is_blur"      : True,      # ぼかしを入れる
        # "is_blur"      : False,     # (debug) ぼかしを入れない

        # "is_debug"     : False,       # 検出枠出力なし
        "is_debug"     : True,      # (debug) 検出枠（時系列処理された枠）出力あり

        "is_output_movie" : True,   # 結果を動画出力
        # "is_output_movie" : False,  # (debug) 動画出力しない

        # "is_output_image" : True,   # （debug) 結果を画像（フレーム毎）出力
        "is_output_image" : False,  # 画像出力しない

        "is_disp_debug": True,        # ウィンドウに検出枠表示あり(※is_debug==True時のみ有効)
        # "is_disp_debug": True,        # ウィンドウに検出枠表示なし
        # "is_output_debug": True,      # 動画or画像出力に検出枠表示あり(※is_debug==True時のみ有効)
        "is_output_debug": False,     # 動画or画像出力に検出枠表示なし

        # (トラッキング) 検出時の、累積信頼度の上限（これ以上は累積信頼度を上昇させない）
        "ACCUM_CONF_MAX" : 10.0,

        # (トラッキング) 過去の車と現在の車の外接矩形の重なり(iou)閾値
        "same_cur_iou_th" : 0.2,

        # ナンバープレートが車に所有されているかどうかの判定閾値
        "own_car_rate_th" : 0.5,

        # (SSDモデル:VGGベース)ネットワーク種別/パラメータ/バッチ処理数(※)
        #   (※) バッチ処理数 ＝検出範囲数 x フレーム数
        # "ssd_model_net_type"        : "vgg16-ssd",
        # "ssd_model_weight_fpath"    : "./weights/vgg16-ssd_best_od_cars.pth", 
        # "ssd_model_num_batch"       : 32,
        # "ssd_model_is_det_detail"   : False, # predictDetail()実行有無（実行すると処理速度は低下するが未検出小）
        # "ssd_model_detail_minsize"  : 100,   # predictDetail()実行時の検出範囲最小サイズ[px]

        # (SSDモデル:mobilenetベース)ネットワーク種別/パラメータ/バッチ処理数
        "ssd_model_net_type"        : "mb2-ssd",
        "ssd_model_weight_fpath"    : "./weights/mb2-ssd_best_od_cars.pth", 
        "ssd_model_num_batch"       : 64,
        "ssd_model_is_det_detail"   : True, # predictDetail()実行有無（実行すると処理速度は低下するが未検出小）
        "ssd_model_detail_minsize"  : 100,  # predictDetail()実行時の検出範囲最小サイズ[px]

        # (SSDモデル) 信頼度confの足切り閾値
        "ssd_model_conf_lower_th" : 0.5,

        # (SSDモデル) 重複枠削除する重なり率(iou)閾値
        # "ssd_model_iou_th" : 0.5,
        "ssd_model_iou_th" : 0.4,
    }

    root = tk.Tk()
    root.withdraw() # メインウィンドウは非表示

    # ファイルダイアログでmp4ファイル選択
    file_type = [("mp4ファイル","*.mp4")] 
    media_fpath = filedialog.askopenfilename(filetypes = file_type) 

    if media_fpath != "":
        print(f"open file [{media_fpath}]")
        main(media_fpath, cfg)

