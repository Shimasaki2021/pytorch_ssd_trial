#!/usr/bin/env python3

import sys
import cv2
import os
import copy
import math
import time
from typing import List,Dict,Any

import numpy as np
import torch

from utils.data_augumentation import jaccard_numpy
from common_ssd import ImageProc, DetResult, DrawPen, Logger
from predict_ssd import SSDModelDetector

# 検出されたナンバープレート1つ分
# （ナンバープレートと、ナンバープレートを包含する車のペアを保持）
class DetNumberPlate:

    def __init__(self, cfg:Dict[str,Any]):
        self.id_          = 0
        self.is_valid_    = False
        self.is_det_cur_  = False # 今周期検出済フラグ
        self.det_frame_no = 0     # 検出時のフレーム番号
        self.obj_number_:DetResult = None
        self.obj_car_:DetResult    = None
        self.accum_conf_ = 0.0   # 累積信頼度（検出した信頼度confを積算）

        self.param_aconf_max1_:float = cfg["ACCUM_CONF_MAX1"]
        self.param_aconf_max2_:float = cfg["ACCUM_CONF_MAX2"]
        self.param_aconf_dec_:float  = cfg["ACCUM_CONF_DEC"]
        self.param_aconf_add_rate1_:float = cfg["ACCUM_CONF_ADD_RATE1"]
        self.param_aconf_add_rate2_:float = cfg["ACCUM_CONF_ADD_RATE2"]
        return
    
    def isValid(self) -> bool:
        if (self.obj_number_ is None) or (self.obj_car_ is None):
            self.is_valid_ = False
        return self.is_valid_
    
    def __eq__(self, other) -> bool:
        ret = False
        if (self.isValid() == True) and (other.isValid() == True):
            if self.id_ == other.id_:
                ret = True
        return ret

    def updateDetObject(self, frame_no:int, obj_number:DetResult, obj_car:DetResult):
        if (obj_number is not None) and (obj_car is not None):
            self.is_valid_   = True
            self.is_det_cur_ = True

            if self.det_frame_no == frame_no:
                # [更新時刻（フレーム番号）が同じ場合] 登録済みナンバープレートより信頼度が高ければ更新
                if self.obj_number_.score_ < obj_number.score_:
                    self.obj_number_ = copy.deepcopy(obj_number)
                    self.obj_car_    = copy.deepcopy(obj_car)
            else:
                # [更新時刻（フレーム番号）が異なる場合] 更新
                self.det_frame_no = frame_no
                self.obj_number_  = copy.deepcopy(obj_number)
                self.obj_car_     = copy.deepcopy(obj_car)

        return

    def setDetObject(self, frame_no:int, id:int, obj_number:DetResult, obj_car:DetResult):
        if (obj_number is not None) and (obj_car is not None):
            self.id_ = id
            self.det_frame_no = frame_no
            self.is_valid_    = True
            self.is_det_cur_  = True
            self.obj_number_  = copy.deepcopy(obj_number)
            self.obj_car_     = copy.deepcopy(obj_car)
        return

    def updateAccumConf(self, conf_val:float):
        # 値更新
        #   累積信頼度が一定値をこえたら、上昇がゆるやかになるようにする（減算時には適用しない）
        conf_add_val = conf_val
        if conf_val > 0.0:
            if self.accum_conf_ < self.param_aconf_max1_:
                conf_add_val = conf_val * self.param_aconf_add_rate1_
            else:
                conf_add_val = conf_val * self.param_aconf_add_rate2_

        self.accum_conf_ += conf_add_val

        # 上限でクリップ
        if self.accum_conf_ > self.param_aconf_max2_:
            self.accum_conf_ = self.param_aconf_max2_

        if self.accum_conf_ < 0.0:
            # [下限（＝0）を下回った場合] 無効化
            self.is_valid_ = False

        return

    @staticmethod
    def searchIncludedCar(obj_number:DetResult, det_objs:List[DetResult], include_car_rate_th:float) -> DetResult:
        # ナンバープレート（obj_number）を包含する車の検出結果を探す
        #    「重なり矩形面積／ナンバープレートの面積」が閾値以上の場合に、「包含する」と判定
        # 複数見つかった場合は、一番大きな車を返す
        obj_car:DetResult = None
        
        bbox_n = obj_number.bbox_
        bbox_n_area = float((bbox_n[2] - bbox_n[0]) * (bbox_n[3] - bbox_n[1]))

        if bbox_n_area > 0:

            for det_obj in det_objs:

                if det_obj.class_name_ == "car":

                    bbox_c = det_obj.bbox_
                    bbox_inter = np.concatenate([np.maximum(bbox_n[:2], bbox_c[:2]), np.minimum(bbox_n[2:], bbox_c[2:])])

                    if (bbox_inter[0] < bbox_inter[2]) and (bbox_inter[1] < bbox_inter[3]) :
                        # [外接矩形に重なりがある場合] 
                        
                        # 重なり部分の面積 / ナンバープレートの面積　を算出
                        bbox_inter_area = float((bbox_inter[2] - bbox_inter[0]) * (bbox_inter[3] - bbox_inter[1]))
                        include_rate    = bbox_inter_area / bbox_n_area

                        if include_rate > include_car_rate_th:
                            # [det_objがナンバープレート(obj_number)を包含する、と判定された場合]

                            if obj_car is not None:
                                # [既に１つ以上車が見つかっている場合] 一番大きい車を選択
                                if det_obj.bbox_area_ > obj_car.bbox_area_:
                                    obj_car = copy.deepcopy(det_obj)
                            else:
                                # [まだ車が見つかっていない場合] det_objを選択
                                obj_car = copy.deepcopy(det_obj)

        return obj_car

# 検出されたナンバープレートを時系列管理
class DetNumberPlateMng:

    def __init__(self, cfg:Dict[str,Any]):
        self.cfg_    = cfg
        self.new_id_ = 1
        self.det_obj_buf_:List[DetNumberPlate] = [] # 検出された物体（ナンバープレート、車）
        return

    def initCycle(self):
        # 周期処理開始時の初期化
        #   今周期検出済フラグをリセット
        for det_obj in self.det_obj_buf_:
            det_obj.is_det_cur_ = False
        return

    def addCurDetNumber(self, frame_no:int, det_results:List[DetResult], same_cur_iou_th=0.5, include_car_rate_th=0.5):
        # 今周期の検出結果を追加
        #   - ナンバープレートと、ナンバープレートを包含する車のペアを追加する
        #      - 既に保持されているナンバープレートと同一の場合は、差し替え
        #      - 同一ものがない場合は、新規追加

        for obj_det in det_results:
            if obj_det.class_name_ == "number":
                obj_number = obj_det
                obj_car    = DetNumberPlate.searchIncludedCar(obj_number, det_results, include_car_rate_th)

                if obj_car is not None:
                    # [obj_car(車)が、obj_number(ナンバープレート)を包含している場合]

                    # 今回のobj_carと同一のものが、既に過去に登録されているかどうかを確認（トラッキング）
                    buf_idx = self.trackingPastCar(obj_car, same_cur_iou_th)
                    if buf_idx >= 0:
                        # [登録されている場合] 差し替え
                        self.det_obj_buf_[buf_idx].updateDetObject(frame_no, obj_number, obj_car)

                    else:
                        # [未登録の場合] 新規追加
                        new_det_obj = DetNumberPlate(self.cfg_)
                        new_det_obj.setDetObject(frame_no, self.new_id_, obj_number, obj_car)
                        self.det_obj_buf_.append(new_det_obj)
                        self.new_id_ = self.new_id_ + 1

        return

    def trackingPastCar(self, obj_car:DetResult, same_cur_iou_th:float) -> int:
        # 今回のobj_carと同一の車が、既に過去に登録されているかどうかを確認（トラッキング）
        #   - 既に登録済みであれば、indexを返す
        #   - 複数該当する場合は、最も重なりの大きな車のindexを返す
        #   - 未登録の場合は、-1を返す
        # 同一かどうかは、車同士の外接矩形の重なり(iou)＞閾値(same_cur_iou_th) で判定

        ret_idx = -1

        # 過去物体の外接矩形bboxを抽出
        det_past_cars_bbox = []
        for det_past in self.det_obj_buf_:
            if det_past.isValid() == True:
                det_past_cars_bbox.append(det_past.obj_car_.bbox_)
            else:
                det_past_cars_bbox.append(np.array([-1,-1,-1,-1])) # 重なりなしと判定される外接矩形を入れておく

        if len(det_past_cars_bbox) > 0:
            # 重なり(iou)を算出（全登録分の車との重なりを一括で算出）
            obj_car_ious = jaccard_numpy(np.array(det_past_cars_bbox), obj_car.bbox_)

            # 外接矩形の重なり(iou)が閾値以上を抽出
            # 閾値以上が複数存在する場合は、重なり(iou)最大のものを抽出
            iou_max = same_cur_iou_th

            for buf_idx, obj_car_iou in enumerate(obj_car_ious):
                if obj_car_iou > iou_max:
                    ret_idx = buf_idx
                    iou_max = obj_car_iou

        return ret_idx

    def updateCycle(self):
        # 個々の物体（ナンバープレート＆車）の時系列管理（検出が続く物体のみを残す、未検出物体もすぐには消さない）
        #   - 今周期に検出ありの物体は、累積信頼度を加算
        #   - 今周期に検出なしの物体は、累積信頼度を減算
        #   - 累積信頼度が高いものを残す（低くなった物体（検出されなくなったもの）を削除）
        det_obj_buf_updated: List[DetNumberPlate] = []

        for det_obj in self.det_obj_buf_:

            if (det_obj.is_valid_ == True) and (det_obj.is_det_cur_ == True):
                # [今周期に検出ありの場合] 
                #  累積信頼度を加算。そのまま残す
                det_obj.updateAccumConf(det_obj.obj_number_.score_)
                det_obj_buf_updated.append(det_obj)
            else:
                # [今周期に検出なしの場合] 
                #  累積信頼度を減算。低くなったら無効化
                det_obj.updateAccumConf(det_obj.param_aconf_dec_)

                if det_obj.isValid() == True:
                    #  [無効化されていない場合] 残す
                    det_obj_buf_updated.append(det_obj)

        self.det_obj_buf_ = det_obj_buf_updated
        return

    def getNumberPlates(self) -> List[DetResult]:
        # 登録物体をそのまま返す
        # ret_objs = [det_obj.obj_number_ for det_obj in self.det_obj_buf_ if det_obj.isValid() == True]
        
        # ナンバープレートの信頼度を、累積信頼度に書き換えて返す
        ret_objs:List[DetResult] = []
        for det_obj in self.det_obj_buf_:
            if det_obj.isValid() == True:
                det_obj_new = copy.deepcopy(det_obj)
                det_obj_new.obj_number_.score_ = det_obj_new.accum_conf_
                ret_objs.append(det_obj_new.obj_number_)

        return ret_objs
    
    def getCars(self) -> List[DetResult]:
        # 登録物体をそのまま返す
        ret_objs = [det_obj.obj_car_ for det_obj in self.det_obj_buf_ if det_obj.isValid() == True]
        return ret_objs

def main_blur_movie(movie_fpath:str, ssd_model:SSDModelDetector, cfg:Dict[str,Any]):

    play_fps:float            = cfg["play_fps"]
    img_procs:List[ImageProc] = cfg["img_procs"]
    conf:float                = cfg["ssd_model_conf_lower_th"]
    overlap:float             = cfg["ssd_model_iou_th"]
    blur_kernel_size:int      = cfg["blur_kernel_size"]
    is_blur:bool              = cfg["is_blur"]
    is_debug:bool             = cfg["is_debug"]
    is_output_movie:bool      = cfg["is_output_movie"]
    is_output_image:bool      = cfg["is_output_image"]
    same_cur_iou_th:float     = cfg["same_cur_iou_th"]
    include_car_rate_th:float = cfg["include_car_rate_th"]

    # 画像出力用フォルダ作成
    if is_debug == True:
        output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0] + ".dbg"
    else:
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

    out_movie:cv2.VideoWriter = None

    if is_output_movie == True:
        fourcc    = cv2.VideoWriter_fourcc("m","p","4","v")
        out_movie = cv2.VideoWriter(f"{output_imgdir_path}/{output_imgdir_name}.mp4", fourcc, play_fps, (frame_w, frame_h))

    # 動画再生
    frame_no = 0

    det_numbers_mng = DetNumberPlateMng(cfg)

    while frame_no < num_frame:

        # 画像読み込み
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        img_org:np.ndarray = None
        (_, img_org) = cap.read()

        if frame_no % frame_play_step == 0:

            if not (img_org is None):
                
                print(f"processing F{frame_no:05} / F{num_frame:05}...")

                # 今周期（今フレーム）の検出結果をクリア
                det_numbers_mng.initCycle()
                
                time_s = time.perf_counter()

                # SSD物体検出
                det_results = ssd_model.predict(img_procs, img_org, conf, overlap)

                # 今周期の検出結果を登録＆更新
                det_numbers_mng.addCurDetNumber(frame_no, det_results, same_cur_iou_th, include_car_rate_th)
                det_numbers_mng.updateCycle() # ここで、未検出が続いている物体が削除される
                # print(f"len(det_buf)= {len(det_numbers_mng.det_obj_buf_)}")

                # 検出結果（ナンバープレート）を取得
                det_numbers  = det_numbers_mng.getNumberPlates()

                if is_blur == True:
                    # ナンバープレート検出位置にぼかしを入れる
                    img_org = ImageProc.blurDetObject(img_org, det_numbers, blur_kernel_size)

                if is_debug == True:
                    # 検出結果（車）を追加取得
                    det_numbers += det_numbers_mng.getCars() 

                    # 検出結果描画
                    img_org = ImageProc.drawResultDet(img_org, det_results, DrawPen((255,255,255), 1, 0.4))
                    img_org = ImageProc.drawResultDet(img_org, det_numbers, DrawPen((0,255,0), 1, 0.4))

                time_e = time.perf_counter()

                if is_debug == True:
                    # FPS等を描画
                    img_org = img_procs[0].drawResultSummary(img_org, frame_no, num_frame, 
                                                        ssd_model.device_.type, 
                                                        (time_e - time_s),
                                                        DrawPen((255,255,255), 2, 0.6))

                # 画像保存
                if is_output_image == True:
                    frame_img_fpath = f"{output_imgdir_path}/F{frame_no:05}.jpg" 
                    #frame_img_fpath = output_imgdir_path + "/F{:05}".format(frame_no) + ".jpg" 
                    cv2.imwrite(frame_img_fpath, img_org)

                # 動画出力
                if out_movie is not None:
                    out_movie.write(img_org)
                
                # 表示
                cv2.imshow(output_imgdir_name, img_org)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("output: ", f"{output_imgdir_path}/{output_imgdir_name}.mp4")

    if out_movie is not None:
        out_movie.release()
        out_movie = None

    return


def main(media_fpath:str, cfg:Dict[str,Any]):

    weight_fpath:str = cfg["ssd_model_weight_fpath"]

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
            main_blur_movie(media_fpath, ssd_model, cfg)

        else:
            print("No support ext [", media_fname, "]")
    
    return

if __name__ == "__main__":
    args = sys.argv

    cfg = {

        # 動画再生fps (負値＝入力動画のfpsそのまま)
        "play_fps"     : -1.0,

        # 検出範囲
        # "img_procs"    : [ImageProc(180, 250, 530, 600), 
        #                   ImageProc(480, 200, 780, 500), 
        #                   ImageProc(730, 200, 1030, 500), 
        #                   ImageProc(930, 250, 1280, 600)],
        "img_procs"    : [ImageProc(480, 200, 780, 500), 
                          ImageProc(730, 200, 1030, 500), 
                          ImageProc(930, 250, 1280, 600)], 
        
        # ぼかし強度(カーネルサイズ)
        "blur_kernel_size" : 10,

        "is_blur"      : True,      # ぼかしを入れる
        # "is_blur"      : False,     # (debug) ぼかしを入れない

        "is_debug"     : False,     # 検出枠表示なし
        # "is_debug"     : True,      # (debug) 検出枠（時系列処理された枠）を表示

        "is_output_movie" : True,   # 結果を動画出力
        # "is_output_movie" : False,  # (debug) 動画出力しない

        # "is_output_image" : True,   # （debug) 結果を画像（フレーム毎）出力
        "is_output_image" : False,  # 画像出力しない

        # (トラッキング) 検出時の、累積信頼度の上限1（ここを超えると累積信頼度の上昇がゆるやかになる）
        "ACCUM_CONF_MAX1" : 10.0,

        # (トラッキング) 検出時の、累積信頼度の上限2（これ以上は累積信頼度を上昇させない）
        "ACCUM_CONF_MAX2" : 15.0,

        # (トラッキング) 未検出時の、累積信頼度の減算値（1周期分）
        #   ナンバープレートが画面外になった際、 3(ACCUM_CONF_MAX2 / ACCUM_CONF_DEC) / fps [sec]で物体が消去される
        #   （30fps、MAX2=15, DEC=-0.5の場合、1[sec]で物体を消去）
        "ACCUM_CONF_DEC"  : -0.5,

        # (トラッキング) 累積信頼度の上昇率1（信頼度conf * ADD_RATE1 だけ上昇）
        "ACCUM_CONF_ADD_RATE1" : 1.0,

        # (トラッキング) 累積信頼度の上昇率2（信頼度conf * ADD_RATE2 だけ上昇）
        "ACCUM_CONF_ADD_RATE2" : 0.2,

        # (トラッキング) 過去の車と現在の車の外接矩形の重なり(iou)閾値
        "same_cur_iou_th" : 0.2,

        # ナンバープレートが車に包含されているかどうかの包含率閾値
        "include_car_rate_th" : 0.5,

        # (SSDモデル)パラメータ
        "ssd_model_weight_fpath" : "./weights/ssd_best_od_cars.pth", 

        # (SSDモデル) 信頼度confの足切り閾値
        "ssd_model_conf_lower_th" : 0.5,

        # (SSDモデル) 重複枠削除する重なり率(iou)閾値
        "ssd_model_iou_th" : 0.5,

    }

    if len(args) < 2:
        print("Usage: ", args[0], " [movie file path] ([play fps])")
    else:
        if len(args) >= 3:
            cfg["play_fps"] = float(args[2])

        main(args[1], cfg)
