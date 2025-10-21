#!/usr/bin/env python3

import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import time
import os
import copy
import numpy as np
from enum import IntEnum, auto
from typing import List,Tuple,Dict,Any
from tqdm import tqdm
import torch

from common_ssd import ImageProc, DrawPen, MovieLoader, DetResult
from common_ssd import readDetResultsFromCsvLine, dumpDetResultsToCsvLine
from predict_ssd import SSDModelDetector

class FrameProc:
    def __init__(self, frame_no:int, proc_area:List[ImageProc],target_cls:str, det_minsize:int):
        self.frame_no_    = frame_no
        self.proc_area_   = proc_area
        self.target_cls_  = target_cls
        self.det_minsize_ = det_minsize

        self.det_result_:List[DetResult] = []
        self.is_save_area_:List[bool] = [False] * len(self.proc_area_)
        self.is_save_det_:List[bool]  = []
        return

    def setDetResult(self, det_result:List[DetResult]):
        self.det_result_  = copy.deepcopy(det_result)
        self.is_save_det_ = [False] * len(self.det_result_)
        return
    
    def crearDetResult(self):
        self.det_result_.clear()
        return
    
    def isExistDetResult(self) -> bool:
        return len(self.det_result_) > 0
    
    def isExistSaveArea(self) -> bool:
        is_exist = False
        num_true = self.is_save_area_.count(True)
        if num_true > 0:
            is_exist = True

        return is_exist
    
    def isExistSaveDet(self) -> bool:
        is_exist = False
        num_true = self.is_save_det_.count(True)
        if num_true > 0:
            is_exist = True

        return is_exist

    def setToggleIsSaveArea(self, loc_x:int, loc_y:int):
        # print(f"In FrameProc.setToggleIsSaveArea({loc_x}, {loc_y})")
        for area_no, cur_area in enumerate(self.proc_area_):
            if cur_area.isInArea(loc_x, loc_y) == True:
                self.is_save_area_[area_no] = not self.is_save_area_[area_no]
        return

    def isSelectable(self, det_result:DetResult) -> bool:
        is_selectable = False
        if det_result.class_name_ == self.target_cls_:
            if (det_result.bbox_w_ > self.det_minsize_) and (det_result.bbox_h_ > self.det_minsize_):
                is_selectable = True
        return is_selectable

    def setToggleIsSaveDet(self, loc_x:int, loc_y:int):
        # print(f"In FrameProc.setToggleIsSaveDet({loc_x}, {loc_y})")
        for det_no, cur_det in enumerate(self.det_result_):
            if (cur_det.isInArea(loc_x, loc_y) == True) and (self.isSelectable(cur_det) == True):
                self.is_save_det_[det_no] = not self.is_save_det_[det_no]
        return

    def paint(self, img:np.ndarray) -> np.ndarray:
        # 切り出し範囲を描画
        for area_no, cur_area in enumerate(self.proc_area_):
            pen = DrawPen((0,128,0), 1, 0.4)
            if self.is_save_area_[area_no] == True:
                pen = DrawPen((0,255,255), 1, 0.4)

            img = cur_area.drawDetArea(img, pen, f"area {area_no}")

        # 検出結果を描画
        if self.isExistDetResult() == True:

            for det_no, cur_det in enumerate(self.det_result_):
                pen = DrawPen((255,255,255), 1, 0.4)
                if self.is_save_det_[det_no] == True:
                    pen = DrawPen((0,255,255), 1, 0.4)
                elif self.isSelectable(cur_det) == True:
                    pen = DrawPen((255,255,0), 1, 0.4)
                else:
                    pass

                img = ImageProc.drawResultDetOne(img, cur_det, pen)

        return img

    def saveFrame(self, frame_img:np.ndarray, fpath_base:str, det_minsize:int) -> bool:
        is_saved = False

        # 選択した固定領域をカット＆保存
        for area_no, cur_area in enumerate(self.proc_area_):

            if self.is_save_area_[area_no] == True:

                fpath_base_area = f"{fpath_base}_area{area_no}"
                self.saveOneArea(frame_img, cur_area, fpath_base_area)
                is_saved = True

        # 選択した検出領域をカット＆保存
        for det_no, cur_det in enumerate(self.det_result_):
            if self.is_save_det_[det_no] == True:
                fpath_base_det = f"{fpath_base}_{cur_det.class_name_}{det_no}"

                cur_det_area = ImageProc()
                cur_det_area.initFromDet(frame_img, cur_det, det_minsize)

                self.saveOneArea(frame_img, cur_det_area, fpath_base_det)
                is_saved = True

        return is_saved

    def saveOneArea(self, frame_img:np.ndarray, cur_area:ImageProc, fpath_base_area:str):

        # 画像保存
        frame_img_cliped = cur_area.clip(frame_img)
        frame_img_fpath  = f"{fpath_base_area}.jpg"
        cv2.imwrite(frame_img_fpath, frame_img_cliped)

        # cur_areaに含まれる検出結果をxml保存
        target_area_bbox  = cur_area.getBBox()
        target_det_result = [det
                            for det in self.det_result_
                            if DetResult.calcOverlapAreaBBox(target_area_bbox, det.bbox_) > 0.0]
        
        if len(target_det_result) > 0:
            self.saveAnnoXml(cur_area, target_det_result, frame_img_fpath)

        return

    def saveAnnoXml(self, target_area:ImageProc, target_det_result:List[DetResult], frame_img_fpath:str):

        fpath_dir        = os.path.dirname(frame_img_fpath)
        fpath_base_area  = os.path.splitext(os.path.basename(frame_img_fpath))[0]
        frame_xml_fpath  = f"{fpath_dir}/{fpath_base_area}.xml"

        with open(frame_xml_fpath, "w") as fp_xml:

            target_area_bbox  = target_area.getBBox()

            fp_xml.write("<annotation>\n")
            fp_xml.write(f"\t<filename>{os.path.basename(frame_img_fpath)}</filename>\n")
            fp_xml.write("\t<size>\n")
            fp_xml.write(f"\t\t<width>{target_area.darea_w_-1}</width>\n")
            fp_xml.write(f"\t\t<height>{target_area.darea_h_-1}</height>\n")
            fp_xml.write(f"\t\t<depth>3</depth>\n")
            fp_xml.write("\t</size>\n")

            for tar_det in target_det_result:
                fp_xml.write("\t<object>\n")
                fp_xml.write(f"\t\t<name>{tar_det.class_name_}</name>\n")
                fp_xml.write("\t\t<pose>Unspecified</pose>\n")
                fp_xml.write("\t\t<truncated>0</truncated>\n")
                fp_xml.write("\t\t<difficult>0</difficult>\n")
                fp_xml.write("\t\t<bndbox>\n")
                fp_xml.write(f"\t\t\t<xmin>{tar_det.bbox_[0] - target_area_bbox[0]}</xmin>\n")
                fp_xml.write(f"\t\t\t<ymin>{tar_det.bbox_[1] - target_area_bbox[1]}</ymin>\n")
                fp_xml.write(f"\t\t\t<xmax>{tar_det.bbox_[2] - target_area_bbox[0]}</xmax>\n")
                fp_xml.write(f"\t\t\t<ymax>{tar_det.bbox_[3] - target_area_bbox[1]}</ymax>\n")
                fp_xml.write("\t\t</bndbox>\n")
                fp_xml.write("\t</object>\n")

            fp_xml.write("</annotation>\n")

        return 

class FrameSet:
    def __init__(self, cfg:Dict[str,Any]):
        self.cfg_       = cfg
        self.num_frame_ = 0

        self.frames_:List[FrameProc]    = []
        self.proc_area_:List[ImageProc] = cfg["proc_area"]

        self.outdir_path_       = ""
        self.outimg_fname_base_ = ""

        # SSDモデル作成
        device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net_type_       = str(self.cfg_["ssd_model_net_type"])
        weight_fpath    = str(self.cfg_["ssd_model_weight_fpath"])
        self.ssd_model_ = SSDModelDetector(device, net_type_, weight_fpath)

        self.ssd_conf_th_ = float(cfg["ssd_model_conf_th"])
        self.ssd_iou_th_  = float(cfg["ssd_model_iou_th"])
        self.ssd_detail_minsize_ = int(cfg["ssd_model_detail_minsize"]) # predictDetail()実行時の検出範囲最小サイズ[px]
        self.ssd_detail_areacls_ = str(cfg["ssd_model_detail_areacls"]) # predictDetail()実行時の検出範囲にするクラス

        return

    def createOutputDir(self, movie_fpath:str):
        self.outimg_fname_base_ = os.path.splitext(os.path.basename(movie_fpath))[0]
        self.outdir_path_       = f"./output.data/{self.outimg_fname_base_}"
        if os.path.isdir(self.outdir_path_) == False:
            os.makedirs(self.outdir_path_)
        return

    def createFrames(self, num_frame:int):
        if len(self.frames_) > 0:
            self.frames_.clear()

        self.num_frame_ = num_frame
        for frame_no in range(num_frame):
            self.frames_.append(FrameProc(frame_no, self.proc_area_, self.ssd_detail_areacls_, self.ssd_detail_minsize_))
        return

    def delFrames(self):
        self.num_frame_ = 0
        self.frames_.clear()
        return
    
    def loadDetResults(self, det_csv_fpath:str):
        with open(det_csv_fpath,"r") as fp:
            lines_csv = fp.readlines()
            for line_csv in lines_csv:
                (frame_no, det_results) = readDetResultsFromCsvLine(line_csv)

                frame = self.getFrame(frame_no)
                if frame is not None:
                    frame.setDetResult(det_results)
        return
    
    def saveDetResults(self, det_csv_fpath:str):
        if self.isExistDetResult() == True:
            with open(det_csv_fpath,"w") as fp:
                for cur_frame in self.frames_:
                    line_csv_str = dumpDetResultsToCsvLine(cur_frame.frame_no_, cur_frame.det_result_)
                    fp.write(f"{line_csv_str}\n")
        return
    
    def isExistDetResult(self) -> bool:
        is_exist = False
        for cur_frame in self.frames_:
            if cur_frame.isExistDetResult() == True:
                is_exist = True
                break

        return is_exist

    def getFrame(self, frame_no:int) -> FrameProc:
        frame = None
        if (0 <= frame_no) and (frame_no < self.num_frame_):
            frame = self.frames_[frame_no]
        return frame
    
    def setToggleIsSaveArea(self, frame_no:int, loc_x:int, loc_y:int):
        frame = self.getFrame(frame_no)
        if frame is not None:
            frame.setToggleIsSaveArea(loc_x, loc_y)
        return

    def setToggleIsSaveDet(self, frame_no:int, loc_x:int, loc_y:int):
        if self.isExistDetResult() == True:
            frame = self.getFrame(frame_no)
            if frame is not None:
                frame.setToggleIsSaveDet(loc_x, loc_y)
        return

    def isExistSaveAreaDet(self) -> bool:
        is_exist = False

        for cur_frame in self.frames_:
            if (cur_frame.isExistSaveArea() == True) or (cur_frame.isExistSaveDet() == True):
                is_exist = True
                break

        return is_exist

    def __len__(self) -> int:
        return self.num_frame_
    
    def detectSSDmodel(self, batch_imgs:List[np.ndarray], num_batch_real:int) -> List[List[DetResult]]:
        # SSSD物体検出（複数フレーム分をまとめて検出）
        det_results = self.ssd_model_.predict(self.proc_area_, 
                                              batch_imgs, 
                                              self.ssd_conf_th_, 
                                              self.ssd_iou_th_)
        
        # SSD物体検出（detail_areaclsの枠内を検出）
        det_results = self.ssd_model_.predictDetail(det_results, 
                                                    batch_imgs, 
                                                    self.ssd_detail_minsize_, 
                                                    num_batch_real, 
                                                    self.ssd_detail_areacls_, 
                                                    self.ssd_conf_th_, 
                                                    self.ssd_iou_th_)

        return det_results


class VPEvent(IntEnum):
    OPEN_MOVIE_      = auto()
    CLICK_BTN_NEXT_  = auto()
    CLICK_BTN_PREV_  = auto()
    MOVE_SEEKBAR_    = auto()
    MOUSE_CLK_LEFT_  = auto()
    MOUSE_CLK_RIGHT_ = auto()

    CLICK_BTN_DET_START_ = auto()
    CLICK_BTN_DET_STOP_  = auto()
    DETECT_END_          = auto()

    CLICK_BTN_PLAY_START_ = auto()
    CLICK_BTN_PLAY_STOP_  = auto()
    PLAY_TO_END_          = auto()

    CLICK_BTN_SAVE_START_ = auto()
    CLICK_BTN_SAVE_STOP_  = auto()
    SAVE_END_             = auto()

class VPStatus:
    class StDef(IntEnum):
        INIT_            = auto()
        READY_           = auto()
        PLAYING_         = auto()
        PLAY_STOPPING_   = auto()
        DETECTING_       = auto()
        DETECT_STOPPING_ = auto()
        SAVING_          = auto()
        SAVE_STOPPING_   = auto()

    def __init__(self):
        self.status_ = VPStatus.StDef.INIT_
        return

    def __eq__(self, other_stval:StDef) -> bool:
        return self.status_ == other_stval

    def __str__(self) -> str:
        str_status = "UNKNOWN"
        if self.status_ == VPStatus.StDef.INIT_:
            str_status = "INIT"
        elif self.status_ == VPStatus.StDef.READY_:
            str_status = "READY"
        elif self.status_ == VPStatus.StDef.PLAYING_:
            str_status = "PLAYING"
        elif self.status_ == VPStatus.StDef.PLAY_STOPPING_:
            str_status = "PLAY_STOPPING"
        elif self.status_ == VPStatus.StDef.DETECTING_:
            str_status = "DETECTING"
        elif self.status_ == VPStatus.StDef.DETECT_STOPPING_:
            str_status = "DETECT_STOPPING"
        elif self.status_ == VPStatus.StDef.SAVING_:
            str_status = "SAVING"
        elif self.status_ == VPStatus.StDef.SAVE_STOPPING_:
            str_status = "SAVE_STOPPING"
        else:
            pass
        return str_status

    def trans(self, event:VPEvent) -> bool:
        is_change_state = False

        if self.status_ == VPStatus.StDef.INIT_:
            if event == VPEvent.OPEN_MOVIE_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True
            
        elif self == VPStatus.StDef.READY_:
            if event == VPEvent.CLICK_BTN_PLAY_START_:
                self.status_    = VPStatus.StDef.PLAYING_
                is_change_state = True
            
            elif event == VPEvent.CLICK_BTN_DET_START_:
                self.status_    = VPStatus.StDef.DETECTING_
                is_change_state = True

            elif event == VPEvent.CLICK_BTN_SAVE_START_:
                self.status_    = VPStatus.StDef.SAVING_
                is_change_state = True

            else:
                # 状態遷移はないが、UI部品の状態を更新したいケース
                if     (event == VPEvent.OPEN_MOVIE_) \
                    or (event == VPEvent.CLICK_BTN_NEXT_) \
                    or (event == VPEvent.CLICK_BTN_PREV_) \
                    or (event == VPEvent.MOVE_SEEKBAR_) \
                    or (event == VPEvent.MOUSE_CLK_LEFT_) \
                    or (event == VPEvent.MOUSE_CLK_RIGHT_):

                    is_change_state = True

        elif self.status_ == VPStatus.StDef.PLAYING_:
            if event == VPEvent.PLAY_TO_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True

            elif event == VPEvent.CLICK_BTN_PLAY_STOP_:
                self.status_    = VPStatus.StDef.PLAY_STOPPING_
                is_change_state = True
            else:
                pass

        elif self.status_ == VPStatus.StDef.PLAY_STOPPING_:
            if event == VPEvent.PLAY_TO_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True

        elif self.status_ == VPStatus.StDef.DETECTING_:
            if event == VPEvent.DETECT_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True
            elif event == VPEvent.CLICK_BTN_DET_STOP_:
                self.status_    = VPStatus.StDef.DETECT_STOPPING_
                is_change_state = True
            else:
                pass

        elif self.status_ == VPStatus.StDef.DETECT_STOPPING_:
            if event == VPEvent.DETECT_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True

        elif self.status_ == VPStatus.StDef.SAVING_:
            if event == VPEvent.SAVE_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True
            elif event == VPEvent.CLICK_BTN_SAVE_STOP_:
                self.status_    = VPStatus.StDef.SAVE_STOPPING_
                is_change_state = True
            else:
                pass

        elif self.status_ == VPStatus.StDef.SAVE_STOPPING_:
            if event == VPEvent.SAVE_END_:
                self.status_    = VPStatus.StDef.READY_
                is_change_state = True

        else:
            pass

        # print(f"trans: -> {self}")

        return is_change_state

class CanvasPaintEventArg:
    def __init__(self, frame_no:int, frame_img:np.ndarray = None):
        self.frame_no_  = frame_no
        self.frame_img_:np.ndarray = None
        if frame_img is not None:
            self.frame_img_ = copy.deepcopy(frame_img)
        return

class VideoPlayer:

    BTN_LABEL_OPENMOVIE = "open movie"
    BTN_LABEL_PLAY      = "▶ play"
    BTN_LABEL_PREV      = "⏮ prev"
    BTN_LABEL_NEXT      = "⏭ next"
    BTN_LABEL_STOP      = "■ stop"
    BTN_LABEL_DET       = "detect"
    BTN_LABEL_SAVE      = "save"

    def __init__(self, root:tk.Tk, cfg:Dict[str,Any], video_path=None):
        self.status_ = VPStatus()

        self.cfg_ = cfg

        self.root_ = root
        self.root_.title("Train Image Extractor")
        self.movie_ = MovieLoader()

        self.frame_set_ = FrameSet(self.cfg_)

        # --- UI構築 ---
        self.createUi()

        if video_path is not None:
            self.loadMovie(video_path)
        return

    def createUi(self):
        # キー入力
        self.root_.bind("<KeyPress>", self.onPressKey)

        # 動画キャンバス
        self.canvas_ = tk.Label(self.root_, bg="black")
        self.canvas_.pack(padx=10, pady=10)
        self.canvas_.bind("<Button-1>", self.onMouseLeftClickCanvas)  # マウス左クリック
        self.canvas_.bind("<Button-3>", self.onMouseRightClickCanvas) # マウス右クリック

        self.canvas_event_lock_ = threading.Lock()
        self.canvas_event_arg_  = CanvasPaintEventArg(0)
        self.canvas_.bind("<<PaintFrame>>", self.onPaintFrame)

        # コントロールボタン
        control_frame = ttk.Frame(self.root_)
        control_frame.pack()

        self.open_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_OPENMOVIE, command=self.onClickBtnOpenFile)
        self.open_btn_.grid(row=0, column=0, padx=5)

        self.play_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_PLAY, command=self.onClickBtnPlayStop)
        self.play_btn_.grid(row=0, column=1, padx=5)

        self.prev_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_PREV, command=self.onClickBtnPrevFrame)
        self.prev_btn_.grid(row=0, column=2, padx=5)

        self.next_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_NEXT, command=self.onClickBtnNextFrame)
        self.next_btn_.grid(row=0, column=3, padx=5)

        self.det_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_DET, command=self.onClickBtnDetectStop)
        self.det_btn_.grid(row=0, column=4, padx=5)

        self.save_btn_ = ttk.Button(control_frame, text=VideoPlayer.BTN_LABEL_SAVE, command=self.onClickBtnSaveStop)
        self.save_btn_.grid(row=0, column=5, padx=5)

        # ステータス表示
        self.status_label_ = ttk.Label(control_frame, text="---")
        self.status_label_.grid(row=0, column=6, padx=5)

        # シークバー
        self.seek_var_ = tk.DoubleVar()
        self.seekbar_ = ttk.Scale(self.root_, from_=0, to=100, orient="horizontal",
                                 variable=self.seek_var_, command=self.onMoveSeekbar)
        self.seekbar_.pack(fill="x", padx=10, pady=5)

        # Usage
        self.usage_label_ = ttk.Label(self.root_, text=f"[{VideoPlayer.BTN_LABEL_OPENMOVIE}] Open movie(mp4)")
        self.usage_label_.pack(pady=5)

        self.changeUiStatus(self.status_, False, False)
        return

    def changeUiStatus(self, status:VPStatus, is_exist_det:bool, is_exist_savearea:bool):
        if status == VPStatus.StDef.INIT_:
            self.open_btn_.config(state=tk.NORMAL)
            self.det_btn_.config(state=tk.DISABLED)
            self.play_btn_.config(state=tk.DISABLED)
            self.prev_btn_.config(state=tk.DISABLED)
            self.next_btn_.config(state=tk.DISABLED)
            self.save_btn_.config(state=tk.DISABLED)
            self.seekbar_.config(state=tk.DISABLED)

        elif status == VPStatus.StDef.READY_:
            self.open_btn_.config(state=tk.NORMAL)
            if is_exist_det == False:
                self.det_btn_.config(state=tk.NORMAL)
            else:
                self.det_btn_.config(state=tk.DISABLED)

            self.play_btn_.config(state=tk.NORMAL)
            self.prev_btn_.config(state=tk.NORMAL)
            self.next_btn_.config(state=tk.NORMAL)

            if (is_exist_det == False) and (is_exist_savearea == False):
                self.save_btn_.config(state=tk.DISABLED)
            else:
                self.save_btn_.config(state=tk.NORMAL)

            self.seekbar_.config(state=tk.NORMAL)

        elif status == VPStatus.StDef.PLAYING_:
            self.open_btn_.config(state=tk.DISABLED)
            self.det_btn_.config(state=tk.DISABLED)
            self.play_btn_.config(state=tk.NORMAL)
            self.prev_btn_.config(state=tk.DISABLED)
            self.next_btn_.config(state=tk.DISABLED)
            self.save_btn_.config(state=tk.DISABLED)
            self.seekbar_.config(state=tk.DISABLED)

        elif status == VPStatus.StDef.DETECTING_:
            self.open_btn_.config(state=tk.DISABLED)
            self.det_btn_.config(state=tk.NORMAL)
            self.play_btn_.config(state=tk.DISABLED)
            self.prev_btn_.config(state=tk.DISABLED)
            self.next_btn_.config(state=tk.DISABLED)
            self.save_btn_.config(state=tk.DISABLED)
            self.seekbar_.config(state=tk.DISABLED)

        elif status == VPStatus.StDef.SAVING_:
            self.open_btn_.config(state=tk.DISABLED)
            self.det_btn_.config(state=tk.DISABLED)
            self.play_btn_.config(state=tk.DISABLED)
            self.prev_btn_.config(state=tk.DISABLED)
            self.next_btn_.config(state=tk.DISABLED)
            self.save_btn_.config(state=tk.NORMAL)
            self.seekbar_.config(state=tk.DISABLED)

        elif (status == VPStatus.StDef.PLAY_STOPPING_) \
             or (status == VPStatus.StDef.DETECT_STOPPING_) \
             or (status == VPStatus.StDef.SAVE_STOPPING_):

            self.open_btn_.config(state=tk.DISABLED)
            self.det_btn_.config(state=tk.DISABLED)
            self.play_btn_.config(state=tk.DISABLED)
            self.prev_btn_.config(state=tk.DISABLED)
            self.next_btn_.config(state=tk.DISABLED)
            self.save_btn_.config(state=tk.DISABLED)
            self.seekbar_.config(state=tk.DISABLED)

        else:
            pass

        return

    def transState(self, event:VPEvent):
        is_change_state = self.status_.trans(event)
        if is_change_state == True:
            self.changeUiStatus(self.status_, 
                                self.frame_set_.isExistDetResult(),
                                self.frame_set_.isExistSaveAreaDet())
        return

    def onPressKey(self, e:tk.Event):
        # print(f"e.keysym = {e.keysym}")

        if e.keysym == "Left":
            self.onClickBtnPrevFrame()
        elif e.keysym == "Right":
            self.onClickBtnNextFrame()
        else:
            pass
        return

    def onMouseLeftClickCanvas(self, e:tk.Event):
        if self.status_ == VPStatus.StDef.READY_:
            self.transState(VPEvent.MOUSE_CLK_LEFT_)
            self.frame_set_.setToggleIsSaveArea(self.movie_.cur_frame_no_, e.x, e.y)
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onMouseRightClickCanvas(self, e:tk.Event):
        if self.status_ == VPStatus.StDef.READY_:
            self.transState(VPEvent.MOUSE_CLK_RIGHT_)
            self.frame_set_.setToggleIsSaveDet(self.movie_.cur_frame_no_, e.x, e.y)
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onClickBtnOpenFile(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.loadMovie(path)
        return

    def loadMovie(self, path:str):

        self.movie_.load(path, int(self.cfg_["ssd_model_num_batch"]))

        if self.movie_.isOpened() == False:
            self.status_label_.config(text=f"Can't open movie file [{path}]")

        else:
            self.frame_set_.createOutputDir(path)

            num_frame = self.movie_.getNumFrame()
            self.frame_set_.createFrames(num_frame)

            # 検出結果csvが動画と同じディレクトリにあれば、読み込み
            det_csv_fpath = f"{os.path.splitext(path)[0]}.csv"
            if os.path.isfile(det_csv_fpath) == True:
                self.frame_set_.loadDetResults(det_csv_fpath)

            self.seekbar_.config(to=num_frame - 1)

            path_basename = os.path.basename(path)
            self.status_label_.config(text=f"Success to load [{path_basename}]")
            self.root_.title(path_basename)

            self.transState(VPEvent.OPEN_MOVIE_)
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onClickBtnNextFrame(self):
        if self.movie_.isOpened() == True:
            self.transState(VPEvent.CLICK_BTN_NEXT_)

            self.movie_.nextCurFrame()
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def onClickBtnPrevFrame(self):
        if self.movie_.isOpened() == True:
            self.transState(VPEvent.CLICK_BTN_PREV_)

            self.movie_.prevCurFrame()
            self.showFrame(self.movie_.cur_frame_no_)
        return
    
    def onPaintFrame(self, event:tk.Event):
        frame_no = 0
        frame_img:np.ndarray = None

        with self.canvas_event_lock_:
            frame_no = self.canvas_event_arg_.frame_no_
            if self.canvas_event_arg_.frame_img_ is not None:
                frame_img = copy.deepcopy(self.canvas_event_arg_.frame_img_)

        self.showFrame(frame_no, frame_img)
        return
    
    def sendPaintFrameEvent(self, frame_no:int, frame_img:np.ndarray = None):
        with self.canvas_event_lock_:
            self.canvas_event_arg_ = CanvasPaintEventArg(frame_no, frame_img)
        
        self.canvas_.event_generate("<<PaintFrame>>")
        return 

    def showFrame(self, frame_index:int, frame:np.ndarray=None):
        if self.movie_.isOpened() == True:
            
            ret = True
            if frame is None:
                self.movie_.setCurFrame(frame_index)
                ret, frame = self.movie_.getCurFrame()

            if ret == True:

                # frameに重畳描画
                cur_frame = self.frame_set_.getFrame(frame_index)
                if cur_frame is not None:
                    frame = cur_frame.paint(frame)

                # frameを表示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                # img = img.resize((640, 360))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas_.imgtk = imgtk
                self.canvas_.config(image=imgtk)
                self.seek_var_.set(frame_index)

                # txt表示
                if self.status_ == VPStatus.StDef.READY_:
                    usage_str = "[Mouse Left] Select area"
                    if cur_frame.isExistDetResult() == True:
                        min_size = self.frame_set_.ssd_detail_minsize_
                        tar_cls  = self.frame_set_.ssd_detail_areacls_
                        # usage_str = f"{usage_str}, [Mouse Right] Select det rect({tar_cls}(over {min_size}x{min_size}px) only)"
                        usage_str = f"{usage_str}, [Mouse Right] Select det rect(skyblue only)"

                    usage_str = f"{usage_str}, [←] prev, [→] next"
                    self.usage_label_.config(text=usage_str)

                self.status_label_.config(text=f"Frame: {frame_index}/{self.movie_.num_cap_frame_}")
        return

    def onClickBtnPlayStop(self):

        if self.movie_.isOpened() == True:
            if self.status_ == VPStatus.StDef.READY_:
                self.transState(VPEvent.CLICK_BTN_PLAY_START_)
                t = threading.Thread(target=self.threadPlayMovie, daemon=True)
                t.start()
            else:
                self.transState(VPEvent.CLICK_BTN_PLAY_STOP_)

        return


    def onClickBtnDetectStop(self):
        if self.movie_.isOpened() == True:

            if self.status_ == VPStatus.StDef.READY_:
                self.transState(VPEvent.CLICK_BTN_DET_START_)
                t = threading.Thread(target=self.threadProcDetectMovie, daemon=True)
                t.start()
            else:
                self.transState(VPEvent.CLICK_BTN_DET_STOP_)

        return

    def onClickBtnSaveStop(self):
        if self.movie_.isOpened() == True:

            if self.status_ == VPStatus.StDef.READY_:
                self.transState(VPEvent.CLICK_BTN_SAVE_START_)
                t = threading.Thread(target=self.threadSaveFrames, daemon=True)
                t.start()
            else:
                self.transState(VPEvent.CLICK_BTN_SAVE_STOP_)

        return

    def onMoveSeekbar(self, val):
        if self.movie_.isOpened() == True:
            self.transState(VPEvent.MOVE_SEEKBAR_)

            frame_idx = int(float(val))
            self.movie_.setCurFrame(frame_idx)
            self.showFrame(self.movie_.cur_frame_no_)

        return

    def threadPlayMovie(self):
        self.play_btn_.config(text=VideoPlayer.BTN_LABEL_STOP) # STOPボタンにする

        str_usage_label_tmp = str(self.usage_label_.cget("text"))
        self.usage_label_.config(text=f"[{VideoPlayer.BTN_LABEL_STOP}] Stop play")

        sleep_time_sec = 1.0 / self.movie_.play_fps_

        is_end = False
        while (is_end == False) and (self.status_ == VPStatus.StDef.PLAYING_):

            is_end = self.movie_.nextCurFrame()

            self.sendPaintFrameEvent(self.movie_.cur_frame_no_)
            time.sleep(sleep_time_sec)

        # 再生終了
        self.transState(VPEvent.PLAY_TO_END_)
        self.play_btn_.config(text=VideoPlayer.BTN_LABEL_PLAY) # ボタンを元に戻す
        self.usage_label_.config(text=str_usage_label_tmp)
        return

    def threadProcDetectMovie(self):
        self.det_btn_.config(text=VideoPlayer.BTN_LABEL_STOP) # STOPボタンにする

        str_usage_label_tmp = str(self.usage_label_.cget("text"))
        self.usage_label_.config(text=f"[{VideoPlayer.BTN_LABEL_STOP}] Stop detect")

        num_batch = int(self.cfg_["ssd_model_num_batch"])

        sleep_time_sec = 1.0 / self.movie_.play_fps_

        num_proc_area   = len(self.frame_set_.proc_area_)
        num_batch_frame = int(num_batch / num_proc_area)
        if num_batch_frame < 1:
            num_batch_frame = 1

        num_batch_real = num_batch_frame * num_proc_area

        # 開始メッセージを表示
        pen = DrawPen((255,255,255), 2, 0.6)
        ssd_nettype = self.frame_set_.ssd_model_.net_type_
        ssd_device  = self.frame_set_.ssd_model_.device_.type

        self.movie_.setCurFrame(0)
        ret, frame_img = self.movie_.getCurFrame()
        if ret == True:
            ImageProc.drawText(frame_img, 
                                f"Preparing ssd model..", 
                                (10, 15), 
                                pen.char_size_, pen.col_, pen.thick_, True)
            
            self.sendPaintFrameEvent(self.movie_.cur_frame_no_, frame_img)
            time.sleep(sleep_time_sec)

        # フレーム読み込み ＆ 検出実行
        tmp_cur_frame_no = self.movie_.cur_frame_no_
        self.movie_.resetIter() # 先頭から実行

        for batch_frame_nos, batch_imgs in self.movie_:

            if self.status_ == VPStatus.StDef.DETECTING_:
                if len(batch_imgs) > 0:

                    # SSD物体検出（複数フレーム分をまとめて検出）
                    det_results = self.frame_set_.detectSSDmodel(batch_imgs, num_batch_real)

                    # フレーム毎の処理
                    for batch_frame_no, frame_img, det_result in zip(batch_frame_nos, batch_imgs, det_results):
                        cur_frame = self.frame_set_.getFrame(batch_frame_no)
                        if cur_frame is not None:
                            # 検出結果を登録
                            cur_frame.setDetResult(det_result)

                            # 検出結果等をフレームに描画
                            ImageProc.drawText(frame_img, 
                                            f"Now detecting.. (net:{ssd_nettype}, dev:{ssd_device})", 
                                            (10, 15), 
                                            pen.char_size_, pen.col_, pen.thick_, True)

                            self.sendPaintFrameEvent(batch_frame_no, frame_img)

                        time.sleep(sleep_time_sec)

            else:
                # [Stopボタン押下で検出が中断した場合] 
                break

        # 検出結果csvを保存
        det_csv_fpath = f"{self.frame_set_.outdir_path_}/{self.frame_set_.outimg_fname_base_}.csv"
        self.frame_set_.saveDetResults(det_csv_fpath)

        # 検出終了
        self.transState(VPEvent.DETECT_END_)
        self.det_btn_.config(text=VideoPlayer.BTN_LABEL_DET) # ボタンを元に戻す
        self.usage_label_.config(text=str_usage_label_tmp)

        # 元の位置に戻す
        self.movie_.setCurFrame(tmp_cur_frame_no)
        self.sendPaintFrameEvent(self.movie_.cur_frame_no_)

        return

    def threadSaveFrames(self):
        self.save_btn_.config(text=VideoPlayer.BTN_LABEL_STOP) # STOPボタンにする

        str_usage_label_tmp = str(self.usage_label_.cget("text"))
        self.usage_label_.config(text=f"[{VideoPlayer.BTN_LABEL_STOP}] Stop save")

        det_minsize = int(self.cfg_["ssd_model_detail_minsize"])

        pen = DrawPen((255,255,255), 2, 0.6)
        sleep_time_sec = 1.0 / self.movie_.play_fps_

        tmp_cur_frame_no = self.movie_.cur_frame_no_

        for frame_no, cur_frame in enumerate(self.frame_set_.frames_):

            if self.status_ == VPStatus.StDef.SAVING_:

                fpath_base = f"{self.frame_set_.outdir_path_}/{self.frame_set_.outimg_fname_base_}_F{frame_no:05}" 

                self.movie_.setCurFrame(frame_no)
                (ret, frame_img) = self.movie_.getCurFrame()

                if ret == True:
                    
                    # フレーム情報を保存
                    is_saved = cur_frame.saveFrame(frame_img, fpath_base, det_minsize)

                    # 進捗表示
                    if is_saved == True:
                        ImageProc.drawText(frame_img, "Saved Frame", 
                                        (10, 15), pen.char_size_, pen.col_, pen.thick_, True)
                        
                        self.sendPaintFrameEvent(self.movie_.cur_frame_no_, frame_img)
                        time.sleep(sleep_time_sec)
                    
                    self.status_label_.config(text=f"Processing Frame: {frame_no+1}/{self.movie_.num_cap_frame_}..")

            else:
                # [Stopボタン押下でSave処理が中断した場合] 
                break

        # Save終了
        self.transState(VPEvent.SAVE_END_)
        self.save_btn_.config(text=VideoPlayer.BTN_LABEL_SAVE) # ボタンを元に戻す
        self.usage_label_.config(text=str_usage_label_tmp)

        # 元の位置に戻す
        self.movie_.setCurFrame(tmp_cur_frame_no)
        self.sendPaintFrameEvent(self.movie_.cur_frame_no_)

        return

if __name__ == "__main__":

    cfg = {
        # 切り出し範囲(1280x720用の設定)
        "proc_area"    : [ImageProc(0, 250, 350, 600), 
                          ImageProc(250, 200, 550, 500), 
                          ImageProc(480, 200, 780, 500), 
                          ImageProc(730, 200, 1030, 500), 
                          ImageProc(930, 250, 1280, 600)],

        # (SSDモデル:VGGベース)ネットワーク種別/パラメータ/バッチ処理数(※)
        #   (※) バッチ処理数 ＝検出範囲数 x フレーム数
        # "ssd_model_net_type"        : "vgg16-ssd",
        # "ssd_model_weight_fpath"    : "./weights/vgg16-ssd_best_od_cars.pth", 
        # "ssd_model_num_batch"       : 32,
        # "ssd_model_is_det_detail"   : False, # predictDetail()実行有無（実行すると処理速度は低下するが未検出小）
        # "ssd_model_detail_minsize"  : 100,   # predictDetail()実行時の検出範囲最小サイズ[px]
        # "ssd_model_detail_areacls"  : "car", # predictDetail()実行時の検出範囲にするクラス
        # "ssd_model_conf_th"         : 0.5,
        # "ssd_model_iou_th"          : 0.4,

        # (SSDモデル:mobilenetベース)ネットワーク種別/パラメータ/バッチ処理数
        "ssd_model_net_type"        : "mb2-ssd",
        "ssd_model_weight_fpath"    : "./weights/mb2-ssd_best_od_cars.pth", 
        "ssd_model_num_batch"       : 64,
        "ssd_model_is_det_detail"   : True,  # predictDetail()実行有無（実行すると処理速度は低下するが未検出小）
        "ssd_model_detail_minsize"  : 100,   # predictDetail()実行時の検出範囲最小サイズ[px]
        "ssd_model_detail_areacls"  : "car", # predictDetail()実行時の検出範囲にするクラス
        "ssd_model_conf_th"         : 0.5,
        "ssd_model_iou_th"          : 0.4,

    }

    # UI作成
    root = tk.Tk()
    app = VideoPlayer(root, cfg)
    root.mainloop()
