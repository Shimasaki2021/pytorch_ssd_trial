import os
import glob
import random
import copy
import math

import cv2
import numpy as np
from typing import List,Dict,Tuple
from io import TextIOWrapper

import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

from utils.ssd_model import VOCDataset, DataTransform, Anno_xml2list, od_collate_fn
from utils.data_augumentation import jaccard_numpy

from vision.ssd.config import mobilenetv1_ssd_config as ssd_mb2_cfg

class DetResult:
    """ 検出結果1つ分 """

    def __init__(self, class_name:str, bbox:np.ndarray, score:float, is_det_cur=True):
        """ コンストラクタ

        Args:
            class_name (str)            : クラス
            bbox (np.ndarray)           : 外接矩形(BoundingBox)[xmin,ymin,xmax,ymax]
            score (float)               : 信頼度conf
            is_det_cur (bool, optional) : 今周期検出済みフラグ. Defaults to True.
        """
        self.class_name_ = class_name
        self.bbox_       = bbox # [xmin,ymin,xmax,ymax]
        self.score_      = score
        self.is_det_cur_ = is_det_cur # 今周期検出済フラグ

        self.bbox_w_:int  = bbox[2] - bbox[0]
        self.bbox_h_:int  = bbox[3] - bbox[1]
        self.bbox_area_     = self.bbox_w_ * self.bbox_h_
        self.bbox_ave_size_ = math.sqrt(float(self.bbox_area_))
        return
    
    def getBboxCenter(self) -> np.ndarray:
        return np.array([(self.bbox_[0] + self.bbox_[2])/2, 
                         (self.bbox_[1] + self.bbox_[3])/2])

    def __str__(self) -> str:
        return f"({self.class_name_},{self.bbox_},{self.score_},{self.is_det_cur_})"

    @staticmethod
    def calcOverlapAreaBBox(bbox1:np.ndarray, bbox2:np.ndarray) -> float:
        """ 外接矩形の重なり部分の面積を算出（ない場合は0）

        Args:
            bbox1 (np.ndarray): 外接矩形1(BoundingBox1)
            bbox2 (np.ndarray): 外接矩形2(BoundingBox2)

        Returns:
            float: 外接矩形1,2の重なり部分の面積[px]
        """
        # 重なり部分の面積を算出（ない場合は0）
        bbox_overlap_area = 0.0
        bbox_inter = np.concatenate([np.maximum(bbox1[:2], bbox2[:2]), np.minimum(bbox1[2:], bbox2[2:])])

        if (bbox_inter[0] < bbox_inter[2]) and (bbox_inter[1] < bbox_inter[3]):
            bbox_overlap_area = float((bbox_inter[2] - bbox_inter[0]) * (bbox_inter[3] - bbox_inter[1]))

        return bbox_overlap_area

class AnnoData:
    def __init__(self, class_name:str, bbox:np.ndarray):
        self.class_name_ = class_name
        self.bbox_       = bbox # [xmin,ymin,xmax,ymax]

        self.bbox_area_:int  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.bbox_ave_size_  = math.sqrt(float(self.bbox_area_))
        return
    
    def extractPositiveDetResult(self, det_results:List[DetResult], jaccard_thres=0.5) -> Tuple[DetResult,float]:
        # 正検出（jaccard係数＞thres の検知）を抽出
        #   複数存在する場合は最大値の結果を抽出
        #   存在しない場合はNoneを返す
        jaccard_max              = jaccard_thres
        det_result_max:DetResult = None

        if len(det_results) > 0:
            det_bbox     = np.array([det.bbox_ for det in det_results])
            jaccard_vals = jaccard_numpy(det_bbox, self.bbox_)

            for jaccard_val, det in zip(jaccard_vals, det_results):
                if jaccard_val > jaccard_max:
                    jaccard_max    = jaccard_val
                    det_result_max = det

        return (det_result_max, jaccard_max)


def getAnnoData(anno_file:str, parse_anno:Anno_xml2list, voc_classes:List[str], img_w:int, img_h:int) -> List[AnnoData]:
    # anno_list: [[xmin, ymin, xmax, ymax, label_ind], ... ]
    anno_list = parse_anno(anno_file, img_w, img_h)

    ret_results:List[AnnoData] = []

    for anno_data in anno_list:
        label_name = voc_classes[int(anno_data[4])]
        bb_org = np.array(anno_data[:4]) * np.array([img_w, img_h, img_w, img_h])
        bb_i   = bb_org.astype(np.int64)

        ret_results.append(AnnoData(label_name, bb_i))

    return ret_results

class DrawPen:
    def __init__(self, col:Tuple[int], thick:int, char_size:float):
        self.col_       = col
        self.thick_     = thick
        self.char_size_ = char_size
        return

def clipCoord(coord_val:int, coord_min:int, coord_max:int) -> int:
    """ 座標値のクリップ

    Args:
        coord_val (int): 座標値
        coord_min (int): 最小値
        coord_max (int): 最大値

    Returns:
        int: 座標値（クリップ後）
    """    
    if coord_val < coord_min:
        ret_val = coord_min
    elif coord_val > coord_max:
        ret_val = coord_max
    else:
        ret_val = coord_val

    return ret_val

class ImageProc:
    """ 画像処理（クリップ、ぼかし、文字描画等） """
    def __init__(self, lu_x=0, lu_y=0, rb_x=0, rb_y=0):
        self.init(lu_x, lu_y, rb_x, rb_y)
        return 

    def init(self, lu_x=0, lu_y=0, rb_x=0, rb_y=0, img:np.ndarray=None):
        """ コンストラクタ

        Args:
            lu_x (int, optional)        : 矩形左上x. Defaults to 0.
            lu_y (int, optional)        : 矩形左上y. Defaults to 0.
            rb_x (int, optional)        : 矩形右下x. Defaults to 0.
            rb_y (int, optional)        : 矩形右下y. Defaults to 0.
            img (np.ndarray, optional)  : 画像. Defaults to None.
        """
        self.darea_lu_x_ = lu_x
        self.darea_lu_y_ = lu_y
        self.darea_rb_x_ = rb_x
        self.darea_rb_y_ = rb_y
        self.darea_w_ = rb_x - lu_x + 1
        self.darea_h_ = rb_y - lu_y + 1
        if img is not None:
            (self.img_h_, self.img_w_, _) = img.shape
        else:
            self.img_w_ = 0
            self.img_h_ = 0

        self.is_no_proc_ = False
        if self.darea_lu_x_ == 0 and \
            self.darea_lu_y_ == 0 and \
            self.darea_rb_x_ == 0 and \
            self.darea_rb_y_ == 0:

            self.is_no_proc_ = True
        else:
            if self.img_w_ > 0 and self.img_h_ > 0:
                self.darea_lu_x_ = clipCoord(self.darea_lu_x_, 0, self.img_w_-1)
                self.darea_lu_y_ = clipCoord(self.darea_lu_y_, 0, self.img_h_-1)
                self.darea_rb_x_ = clipCoord(self.darea_rb_x_, 0, self.img_w_-1)
                self.darea_rb_y_ = clipCoord(self.darea_rb_y_, 0, self.img_h_-1)
        return

    def initFromDet(self, img:np.ndarray, det:DetResult, min_size:int):
        """ コンストラクタ（検出結果から生成）

        Args:
            img (np.ndarray): 画像
            det (DetResult) : 検出結果
            min_size (int)  : 矩形最小サイズ[px]
        """
        if det.bbox_w_ > min_size and det.bbox_h_ > min_size:
            self.init(det.bbox_[0], det.bbox_[1], det.bbox_[2], det.bbox_[3], img)

        # ※未使用（comment out） min_sizeになるよう範囲拡大
        # else:
        #     
        #     min_size_half = int(min_size / 2)
        #     (img_h, img_w, _) = img.shape

        #     cx = int((det.bbox_[0] + det.bbox_[2]) / 2)
        #     cy = int((det.bbox_[1] + det.bbox_[3]) / 2)
        #     if cx - min_size_half < 0:
        #         cx = min_size_half
        #     elif cx + min_size_half >= img_w:
        #         cx = img_w - min_size_half
        #     else:
        #         pass
        #     if cy - min_size_half < 0:
        #         cy = min_size_half
        #     elif cy + min_size_half >= img_h:
        #         cy = img_h - min_size_half
        #     else:
        #         pass

        #     lu_x = cx - min_size_half
        #     lu_y = cy - min_size_half
        #     rb_x = lu_x + min_size
        #     rb_y = lu_y + min_size
        #     self.init(lu_x, lu_y, rb_x, rb_y, img)

        return

    def clip(self, img:np.ndarray) -> np.ndarray:
        """ クリップ

        Args:
            img (np.ndarray): 画像

        Returns:
            np.ndarray: クリップ後の画像
        """
        (self.img_h_, self.img_w_, _) = img.shape

        if self.is_no_proc_ == False:
            return copy.deepcopy(img[self.darea_lu_y_:self.darea_rb_y_, self.darea_lu_x_:self.darea_rb_x_])
        else:
            return copy.deepcopy(img)

    @staticmethod
    def clipBBox(img_w:int, img_h:int, bbox:np.ndarray):

        if bbox[0] < 0:
            bbox[0] = 0
        elif bbox[0] >= img_w:
            bbox[0] = img_w - 1

        if bbox[2] < 0:
            bbox[2] = 0
        elif bbox[2] >= img_w:
            bbox[2] = img_w - 1

        if bbox[1] < 0:
            bbox[1] = 0
        elif bbox[1] >= img_h:
            bbox[1] = img_h - 1

        if bbox[3] < 0:
            bbox[3] = 0
        elif bbox[3] >= img_h:
            bbox[3] = img_h - 1

        return

    def convBBox(self, bbox:np.ndarray) -> np.ndarray:
        """ 外接矩形（BoundingBox）を正規座標→画像座標に変換

        Args:
            bbox (np.ndarray): 外接矩形（正規座標）※x,yの値域が0～1

        Returns:
            np.ndarray: 外接矩形（画像座標）※x,yの値域が0～画像サイズ(w,h)
        """
        area_w = self.img_w_
        area_h = self.img_h_
        if self.is_no_proc_ == False:
            area_w = self.darea_w_
            area_h = self.darea_h_
        
        # 検出範囲の幅、高さを掛ける
        bbox = bbox * [area_w, area_h, area_w, area_h] 

        # 検出範囲の左上座標を加える
        bb_f = np.array([bbox[0] + float(self.darea_lu_x_), 
                         bbox[1] + float(self.darea_lu_y_), 
                         bbox[2] + float(self.darea_lu_x_), 
                         bbox[3] + float(self.darea_lu_y_)])
        bb_i = bb_f.astype(np.int64)

        ImageProc.clipBBox(self.img_w_, self.img_h_, bb_i)
        return bb_i


    @staticmethod
    def isValidBBox(img_org:np.ndarray, bbox:np.ndarray) -> bool:
        (img_h, img_w, _) = img_org.shape

        is_valid = True

        if (bbox[0] < 0) or (img_w <= bbox[0]) or (bbox[2] < 0) or (img_w <= bbox[2]):
            is_valid = False
        elif (bbox[1] < 0) or (img_h <= bbox[1]) or (bbox[3] < 0) or (img_h <= bbox[3]):
            is_valid = False
        else:
            is_valid = True
        return is_valid

    def drawDetArea(self, img_org:np.ndarray, pen:DrawPen, area_name:str="det area") -> np.ndarray:
        """ 検出範囲の矩形を描画

        Args:
            img_org (np.ndarray)     : 画像（描画前）
            pen (DrawPen)            : 色、線の太さ等
            area_name (str, optional): 矩形の名前. Defaults to "det area".

        Returns:
            np.ndarray: 画像（描画後）
        """
        if self.is_no_proc_ == False:
            # 検出範囲を描画
            cv2.rectangle(img_org, 
                          (self.darea_lu_x_, self.darea_lu_y_), 
                          (self.darea_rb_x_, self.darea_rb_y_), 
                           pen.col_, pen.thick_)
            ImageProc.drawText(img_org, area_name, 
                               (self.darea_lu_x_, self.darea_rb_y_+15), 
                               pen.char_size_, pen.col_, pen.thick_, False)
        return img_org

    def eraseRectArea(self, img_org:np.ndarray, is_blur:bool) -> np.ndarray:
        """ 領域消去（inpainting）

        Args:
            img_org (np.ndarray): 画像（処理前）
            is_blur (bool)      : 消去後の領域内をぼかすかどうか

        Returns:
            np.ndarray: 画像（処理後）
        """
        if self.is_no_proc_ == False:
            img_mask = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            img_mask[:,:] = 0
            img_mask[self.darea_lu_y_: self.darea_rb_y_, self.darea_lu_x_: self.darea_rb_x_] = 255
            # cv2.imwrite("mask_img.png", img_mask)

            img_org = cv2.inpaint(img_org, img_mask, 3, cv2.INPAINT_TELEA)

            if is_blur == True:
                # 消去後、ぼかしを入れる
                blur_kernel_size = int(min(self.darea_rb_y_ - self.darea_lu_y_, self.darea_rb_x_ - self.darea_lu_x_) / 4) 
                if blur_kernel_size < 5:
                    blur_kernel_size = 5

                s_roi = img_org[self.darea_lu_y_: self.darea_rb_y_, self.darea_lu_x_: self.darea_rb_x_]
                s_roi = cv2.blur(s_roi, (blur_kernel_size, blur_kernel_size))
                img_org[self.darea_lu_y_: self.darea_rb_y_, self.darea_lu_x_: self.darea_rb_x_] = s_roi

        return img_org

    def __str__(self) -> str:
        ret_str = f"{self.darea_lu_x_}_{self.darea_lu_y_}_{self.darea_w_}x{self.darea_h_}"
        return ret_str

    @staticmethod
    def drawResultSummary(img_org:np.ndarray, frame_no:int, frame_num:int, dev_type:str, net_type:str, time_proc_sec:float, pen:DrawPen) -> np.ndarray:
        """ 結果（概要）を描画

        Args:
            img_org (np.ndarray) : 画像（描画前）
            frame_no (int)       : frame番号
            frame_num (int)      : frame数
            dev_type (str)       : デバイス（cpu or cuda)
            net_type (str)       : SSD種別（vgg or mobilenet）
            time_proc_sec (float): 処理時間[sec]
            pen (DrawPen)        : 色、線の太さ等

        Returns:
            np.ndarray: 画像（描画後）
        """
        # FPS等を描画
        str_frame = ""
        if frame_num > 0:
            str_frame = f"{frame_no:05}/{frame_num:05} "
        
        str_fps = f"net:{net_type} dev:{dev_type} fps={(1.0 / time_proc_sec):.1f}"

        ImageProc.drawText(img_org, str_frame + str_fps, (10, 15), pen.char_size_, pen.col_, pen.thick_, True)
        
        return img_org

    @staticmethod
    def drawResultDet(img_org:np.ndarray, det_results:List[DetResult], pen:DrawPen) -> np.ndarray:
        """ 検出結果（枠等）を描画

        Args:
            img_org (np.ndarray)         : 画像（描画前）
            det_results (List[DetResult]): 検出結果
            pen (DrawPen)                : 色、線の太さ等

        Returns:
            np.ndarray: 画像（描画後）
        """
        for det in det_results:
            display_txt = f"{det.class_name_}:{det.score_:.2f}"

            # 検出結果を描画
            cv2.rectangle(img_org, (det.bbox_[0], det.bbox_[1]), 
                                   (det.bbox_[2], det.bbox_[3]), 
                                   pen.col_, pen.thick_*2)
            
            loc_txt_y = det.bbox_[1]-7
            if loc_txt_y < 10:
                loc_txt_y = det.bbox_[3]-7
            
            ImageProc.drawText(img_org, display_txt, 
                                        (det.bbox_[0], loc_txt_y), 
                                        pen.char_size_, pen.col_, pen.thick_, True)

        return img_org

    @staticmethod
    def drawAnnoData(img_org:np.ndarray, anno_data:List[AnnoData], pen:DrawPen) -> np.ndarray:
        """ アノテーション結果を描画

        Args:
            img_org (np.ndarray)        : 画像（描画前）
            anno_data (List[AnnoData])  : アノテーション
            pen (DrawPen)               : 色、線の太さ等

        Returns:
            np.ndarray: 画像（描画後）
        """
        alpha = 0.6
        img_cp = img_org.copy()
        
        for anno in anno_data:
            display_txt = f"{anno.class_name_}"

            # BBoxを描画
            cv2.rectangle(img_cp, (anno.bbox_[0], anno.bbox_[1]), 
                                   (anno.bbox_[2], anno.bbox_[3]), 
                                   pen.col_, pen.thick_*2)

            ImageProc.drawText(img_cp, display_txt, 
                                       (anno.bbox_[0], anno.bbox_[3]+10), 
                                       pen.char_size_, pen.col_, pen.thick_, True)

        img_org = cv2.addWeighted(img_cp, alpha, img_org, 1.0-alpha, 0)

        return img_org

    @staticmethod
    def drawText(img:np.ndarray, text:str, loc:cv2.typing.Point, scale:float, col:cv2.typing.Scalar, thick:int, is_bound:bool):
        """ 文字描画

        Args:
            img (np.ndarray)        : 画像
            text (str)              : 文字
            loc (cv2.typing.Point)  : 描画位置
            scale (float)           : サイズ
            col (cv2.typing.Scalar) : 色
            thick (int)             : 線の太さ
            is_bound (bool)         : 縁取りの有無
        """
        if is_bound == True:
            #col_bg = [255-c for c in col]
            if (col[0] + col[1] + col[2]) > (255*3/2):
                col_bg = [0,0,0]
            else:
                col_bg = [255,255,255]
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col_bg, thick*3, cv2.LINE_AA)

        cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        return
    
    @staticmethod
    def blurDetObject(img_org:np.ndarray, det_results:List[DetResult], blur_kernel_size:int) -> np.ndarray:
        """ 検出結果の枠内にぼかしを入れる

        Args:
            img_org (np.ndarray)         : 画像（処理前）
            det_results (List[DetResult]): 検出結果
            blur_kernel_size (int)       : ぼかす強度(blurカーネルサイズ)

        Returns:
            np.ndarray: 画像（処理後）
        """
        for det in det_results:
            if (True == ImageProc.isValidBBox(img_org, det.bbox_)) and \
               (det.bbox_[2] - det.bbox_[0] > 0) and (det.bbox_[3] - det.bbox_[1] > 0):

                s_roi = img_org[det.bbox_[1]: det.bbox_[3], det.bbox_[0]: det.bbox_[2]]
                # print(str(det.bbox_[2] - det.bbox_[0]), str(det.bbox_[3] - det.bbox_[1]), s_roi.shape)
                # print(f"({det.bbox_[0]},{det.bbox_[1]}) - ({det.bbox_[2]},{det.bbox_[3]}) s_roi.shape={s_roi.shape}")

                s_roi = cv2.blur(s_roi, (blur_kernel_size, blur_kernel_size))
                img_org[det.bbox_[1]: det.bbox_[3], det.bbox_[0]: det.bbox_[2]] = s_roi

        return img_org

def calcIndexesFromBatchIdx(img_procs:List[List[ImageProc]], batch_idx:int) -> Tuple[int,int]:
    """ バッチindexから、画像index, 検出範囲indexを算出

    Args:
        img_procs (List[List[ImageProc]]) : 検出範囲（複数） [img_num, area_num, 検出範囲]
        batch_idx (int)                   : バッチindex

    Returns:
        Tuple[int,int]: (画像index, 検出範囲index)
    """    
    img_idx  = -1
    area_idx = -1
    batch_no = 0
    for img_no, img_procs_in_img in enumerate(img_procs):
        for area_no in range(len(img_procs_in_img)):
            if batch_no == batch_idx:
                img_idx  = img_no
                area_idx = area_no
                break
            batch_no += 1
        else:
            continue # 内側ループが正常に抜けたときは、外側ループを続ける
        break # 内側ループがbreakで抜けたときのみ、外側ループも抜ける

    return (img_idx, area_idx)


class MovieLoader:
    """ 動画読み込み
    """
    def __init__(self, movie_fpath:str, play_fps:float, num_batch_frame:int):
        """ コンストラクタ

        Args:
            movie_fpath (str)    : 動画ファイルパス
            play_fps (float)     : 再生速度(fps)
            num_batch_frame (int): 一度に読み込む（バッチ処理する）フレーム数[frame/cycle]
        """
        # 入力動画読み込み
        self.cap_ = cv2.VideoCapture(movie_fpath)

        self.num_cap_frame_ = int(self.cap_.get(cv2.CAP_PROP_FRAME_COUNT))

        # 再生速度の設定
        cap_fps = self.cap_.get(cv2.CAP_PROP_FPS)
        self.play_fps_ = cap_fps
        if play_fps > 0.0:
            self.play_fps_ = play_fps

        self.frame_play_step_ = int((cap_fps + 0.1) / self.play_fps_)
        if self.frame_play_step_ < 1:
            self.frame_play_step_ = 1

        # バッチ処理するフレーム数
        self.num_batch_frame_ = num_batch_frame

        # 現在のフレーム番号
        self.cur_frame_no_ = 0
        return

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[int], List[np.ndarray]]:
        """ 画像読み込み

        Raises:
            StopIteration: iteration終了

        Returns:
            Tuple[List[int], List[np.ndarray]]: (frame番号, 読み込んだ画像) ※バッチ処理数分のリスト
        """
        ret_batch_frame_nos:List[int]   = []
        ret_batch_imgs:List[np.ndarray] = []

        # 動画末尾フレームまで再生していたらiteration終了
        if self.cur_frame_no_ >= self.num_cap_frame_:
            raise StopIteration()

        # フレーム読み込み
        self.cur_frame_no_ = int(self.cap_.get(cv2.CAP_PROP_POS_FRAMES)) 
        while (len(ret_batch_frame_nos) < self.num_batch_frame_) and (self.cur_frame_no_ < self.num_cap_frame_):

            img_org:np.ndarray = None
            (_, img_org) = self.cap_.read()

            if (img_org is not None) and (self.cur_frame_no_ % self.frame_play_step_ == 0):
                ret_batch_frame_nos.append(self.cur_frame_no_)
                ret_batch_imgs.append(copy.deepcopy(img_org))

            self.cur_frame_no_ = int(self.cap_.get(cv2.CAP_PROP_POS_FRAMES))

        # 動画フレーム画像、フレーム番号を返す（バッチ処理分のリスト）
        return (ret_batch_frame_nos, ret_batch_imgs)

    def __len__(self) -> int:
        num_iter = int(self.num_cap_frame_ // (self.num_batch_frame_ * self.frame_play_step_))
        return num_iter

    def getNumFrame(self) -> int:
        return self.num_cap_frame_

    def getFrameSize(self) -> Tuple[int,int]:
        frame_w = int(self.cap_.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (frame_w, frame_h)

    def getPlayFps(self) -> float:
        return self.play_fps_



class VocDataSetMng:
    """ VOCデータセット管理
    """

    def __init__(self, data_path:str, voc_classes:List[str], input_size:int, color_mean:List[int], color_std:float, batch_size:int, test_rate:float):
        """ コンストラクタ(データローダー作成)

        Args:
            data_path (str)         : データ格納ディレクトリ
            voc_classes (List[str]) : クラス
            input_size (int)        : SSDの入力画像サイズ(=300固定)
            color_mean (List[int])  : 色の平均値
            color_std (float)       : 色の標準偏差
            batch_size (int)        : バッチサイズ
            test_rate (float)       : 検証用画像の割合(全体の何割を検証用画像として使うか)
        """
        self.batch_size_ = batch_size
        self.test_rate_  = test_rate
        self.batch_size_val_ = int(float(batch_size) * test_rate)

        if self.batch_size_val_ < 3:
            self.batch_size_val_ = 3

        # データのリストを取得
        self.data_path_ = data_path

        filename_list_all = [os.path.split(f)[1].split(".")[0] for f in glob.glob(f"{data_path}/*.xml")]

        # objectがないxmlをファイルリストから除外(difficult設定されているobjectはカウント対象外)
        #  ∵VOCDataset::pull_item()のanno_list[:, :4]で落ちる
        parse_anno = Anno_xml2list(voc_classes)
        filename_list = [f for f in filename_list_all if parse_anno.isExistObject(f"{data_path}/{f}.xml") == True]

        (filename_list_train, filename_list_val) = train_test_split(filename_list, test_size=test_rate)

        train_img_list  = [f"{data_path}/{f}.jpg" for f in filename_list_train]
        train_anno_list = [f"{data_path}/{f}.xml" for f in filename_list_train]
        val_img_list    = [f"{data_path}/{f}.jpg" for f in filename_list_val]
        val_anno_list   = [f"{data_path}/{f}.xml" for f in filename_list_val]

        self.voc_classes_ = voc_classes
        self.input_size_  = input_size

        self.train_dataset_ = VOCDataset(train_img_list, 
                                        train_anno_list, 
                                        phase="train", 
                                        transform=DataTransform(input_size, color_mean, color_std), 
                                        transform_anno=parse_anno)
        self.val_dataset_   = VOCDataset(val_img_list, 
                                        val_anno_list, 
                                        phase="val", 
                                        transform=DataTransform(input_size, color_mean, color_std), 
                                        transform_anno=parse_anno)

        # DataLoaderを作成する
        train_dataloader = data.DataLoader(self.train_dataset_, 
                                        batch_size=self.batch_size_, 
                                        shuffle=True, 
                                        collate_fn=od_collate_fn)
        val_dataloader   = data.DataLoader(self.val_dataset_, 
                                        batch_size=self.batch_size_val_, 
                                        shuffle=False, 
                                        collate_fn=od_collate_fn)

        # 辞書オブジェクトにまとめる
        self.dataloaders_dict_ = {"train": train_dataloader, "val": val_dataloader}

        # DataSetの物体数などを集計
        self.objset_info_train_ = self.calcAnnoObjInfo("train")
        self.objset_info_val_   = self.calcAnnoObjInfo("val")

        return
    
    def calcAnnoObjInfo(self, phase:str) -> Dict[str,Dict[str,any]]:
        """ アノテーション情報を集計（クラス毎の物体数、平均サイズ）

        Args:
            phase (str): train or val

        Returns:
            Dict[str,Dict[str,any]]: 集計結果
        """
        obj_info:Dict[str,any] = {"num":0, "w_ave":0.0, "h_ave":0.0}
        objset_info = {}
        for cls_name in self.voc_classes_:
            objset_info[cls_name] = copy.deepcopy(obj_info)

        target_dataset:VOCDataset = None

        if phase == "train":
            target_dataset = self.train_dataset_
        else:
            target_dataset = self.val_dataset_
        
        # 画像枚数分のループ
        img_size_f = float(self.input_size_)

        for data_idx in range(len(target_dataset)):
            (_, anno_list) = target_dataset[data_idx]

            # 1枚の画像内の物体数分のループ
            for anno_data in anno_list:
                # anno_data(1つの物体のbbox, label) [xmin,ymin, xmax,ymax, label_idx]
                anno_data = anno_data * np.array([img_size_f, img_size_f, img_size_f, img_size_f, 1.0])

                obj_w:float    = anno_data[2] - anno_data[0]
                obj_h:float    = anno_data[3] - anno_data[1]
                obj_class_idx  = int(anno_data[4])
                obj_class_name = self.voc_classes_[obj_class_idx]

                objset_info[obj_class_name]["num"] += 1
                objset_info[obj_class_name]["w_ave"] += obj_w
                objset_info[obj_class_name]["h_ave"] += obj_h

        # 集計結果
        for class_name in self.voc_classes_:
            if objset_info[class_name]["num"] > 0:
                objset_info[class_name]["w_ave"] /= float(objset_info[class_name]["num"])
                objset_info[class_name]["h_ave"] /= float(objset_info[class_name]["num"])

        return objset_info

    def getImageNum(self, phase:str) -> int:
        # 画像枚数を取得
        num_image = 0
        if phase == "train":
            num_image = len(self.train_dataset_.img_list)
        else:
            num_image = len(self.val_dataset_.img_list)

        return num_image

    def getAnnoInfo(self, phase:str, class_name:str) -> Tuple[int,float,float]:
        # 物体(正解)情報（数、サイズ）を取得
        num_obj = 0
        w_ave   = 0.0
        h_ave   = 0.0

        if phase == "train":
            num_obj = int(self.objset_info_train_[class_name]["num"])
            w_ave   = float(self.objset_info_train_[class_name]["w_ave"])
            h_ave   = float(self.objset_info_train_[class_name]["h_ave"])
        else:
            num_obj = int(self.objset_info_val_[class_name]["num"])
            w_ave   = float(self.objset_info_val_[class_name]["w_ave"])
            h_ave   = float(self.objset_info_val_[class_name]["h_ave"])

        return (num_obj, w_ave, h_ave)


def iou(box1, box2) -> float:
    """IoU計算"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def compute_ap(recalls, precisions) -> float:
    """11-point interpolationのAP計算（Pascal VOC方式）"""
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def evaluate_map(predictions:Dict[str,List], ground_truths:Dict[str,List], class_names:List[str], iou_threshold=0.5) -> Tuple[float, List[float]]:
    """ mAP算出

    Args:
        predictions (Dict[str,List])    : 検出結果 ※key＝画像IDの辞書
        ground_truths (Dict[str,List])  : 正解 ※key＝画像IDの辞書
        class_names (List[str])         : クラス名
        iou_threshold (float, optional) : 検出結果＝正解とみなすIoU下限閾値. Defaults to 0.5.

    Returns:
        Tuple[float, List[float]]: (mAP, クラス毎のAP)
    """
    aps = []
    for class_name in class_names:
        true_positives = []
        scores = []
        total_gts = 0
        gt_used = {}

        detections = []
        for image_id in predictions:
            preds = [p for p in predictions[image_id] if p[2] == class_name]
            gts = [g for g in ground_truths.get(image_id, []) if g[1] == class_name]
            total_gts += len(gts)
            gt_used[image_id] = [False] * len(gts)

            for p in preds:
                detections.append((image_id, p[0], p[1]))  # (image_id, bbox, score)

        detections.sort(key=lambda x: -x[2])  # スコア順

        for image_id, pred_box, score in detections:
            gts = [g for g in ground_truths.get(image_id, []) if g[1] == class_name]
            ious = [iou(pred_box, gt[0]) for gt in gts]

            max_iou = 0
            max_index = -1
            for idx, iou_val in enumerate(ious):
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_index = idx

            if max_iou >= iou_threshold and not gt_used[image_id][max_index]:
                true_positives.append(1)
                gt_used[image_id][max_index] = True
            else:
                true_positives.append(0)
            scores.append(score)

        if total_gts == 0:
            aps.append(0)
            continue

        tp = np.array(true_positives)
        fp = 1 - tp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP, aps

class CalcMAP:
    """ Mean Average Precision算出 (関数evaluate_map()のラッパークラス)
    """
    def __init__(self, img_list:List[str], voc_classes:List[str], iou_thres:float):
        """ コンストラクタ

        Args:
            img_list (List[str])    : データセット画像名のリスト
            voc_classes (List[str]) : クラス
            iou_thres (float)       : 検出結果＝正解とみなすIoU下限閾値
        """
        img_fnames = [os.path.splitext(os.path.basename(img_fpath))[0] 
                      for img_fpath in img_list]
        
        self.predictions_   = {img:list() for img in img_fnames}
        self.ground_truths_ = {img:list() for img in img_fnames}

        self.voc_classes_   = voc_classes
        self.iou_thres_     = iou_thres
        return
    
    def add(self, img_fpath:str, det_results:List[DetResult], anno_data:List[AnnoData]):
        """ データ追加

        Args:
            img_fpath (str)              : 画像名（ファイルパス）
            det_results (List[DetResult]): 検出結果
            anno_data (List[AnnoData])   : アノテーションデータ
        """
        img_fname = os.path.splitext(os.path.basename(img_fpath))[0]

        det_results_cp = copy.deepcopy(det_results)
        anno_data_cp   = copy.deepcopy(anno_data)

        for det_result in det_results_cp:
            self.predictions_[img_fname].append((det_result.bbox_, det_result.score_, det_result.class_name_))

        for gt in anno_data_cp:
            self.ground_truths_[img_fname].append((gt.bbox_, gt.class_name_))
        return

    def __call__(self) -> Tuple[float, Dict[str,float]]:
        """ mAP算出

        Returns:
            Tuple[float, Dict[str,float]]: (mAP, クラス毎のAP)
        """
        (mAP, per_class_ap) = evaluate_map(self.predictions_, self.ground_truths_,  self.voc_classes_, self.iou_thres_)

        per_class_ap_dict = {cls_name:ap_val for cls_name,ap_val in zip(self.voc_classes_, per_class_ap)}
        return (mAP, per_class_ap_dict)


class SSDModel:
    """ SSDモデル(VGG,mobilenet共通)
    """
    def __init__(self, device:torch.device, voc_classes:List[str], net_type:str):
        """ コンストラクタ

        Args:
            device (torch.device)   : デバイス(cpu or cuda)
            voc_classes (List[str]) : クラス
            net_type (str)          : SSD種別（vgg or mobilenet）
        """
        self.net_type_    = net_type
        self.num_classes_ = len(voc_classes) + 1

        # SSD300(VGGベース)の設定
        self.ssd_vgg_cfg_ = {
            "num_classes"    : self.num_classes_,            # 背景クラスを含めた合計クラス数
            "input_size"     : 300,                          # 画像の入力サイズ
            "bbox_aspect_num": [4, 6, 6, 6, 4, 4],           # 出力するDBoxのアスペクト比の種類
            "feature_maps"   : [38, 19, 10, 5, 3, 1],        # 各sourceの画像サイズ
            "steps"          : [8, 16, 32, 64, 100, 300],    # DBOXの大きさを決める
            "min_sizes"      : [21, 45, 99, 153, 207, 261],  # DBOXの大きさを決める
            "max_sizes"      : [45, 99, 153, 207, 261, 315], # DBOXの大きさを決める
            "aspect_ratios"  : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        }

        if self.net_type_ == "mb2-ssd":
            # (BGR)の色の平均値
            self.color_mean_ = np.array([ssd_mb2_cfg.image_mean[2], 
                                         ssd_mb2_cfg.image_mean[1], 
                                         ssd_mb2_cfg.image_mean[0]])
            self.color_std_ = ssd_mb2_cfg.image_std

            # 画像のinputサイズ
            self.input_size_ = ssd_mb2_cfg.image_size

        else:
            # (BGR)の色の平均値
            self.color_mean_ = np.array([104, 117, 123])
            self.color_std_  = 1.0

            # 画像のinputサイズ
            self.input_size_ = 300

        self.device_      = device
        self.voc_classes_ = voc_classes

        random_seed = 1234

        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        np.random.seed(random_seed)
        random.seed(random_seed)

        return

def makeVocClassesTxtFpath(weight_fpath:str) -> str:
    voc_classes_fpath = f"./weights/{os.path.splitext(os.path.basename(weight_fpath))[0]}_classes.txt"
    return voc_classes_fpath

class Logger:
    """ ログ生成
    """
    def __init__(self, is_out:bool):
        self.is_out_  = is_out
        self.outdir_  = ""
        self.log_fp_:TextIOWrapper = None
        return

    @staticmethod
    def createOutputDir(dev_name:str, net_type:str, dir_name:str) -> str:
        """ ログ出力ディレクトリ生成（デバイス,SSD種別をディレクトリ名に付与）

        Args:
            dev_name (str): デバイス(cpu or cuda)
            net_type (str): SSD種別（vgg or mobilenet）
            dir_name (str): ディレクトリ名

        Returns:
            str: 出力ディレクトリ名
        """
        if dir_name != "":
            outdir_path = f"./output.{dev_name}.{net_type}/{dir_name}"
        else:
            outdir_path = f"./output.{dev_name}.{net_type}"

        if os.path.isdir(outdir_path) == False:
            os.makedirs(outdir_path)
        return outdir_path

    def openLogFile(self, dev_name:str, net_type:str, dir_name:str, f_name:str, mode:str):
        """ ログファイルopen（デバイス,SSD種別をディレクトリ名に付与）

        Args:
            dev_name (str)  : デバイス(cpu or cuda)
            net_type (str)  : SSD種別（vgg or mobilenet）
            dir_name (str)  : ディレクトリ名
            f_name (str)    : ログファイル名
            mode (str)      : w or r
        """
        if self.is_out_ == True:
            self.outdir_ = Logger.createOutputDir(dev_name, net_type, dir_name)
            self.log_fp_ = open(f"{self.outdir_}/{f_name}", mode)
        return

    def closeLogFile(self):
        """ ログファイルclose
        """
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            self.log_fp_.close()
            self.log_fp_ = None
        return

    def isOutputLog(self) -> bool:
        ret = False
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            ret = True
        return ret

class KalmanFilter2D:
    """ 2次元値（例：位置(x,y)）を推定するカルマンフィルタ

      参考: https://qiita.com/matsui_685/items/16b81bf0ad9a24c54e52
    """
    def __init__(self, fps:float):
        """ コンストラクタ

        Args:
            fps (float): frame per second [cycle/sec]
        """
        # 観測値入力有無（一度でも観測値を入力したらTrue）
        self.is_input_measurement_ = False 

        # 計測間隔 [sec/cycle]
        self.dt_ = 1.0 / fps 

        # 推定値: 4次元ベクトル(x, y, dx/dt, dy/dt)
        self.x_ = np.array([[0.], [0.], [0.], [0.]]) 

        # プロセスノイズ
        self.u_ = np.array([[0.], [0.], [0.], [0.]]) 

        # 共分散行列
        self.P_ = np.array([[100.,   0.,   0., 0.],
                            [  0., 100.,   0., 0.],
                            [  0.,   0., 100., 0.], 
                            [  0.,   0.,   0., 100.]]) 

        # 状態遷移行列
        self.F_ = np.array([[1., 0., self.dt_, 0.],
                            [0., 1., 0.,       self.dt_],
                            [0., 0., 1.,       0.],
                            [0., 0., 0.,       1.]])
        # 観測行列
        self.H_ = np.array([[1., 0., 0., 0.], 
                            [0., 1., 0., 0.]])

        # 観測ノイズ
        self.R_ = np.array([[0.1, 0.],
                            [0.,  0.1]]) 

        # 4次元単位行列
        self.I_ = np.identity((len(self.x_)))
        return

    def predict(self):
        """ 予測ステップ
        """
        if self.is_input_measurement_ == True:
            self.x_ = np.dot(self.F_, self.x_) + self.u_
            self.P_ = np.dot(np.dot(self.F_, self.P_), self.F_.T)
        return

    def update(self, measurement:np.ndarray):
        """ 更新ステップ

        Args:
            measurement (np.ndarray): 観測値
        """
        if self.is_input_measurement_ == True:

            Z = np.array([measurement])
            y = Z.T - np.dot(self.H_, self.x_)
            S = np.dot(np.dot(self.H_, self.P_), self.H_.T) + self.R_
            K = np.dot(np.dot(self.P_, self.H_.T), np.linalg.inv(S))
            self.x_ = self.x_ + np.dot(K, y)
            self.P_ = np.dot((self.I_ - np.dot(K, self.H_)), self.P_)
        else:
            # 観測値入力初回は、初期状態の設定のみ
            self.x_ = np.array([[measurement[0]], [measurement[1]], [0.], [0.]])
            self.is_input_measurement_ = True

        return

    def resetPredict(self, measurement:np.ndarray):
        """ 推定値(x,y)を観測値で上書き　※変動量成分(dx/dt, dy/dt)はリセットしない

        Args:
            measurement (np.ndarray): 観測値
        """
        self.x_[0][0] = measurement[0]
        self.x_[1][0] = measurement[1]
        return
    
    def getEstimatedValue(self) -> np.ndarray:
        """ 推定値(x,y)を返す（変動量成分(dx/dt, dy/dt)は返さない）

        Returns:
            np.ndarray: 推定値(x,y)
        """
        return self.x_.reshape((4,))[:2] 

    def getEstimatedStdev(self) -> np.ndarray:
        """ 推定値(x,y)の標準偏差を返す

        Returns:
            np.ndarray: 推定値の標準偏差(std(x), std(y))
        """
        return np.array([math.sqrt(self.P_[0][0]), math.sqrt(self.P_[1][1])]) 

# --- 単体テスト用 ここから ---
import matplotlib.pyplot as plt

def unitTestKalman():
    # https://qiita.com/matsui_685/items/16b81bf0ad9a24c54e52 の実行例
    measurements = np.array([[7., 15.], [8., 14.], [9., 13.], [10., 12.], [11., 11.], [12., 10.]]) #位置[x,y]の計測結果
    initial_xy = np.array([6., 17.]) #初期位置

    fps = 10.0
    kalman_f = KalmanFilter2D(fps)
    kalman_f.update(initial_xy)

    x_est = []
    y_est = []

    for m in measurements:
        kalman_f.predict()
        kalman_f.update(m)
        est = kalman_f.getEstimatedValue()
        x_est.append(est[0])
        y_est.append(est[1])

    print(kalman_f.getEstimatedValue())

    x_measure = [m[0] for m in measurements]
    y_measure = [m[1] for m in measurements]
    plt.plot(x_measure, y_measure)
    plt.plot(x_est, y_est)
    plt.show()

    return

if __name__ == "__main__":
    # 単体テスト: KalmanFilter2D
    unitTestKalman()
