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

class DetResult:
    def __init__(self, class_name:str, bbox:np.ndarray, score:float):
        self.class_name_ = class_name
        self.bbox_       = bbox # [xmin,ymin,xmax,ymax]
        self.score_      = score

        self.bbox_area_     = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.bbox_ave_size_ = math.sqrt(self.bbox_area_)
        return

class AnnoData:
    def __init__(self, class_name:str, bbox:np.ndarray):
        self.class_name_ = class_name
        self.bbox_       = bbox # [xmin,ymin,xmax,ymax]

        self.bbox_area_     = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.bbox_ave_size_ = math.sqrt(self.bbox_area_)
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

class DrawPen:
    def __init__(self, col:Tuple[int], thick:int, char_size:float):
        self.col_       = col
        self.thick_     = thick
        self.char_size_ = char_size
        return

class ImageProc:

    def __init__(self, lu_x:int=0, lu_y:int=0, rb_x:int=0, rb_y:int=0):
        self.darea_lu_x_ = lu_x
        self.darea_lu_y_ = lu_y
        self.darea_rb_x_ = rb_x
        self.darea_rb_y_ = rb_y
        self.darea_w_ = rb_x - lu_x + 1
        self.darea_h_ = rb_y - lu_y + 1
        self.img_w_ = 0
        self.img_h_ = 0

        self.draw_col_darea_  = (0,128,0)

        self.is_no_proc_ = False
        if self.darea_lu_x_ == 0 and \
            self.darea_lu_y_ == 0 and \
            self.darea_rb_x_ == 0 and \
            self.darea_rb_y_ == 0:

            self.is_no_proc_ = True
        return 

    def clip(self, img:np.ndarray) -> np.ndarray:  
        (self.img_h_, self.img_w_, _) = img.shape

        if self.is_no_proc_ == False:
            return img[self.darea_lu_y_:self.darea_rb_y_, self.darea_lu_x_:self.darea_rb_x_]
        else:
            return copy.deepcopy(img)
    
    def convBBox(self, bbox:np.ndarray) -> np.ndarray:
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
        return bb_i

    def getAnnoData(self, anno_file:str, parse_anno:Anno_xml2list, voc_classes:List[str], img_w:int, img_h:int) -> List[AnnoData]:
        # anno_list: [[xmin, ymin, xmax, ymax, label_ind], ... ]
        anno_list = parse_anno(anno_file, img_w, img_h)

        ret_results:List[AnnoData] = []

        for anno_data in anno_list:
            label_name = voc_classes[int(anno_data[4])]
            bb_org = np.array(anno_data[:4]) * np.array([img_w, img_h, img_w, img_h])
            bb_i   = bb_org.astype(np.int64)

            ret_results.append(AnnoData(label_name, bb_i))

        return ret_results
                     
    def drawResultSummary(self, img_org:np.ndarray, frame_no:int, frame_num:int, dev_type:str, time_proc_sec:float, pen:DrawPen) -> np.ndarray:
        # FPS等を描画
        str_frame = ""
        if frame_num > 0:
            str_frame = f"{frame_no:05}/{frame_num:05} "
        
        str_fps = f"fps={(1.0 / time_proc_sec):.1f} dev:{dev_type}"

        # str_fps = "F{:05}".format(frame_no) + "/{:05}".format(frame_num) + " fps={:.1f}".format(1.0 / time_proc_sec) + " dev:" + dev_type
        ImageProc.drawText(img_org, str_frame + str_fps, (10, 15), pen.char_size_, pen.col_, pen.thick_, True)
        
        return img_org

    def drawResultDet(self, img_org:np.ndarray, det_results:List[DetResult], pen:DrawPen) -> np.ndarray:

        if self.is_no_proc_ == False:
            # 検出範囲を描画
            cv2.rectangle(img_org, (self.darea_lu_x_, self.darea_lu_y_), 
                                (self.darea_rb_x_, self.darea_rb_y_), 
                                self.draw_col_darea_, pen.thick_)
            ImageProc.drawText(img_org, "det area", 
                                        (self.darea_lu_x_, self.darea_rb_y_+15), 
                                        pen.char_size_, self.draw_col_darea_, pen.thick_, False)

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

    def drawAnnoData(self, img_org:np.ndarray, anno_data:List[AnnoData], pen:DrawPen) -> np.ndarray:

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

    def __str__(self) -> str:
        ret_str = f"{self.darea_lu_x_}_{self.darea_lu_y_}_{self.darea_w_}x{self.darea_h_}"
        return ret_str

    @staticmethod
    def drawText(img:np.ndarray, text:str, loc:cv2.typing.Point, scale:float, col:cv2.typing.Scalar, thick:int, is_bound:bool):
        if is_bound == True:
            #col_bg = [255-c for c in col]
            if (col[0] + col[1] + col[2]) > (255*3/2):
                col_bg = [0,0,0]
            else:
                col_bg = [255,255,255]
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col_bg, thick*3, cv2.LINE_AA)

        cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        return


class VocDataSetMng:

    def __init__(self, data_path:str, voc_classes:List[str], color_mean:List[int], input_size:int, batch_size:int, test_rate:float):

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
                                        transform=DataTransform(input_size, color_mean), 
                                        transform_anno=parse_anno)
        self.val_dataset_   = VOCDataset(val_img_list, 
                                        val_anno_list, 
                                        phase="val", 
                                        transform=DataTransform(input_size, color_mean), 
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
        # 物体(正解)情報（数、サイズ）を集計
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


class SSDModel:

    def __init__(self, device:torch.device, voc_classes:List[str]):
        # SSD300の設定
        self.ssd_cfg_ = {
            "num_classes"    : len(voc_classes) + 1,         # 背景クラスを含めた合計クラス数
            "input_size"     : 300,                          # 画像の入力サイズ
            "bbox_aspect_num": [4, 6, 6, 6, 4, 4],           # 出力するDBoxのアスペクト比の種類
            "feature_maps"   : [38, 19, 10, 5, 3, 1],        # 各sourceの画像サイズ
            "steps"          : [8, 16, 32, 64, 100, 300],    # DBOXの大きさを決める
            "min_sizes"      : [21, 45, 99, 153, 207, 261],  # DBOXの大きさを決める
            "max_sizes"      : [45, 99, 153, 207, 261, 315], # DBOXの大きさを決める
            "aspect_ratios"  : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        }
        self.color_mean_ = (104, 117, 123)  # (BGR)の色の平均値
        self.input_size_ = 300              # 画像のinputサイズ        

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
    def __init__(self, is_out:bool):
        self.is_out_  = is_out
        self.outdir_  = ""
        self.log_fp_:TextIOWrapper = None
        return

    @staticmethod
    def createOutputDir(dev_name:str, dir_name:str) -> str:
        if dir_name != "":
            outdir_path = f"./output.{dev_name}/{dir_name}"
        else:
            outdir_path = f"./output.{dev_name}"

        if os.path.isdir(outdir_path) == False:
            os.makedirs(outdir_path)
        return outdir_path

    def openLogFile(self, dev_name:str, dir_name:str, f_name:str, mode:str):
        if self.is_out_ == True:
            self.outdir_ = Logger.createOutputDir(dev_name, dir_name)
            self.log_fp_ = open(f"{self.outdir_}/{f_name}", mode)
        return

    def closeLogFile(self):
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            self.log_fp_.close()
            self.log_fp_ = None
        return

    def isOutputLog(self) -> bool:
        ret = False
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            ret = True
        return ret
