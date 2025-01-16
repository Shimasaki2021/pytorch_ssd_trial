import os
import glob
import random
import copy

import cv2
import numpy as np
from typing import List,Dict,Tuple

import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

from utils.ssd_model import VOCDataset, DataTransform, Anno_xml2list, od_collate_fn

class ImageProc:

    def __init__(self, lu_x:int, lu_y:int, rb_x:int, rb_y:int):
        self.darea_lu_x_ = lu_x
        self.darea_lu_y_ = lu_y
        self.darea_rb_x_ = rb_x
        self.darea_rb_y_ = rb_y

        self.draw_char_col_   = (255,255,255)
        self.draw_char_size_  = 0.6
        self.draw_char_thick_ = 2
        self.draw_col_darea_  = (0,128,0)
        return 

    def clip(self, img:np.ndarray) -> np.ndarray:
        return img[self.darea_lu_y_:self.darea_rb_y_, self.darea_lu_x_:self.darea_rb_x_]

    def drawResultSummary(self, img_org:np.ndarray, frame_no:int, frame_num:int, dev_type:str, time_proc_sec:float) -> np.ndarray:
        # FPS等を描画
        str_fps = "F{:05}".format(frame_no) + "/{:05}".format(frame_num) + " fps={:.1f}".format(1.0 / time_proc_sec) + " dev:" + dev_type
        ImageProc.drawText(img_org, str_fps, (10, 15), self.draw_char_size_, self.draw_char_col_, self.draw_char_thick_, True)
        
        return img_org

    def drawResultDet(self, img_org:np.ndarray, voc_classes:List[str], predict_bbox:List[np.ndarray], pre_dict_label_index:List[int], scores:List[float]) -> np.ndarray:

        # 検出範囲を描画
        cv2.rectangle(img_org, (self.darea_lu_x_, self.darea_lu_y_), 
                               (self.darea_rb_x_, self.darea_rb_y_), 
                               self.draw_col_darea_, self.draw_char_thick_)
        ImageProc.drawText(img_org, "det area", 
                                    (self.darea_lu_x_, self.darea_rb_y_+15), 
                                    self.draw_char_size_, self.draw_col_darea_, self.draw_char_thick_, False)

        for i, bb in enumerate(predict_bbox):
            label_name = voc_classes[pre_dict_label_index[i]]

            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # clip前の画像上での座標値に変換
            bb_org = np.array([bb[0] + float(self.darea_lu_x_), 
                               bb[1] + float(self.darea_lu_y_), 
                               bb[2] + float(self.darea_lu_x_), 
                               bb[3] + float(self.darea_lu_y_)])
            bb_i = bb_org.astype(np.int64)

            # 検出結果を描画
            cv2.rectangle(img_org, (bb_i[0], bb_i[1]), 
                                   (bb_i[2], bb_i[3]), 
                                   self.draw_char_col_, self.draw_char_thick_*2)
            ImageProc.drawText(img_org, display_txt, 
                                        (bb_i[0], bb_i[1]-7), 
                                        self.draw_char_size_, self.draw_char_col_, self.draw_char_thick_, True)

        return img_org

    def toString(self) -> str:
        w = self.darea_rb_x_ - self.darea_lu_x_
        h = self.darea_rb_y_ - self.darea_lu_y_
        ret_str = f"{self.darea_lu_x_}_{self.darea_lu_y_}_{w}x{h}"
        return ret_str

    @staticmethod
    def drawText(img:np.ndarray, text:str, loc:cv2.typing.Point, scale:float, col:cv2.typing.Scalar, thick:int, is_bound:bool):
        if is_bound == True:
            col_bg = [255-c for c in col]
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col_bg, thick*3)
        cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)
        return


class VocDataSetMng:

    def __init__(self, data_path:str, voc_classes:List[str], color_mean:List[int], input_size:int):

        # データのリストを取得
        self.data_path_ = data_path

        filename_list_all = [os.path.split(f)[1].split(".")[0] for f in glob.glob(f"{data_path}/*.xml")]

        # objectがないxmlをファイルリストから除外(difficult設定されているobjectはカウント対象外)
        #  ∵VOCDataset::pull_item()のanno_list[:, :4]で落ちる
        parse_anno = Anno_xml2list(voc_classes)
        filename_list = [f for f in filename_list_all if parse_anno.isExistObject(f"{data_path}/{f}.xml") == True]

        (filename_list_train, filename_list_val) = train_test_split(filename_list, test_size=0.1)

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
                                        batch_size=32, 
                                        shuffle=True, 
                                        collate_fn=od_collate_fn)
        val_dataloader   = data.DataLoader(self.val_dataset_, 
                                        batch_size=3, 
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

        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        return

def makeVocClassesTxtFpath(weight_fpath:str) -> str:
    voc_classes_fpath = f"./weights/{os.path.splitext(os.path.basename(weight_fpath))[0]}_classes.txt"
    return voc_classes_fpath

