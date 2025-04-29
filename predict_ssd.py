#!/usr/bin/env python3

import sys
import cv2
import os
import time
import glob
import copy
from typing import List,Tuple,Dict,Any

from utils.ssd_model import SSD as SSD_vgg
from utils.ssd_model import DataTransform, VOCDataset, Anno_xml2list, nm_suppression
import numpy as np
from tqdm import tqdm
import torch
import torchvision

from common_ssd import ImageProc, MovieLoader, calcIndexesFromBatchIdx
from common_ssd import SSDModel, DetResult, AnnoData, CalcMAP, getAnnoData
from common_ssd import DrawPen, Logger, makeVocClassesTxtFpath

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

# 
class SSDModelDetector(SSDModel):
    """ SSDモデル（検知（推論）用）
    """

    def __init__(self, device:torch.device, net_type:str, weight_fpath:str):
        """ コンストラクタ

        Args:
            device (torch.device): デバイス（cpu or cuda)
            net_type (str)       : SSD種別（vgg or mobilenet）
            weight_fpath (str)   : パラメータ(重み)ファイルのパス
        """
        # 学習済み重みをロード
        (net_weights, voc_classes) = self.loadWeight(weight_fpath, device)

        super().__init__(device, voc_classes, net_type)

        # SSDネットワークモデル
        if self.net_type_ == "mb2-ssd":
            # mobilenet-v2-liteベースSSD
            net_body  = create_mobilenetv2_ssd_lite(self.num_classes_, is_test=True, device=device)
            net_body.load_state_dict(net_weights)
            self.net_ = create_mobilenetv2_ssd_lite_predictor(net_body, candidate_size=200, device=device)
        else:
            # VGGベースSSD
            self.net_ = SSD_vgg(phase="inference", cfg=self.ssd_vgg_cfg_)
            self.net_.load_state_dict(net_weights)
            self.net_.eval()
            self.net_.to(self.device_)

        # 前処理を行うクラス(DataTransform)のインスタンス作成
        self.transform_ = DataTransform(self.input_size_, self.color_mean_, self.color_std_)

        # 入力画像dummy
        self.img_dummy_ = np.zeros((self.input_size_, self.input_size_, 3), dtype=np.uint8) 

        # (torch.compile) Windows+anaconda環境では使用不可。WSL2+Ubuntu環境では使用可。
        #   GPU=GTX1660SUPERでは、以下Warningが出るが一応実行は可能。効果はわずか（1分の動画の処理時間が、12分7秒 → 11分57秒）
        #     W0316 07:20:49.853000 15169 torch/_inductor/utils.py:1137] [0/0_1] Not enough SMs to use max_autotune_gemm mode
        # self.net_ = torch.compile(self.net_, mode="default")
        return
    
    def loadWeight(self, weight_fpath:str, device:torch.device) -> Tuple[Any, List[str]]:
        """ パラメータ(重み)ファイルのロード（クラスファイルもロード）

        Args:
            weight_fpath (str)   : パラメータ(重み)ファイルのパス
            device (torch.device): デバイス（cpu or cuda)

        Returns:
            Tuple[Any, List[str]]: (パラメータ(重み)データ、クラス）
        """
        # ネットワーク重みをロード
        net_weights = torch.load(weight_fpath, weights_only=True, map_location=device) # FutureWarning: You are using torch.load..対策
        # net_weights = torch.load(weight_fpath, weights_only=True) # FutureWarning: You are using torch.load..対策

        # クラス名リスト（voc_classes）をロード
        voc_classes:List[str] = []

        voc_classes_fpath = makeVocClassesTxtFpath(weight_fpath)
        voc_classes_fp = open(voc_classes_fpath,"r")
        if voc_classes_fp is not None:
            voc_classes = [name.strip() for name in voc_classes_fp.readlines()]
            voc_classes_fp.close()

        return (net_weights, voc_classes)

    def transImage(self, img:np.ndarray) -> np.ndarray:
        """ 画像前処理

        Args:
            img (np.ndarray): 画像（前処理前）

        Returns:
            np.ndarray: 画像（前処理後）
        """
        (img_trans, _ ,_) = self.transform_(img, "val", "", "")
        return img_trans
    
    def predict(self, img_procs:List[ImageProc], imgs:List[np.ndarray], conf=0.5, overlap=0.45) -> List[List[DetResult]]:
        """ 検知（推論）実行

        Args:
            img_procs (List[ImageProc]) : 検出領域（複数）[area_num, 検出領域]
            imgs (List[np.ndarray])     : 画像（複数） [img_num,h,w,ch(BGR)]
            conf (float, optional)      : 信頼度conf下限閾値. Defaults to 0.5.
            overlap (float, optional)   : 重複有無の判定閾値(IoU). Defaults to 0.45.

        Returns:
            List[List[DetResult]]: 検出結果（複数） [img_num, obj_num, 検出結果]
        """
        num_img  = len(imgs)

        det_results:List[List[DetResult]] = []
        for img_no in range(num_img):
            det_results.append(list())

        if num_img > 0:
            num_area = len(img_procs)

            # print(f"\nIn predict(): img:{num_img}, area:{num_area}")

            imgs_trans:List[np.ndarray] = []
            for img_org in imgs:
                for img_proc in img_procs:
                    # 検出範囲切り出し
                    img_det = img_proc.clip(img_org)
                    # 画像前処理
                    img_trans = self.transImage(img_det)
                    # batch化（リストに追加（SSDモデルにあうようデータ配置組み替えもあわせて実施））
                    imgs_trans.append(img_trans[:, :, (2, 1, 0)]) # [h,w,ch(BGR→RGB)]
            
            # batch化（リスト→torchテンソル型に変換（SSDモデルにあうようデータ配置組み替えもあわせて実施））
            imgs_trans_np = np.array(imgs_trans)                         # [batch_num, h, w, ch(RGB)]
            img_batch = torch.from_numpy(imgs_trans_np).permute(0,3,1,2) # [batch_num, ch(RGB), h, w]
            # for idx,img in enumerate(img_batch):
            #     torchvision.utils.save_image(img, f"img_batch{idx}.jpg")

            # 入力画像をデバイス(GPU or CPU)に送る
            img_batch = img_batch.to(self.device_)
            # print(f"img_batch = {img_batch.shape}")

            # 推論実行
            torch.backends.cudnn.benchmark = True
            outputs = self.net_(img_batch)

            # SSDモデルの出力を閾値処理（確信度confが閾値以上の結果を取り出し）
            outputs    = outputs.cpu().detach().numpy() # (batch_num, label, top200, [conf,xmin,ymin,xmax,ymax])
            find_index = np.where(outputs[:, :, :, 0] >= conf) # (batch_num, label, top200)
            outputs    = outputs[find_index]

            # 抽出した物体数分ループを回す
            for i in range(len(find_index[1])):  

                batch_no:int  = find_index[0][i]    # batch index
                img_no        = batch_no // num_area # 画像no
                area_no       = batch_no %  num_area # 検出範囲no

                label_no = find_index[1][i] # ラベル(クラス)番号

                if label_no > 0:  
                    # [背景クラスでない場合] 結果を取得

                    # 確信度conf
                    sc = outputs[i][0]
                    # クラス名
                    cls_name = self.voc_classes_[label_no-1]
                    # Bounding Box: 入力画像上での座標値に変換
                    bb_i = img_procs[area_no].convBBox( outputs[i][1:] )

                    det_results[img_no].append(DetResult(cls_name, bb_i, sc))

            if num_area > 1:
                for img_no in range(num_img):
                    # [検出領域が複数の場合] 重複領域での重複枠を取り除く
                    det_results[img_no] = self.nmSuppression(det_results[img_no], overlap)

        return det_results

    def predictDetail(self, det_results:List[List[DetResult]], imgs:List[np.ndarray], min_bbox_size:int, num_batch:int, target_class:str, conf=0.5, overlap=0.45) -> List[List[DetResult]]:
        """ 検知（推論）実行（検出結果の中を詳細検知）

        - 検出結果の矩形内に対してSSDモデルの検知を再度実行

        Args:
            det_results (List[List[DetResult]]) : 検出結果（複数） [img_num, obj_num, 検出結果]
            imgs (List[np.ndarray])             : 画像（複数） [img_num,h,w,ch(BGR)]
            min_bbox_size (int)                 : 矩形最小サイズ[px]
            num_batch (int)                     : バッチ数
            target_class (str)                  : 検知再実行する検出結果クラス
            conf (float, optional)              : 信頼度conf下限閾値. Defaults to 0.5.
            overlap (float, optional)           : 重複有無の判定閾値(IoU). Defaults to 0.45.

        Returns:
            List[List[DetResult]]: 検出結果（複数） [img_num, obj_num, 検出結果]
        """
        num_img  = len(imgs)

        img_procs:List[List[ImageProc]] = []
        for img_no in range(num_img):
            img_procs.append(list())

        if num_img > 0:
            # 検出結果から検出領域を作成（target_classの枠を、検出領域として切り出し）
            num_area = 0
            for img_no, img_org in enumerate(imgs):
                for det_obj in det_results[img_no]:
                    if det_obj.class_name_ == target_class:
                        img_proc = ImageProc()
                        img_proc.initFromDet(img_org, det_obj, min_bbox_size)
                        if img_proc.is_no_proc_ == False:
                            # サイズがmin_bbox_size以上の領域のみ採用
                            img_procs[img_no].append(copy.deepcopy(img_proc))
                            num_area += 1
                            if num_area >= num_batch:
                                break
                else:
                    continue
                break

            # print(f"\nIn predictDetail(): img:{num_img}, area:{num_area}")

            if num_area > 0:

                imgs_trans:List[np.ndarray] = []
                for img_no, img_org in enumerate(imgs):
                    for img_proc in img_procs[img_no]:
                        # 検出範囲切り出し
                        img_det = img_proc.clip(img_org)
                        # 画像前処理
                        img_trans = self.transImage(img_det)
                        # batch化（リストに追加（SSDモデルにあうようデータ配置組み替えもあわせて実施））
                        imgs_trans.append(img_trans[:, :, (2, 1, 0)]) # [h,w,ch(BGR→RGB)]
                
                # GPU実行時に処理時間を安定化させるため、batch数を揃える
                while len(imgs_trans) < num_batch:
                    imgs_trans.append(self.img_dummy_)

                # batch化（リスト→torchテンソル型に変換（SSDモデルにあうようデータ配置組み替えもあわせて実施））
                imgs_trans_np = np.array(imgs_trans)                         # [batch_num, h, w, ch(RGB)]
                img_batch = torch.from_numpy(imgs_trans_np).permute(0,3,1,2) # [batch_num, ch(RGB), h, w]
                # for idx,img in enumerate(img_batch):
                #     torchvision.utils.save_image(img, f"img_batch{idx}.jpg")

                # 入力画像をデバイス(GPU or CPU)に送る
                img_batch = img_batch.to(self.device_)
                # print(f"img_batch = {img_batch.shape}")

                # 推論実行
                torch.backends.cudnn.benchmark = True
                outputs = self.net_(img_batch)

                # SSDモデルの出力を閾値処理（確信度confが閾値以上の結果を取り出し）
                outputs    = outputs.cpu().detach().numpy() # (batch_num, label, top200, [conf,xmin,ymin,xmax,ymax])
                find_index = np.where(outputs[:, :, :, 0] >= conf) # (batch_num, label, top200)
                outputs    = outputs[find_index]

                # 抽出した物体数分ループを回す
                for i in range(len(find_index[1])):  

                    batch_no:int  = find_index[0][i]    # batch index
                    img_no, area_no = calcIndexesFromBatchIdx(img_procs, batch_no)

                    if img_no >= 0 and area_no >= 0:
                        label_no = find_index[1][i] # ラベル(クラス)番号

                        if label_no > 0:  
                            # [背景クラスでない場合] 結果を取得

                            # 確信度conf
                            sc = outputs[i][0]
                            # クラス名
                            cls_name = self.voc_classes_[label_no-1]
                            # Bounding Box: 入力画像上での座標値に変換
                            bb_i = img_procs[img_no][area_no].convBBox( outputs[i][1:] )

                            if cls_name != target_class:
                                # target_class以外の検出結果のみ採用
                                det_results[img_no].append(DetResult(cls_name, bb_i, sc))

                for img_no in range(num_img):
                    # 元の検出枠と重複する枠を取り除く
                    det_results[img_no] = self.nmSuppression(det_results[img_no], overlap)

        return det_results


    def nmSuppression(self, det_results:List[DetResult], iou:float=0.45) -> List[DetResult]:
        """ 重複枠の削除
        Args:
            det_results (List[DetResult]): 検出結果（処理前）
            iou (float, optional)        : 重複有無の判定閾値(IoU). Defaults to 0.45.

        Returns:
            List[DetResult]: 検出結果（処理後）
        """
        #   引数iou以上の重なりがある枠が存在する場合、一番確信度が高い枠のみ残し、それ以外を削除する
        #   異なるクラスの枠同士が重なる場合は対象外
        det_results_sup:List[DetResult] = []

        for cls_name in self.voc_classes_:
            det_results_cls = [x for x in det_results if x.class_name_ == cls_name]

            if len(det_results_cls) > 0:
                bb_i_cls  = torch.from_numpy( np.array([x.bbox_  for x in det_results_cls]) )
                score_cls = torch.from_numpy( np.array([x.score_ for x in det_results_cls]) )

                (ids, count) = nm_suppression(bb_i_cls, score_cls, overlap=iou)
                
                bb_i_cls_sup  = bb_i_cls[ids[:count]].cpu().detach().numpy()
                score_cls_sup = score_cls[ids[:count]].cpu().detach().numpy()

                for bb_i, score in zip(bb_i_cls_sup, score_cls_sup):
                    det_results_sup.append(DetResult(cls_name, bb_i, score))

        return det_results_sup

class LogEvalAnno(Logger):
    """ ログ生成（アノテーション結果）
    """
    def __init__(self, is_out:bool):
        super().__init__(is_out)
        return
    
    def openLogFile(self, dev_name:str, net_type:str, output_imgdir_name:str):
        if self.is_out_ == True:
            super().openLogFile(dev_name, net_type, output_imgdir_name, "eval_anno.csv", "w")

            if self.log_fp_ is not None:
                self.log_fp_.write("img,")
                self.log_fp_.write("ano:cls,ano:bbox,,,,ano:bbox_area,ano:bbox_size,")
                self.log_fp_.write("det:is_exist,det:cls,det:bbox,,,,det:bbox_area,det:bbox_size,det:score,det:jaccard\n")
        return

    def writeLog(self, img_fpath:str, anno_data:AnnoData, pos_det:DetResult, jaccard_val:float):
        if self.isOutputLog() == True:
            w_line  = f"{img_fpath},"
            w_line += f"{anno_data.class_name_},"
            w_line += f"{int(anno_data.bbox_[0])},{int(anno_data.bbox_[1])},{int(anno_data.bbox_[2])},{int(anno_data.bbox_[3])},"
            w_line += f"{int(anno_data.bbox_area_)},{int(anno_data.bbox_ave_size_)},"

            if pos_det is not None:
                w_line += f"True,"
                w_line += f"{pos_det.class_name_},"
                w_line += f"{int(pos_det.bbox_[0])},{int(pos_det.bbox_[1])},{int(pos_det.bbox_[2])},{int(pos_det.bbox_[3])},"
                w_line += f"{int(pos_det.bbox_area_)},{int(pos_det.bbox_ave_size_)},"
                w_line += f"{pos_det.score_:.2f},{jaccard_val:.2f}"

            else:
                w_line += f"False"
            
            self.log_fp_.write(f"{w_line}\n")

        return

def main_play_movie(img_procs:List[ImageProc], num_batch:int, ssd_model:SSDModelDetector, movie_fpath:str, play_fps:float, conf:float, overlap:float):
    """ SSD検知実行（動画から）

    Args:
        img_procs (List[ImageProc]) : 検出領域（複数）[area_num, 検出領域]
        num_batch (int)             : バッチ処理数（1回で処理する画像数 x 検出領域数）
        ssd_model (SSDModelDetector): SSDモデル
        movie_fpath (str)           : 入力動画パス
        play_fps (float)            : 動作再生fps
        conf (float)                : 信頼度conf下限閾値
        overlap (float)             : 重複有無の判定閾値(IoU)
    """
    num_batch_frame = int(num_batch / len(img_procs))
    if num_batch_frame < 1:
        num_batch_frame = 1

    # 画像出力用フォルダ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, ssd_model.net_type_, output_imgdir_name)

    logger = Logger(True)
    logger.openLogFile(ssd_model.device_.type, ssd_model.net_type_, output_imgdir_name, "log_predict.csv", "w")
    logger.log_fp_.write(f"frame,fps\n")

    # 入力動画読み込み
    movie_loader = MovieLoader(movie_fpath, play_fps, num_batch_frame)
    num_frame    = movie_loader.getNumFrame()
    play_fps     = movie_loader.getPlayFps()

    # フレーム読み込み ＆ 検出実行
    with tqdm(movie_loader) as movie_iter:
        for batch_frame_nos, batch_imgs in movie_iter:

            if len(batch_imgs) > 0:

                # SSD物体検出（複数フレーム分をまとめて検出）
                time_s = time.perf_counter()
                det_results = ssd_model.predict(img_procs, batch_imgs, conf, overlap)

                # 試作: 車の枠からナンバープレートを再検出
                det_results = ssd_model.predictDetail(det_results, batch_imgs, (num_batch_frame * len(img_procs)), 100, "car", conf, overlap)
                # det_results = ssd_model.predict(img_procs, batch_imgs, conf, overlap)

                time_e = time.perf_counter()

                time_per_frame = (time_e - time_s) / len(batch_imgs)
                # print(f"In main_play_movie(): fps={1.0/time_per_frame}")

                # フレーム毎の処理
                for batch_frame_no, img_org, det_result in zip(batch_frame_nos, batch_imgs, det_results):
                    movie_iter.set_description(f"[{batch_frame_no}/{num_frame}]") # 進捗表示

                    # 検出範囲描画
                    for img_proc in img_procs:
                        img_org = img_proc.drawDetArea(img_org, DrawPen((0,128,0), 1, 0.4))

                    # 検出結果描画
                    img_org = ImageProc.drawResultDet(img_org, det_result, DrawPen((255,255,255), 1, 0.4))

                    # FPS等を描画
                    img_org = ImageProc.drawResultSummary(img_org, batch_frame_no, num_frame, 
                                                        ssd_model.device_.type, 
                                                        ssd_model.net_type_,
                                                        time_per_frame,
                                                        DrawPen((255,255,255), 2, 0.6))
                    
                    logger.log_fp_.write(f"F{batch_frame_no},{(1.0 / time_per_frame):.1f}\n")

                    # 保存
                    frame_img_fpath = f"{output_imgdir_path}/F{batch_frame_no:05}.jpg" 
                    cv2.imwrite(frame_img_fpath, img_org)
                    
                    # 表示
                    cv2.imshow(output_imgdir_name, img_org)

                    key = cv2.waitKey(int(100.0 / play_fps)) & 0xFF
                    if key == ord("q"):
                        break

    logger.closeLogFile()
    cv2.destroyAllWindows()
    print("output: ", output_imgdir_path)

    return

def main_det_img(img_procs:List[ImageProc], ssd_model:SSDModelDetector, img_fpath:str, conf:float, overlap:float):
    """ SSD検知実行（画像1枚）

    Args:
        img_procs (List[ImageProc]) : 検出領域（複数）[area_num, 検出領域]
        ssd_model (SSDModelDetector): SSDモデル
        img_fpath (str)             : 入力画像パス
        conf (float)                : 信頼度conf下限閾値
        overlap (float)             : 重複有無の判定閾値(IoU)
    """
    # 結果出力フォルダ作成
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, ssd_model.net_type_, "")

    # 入力画像読み込み
    img_org:np.ndarray = cv2.imread(img_fpath) 

    if img_org is not None:

        # SSD物体検出
        time_s = time.perf_counter()
        det_results = ssd_model.predict(img_procs, [img_org], conf, overlap)
        time_e = time.perf_counter()

        # 検出範囲描画
        for img_proc in img_procs:
            img_org = img_proc.drawDetArea(img_org, DrawPen((0,128,0), 1, 0.4))

        # 検出結果描画
        img_org = ImageProc.drawResultDet(img_org, det_results[0], DrawPen((255,255,255), 1, 0.4))


        # FPS等を描画
        img_org = ImageProc.drawResultSummary(img_org, 0, 0, 
                                             ssd_model.device_.type, 
                                             ssd_model.net_type_,
                                             (time_e - time_s),
                                             DrawPen((255,255,255), 2, 0.6))

        # 保存
        img_fname = os.path.splitext(os.path.basename(img_fpath))[0]
        img_out_fpath = f"{output_imgdir_path}/{img_fname}_result.jpg"
        cv2.imwrite(img_out_fpath, img_org)

        print("output: ", img_out_fpath)

    return

def main_play_imageset(ssd_model:SSDModelDetector, img_dir:str, conf:float, overlap:float, eval_iou_th:float):
    """ SSD検知実行（ディレクトリ内の複数画像）＆ 評価（mAP算出（アノテーションデータ付属時のみ））

    Args:
        ssd_model (SSDModelDetector): SSDモデル
        img_dir (str): 入力ディレクトリ
        conf (float): 信頼度conf下限閾値
        overlap (float): 重複有無の判定閾値(IoU)
        eval_iou_th (float): 評価（mAP算出）時のIoU閾値
    """
    # 検出領域は、画像全域のみサポート
    img_proc = ImageProc()

    # 画像出力用フォルダ作成
    img_dir = img_dir.replace("\\","/")
    if img_dir[-1] == "/":
        output_imgdir_name = os.path.basename(os.path.dirname(img_dir))
    else:
        output_imgdir_name = os.path.basename(img_dir)

    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, ssd_model.net_type_, output_imgdir_name)

    # 入力画像ファイルリスト読み込み
    parse_anno = Anno_xml2list(ssd_model.voc_classes_)
    val_file_all  = [os.path.split(f)[1].split(".")[0] for f in glob.glob(f"{img_dir}/*.xml")]
    val_file_list = [f for f in val_file_all if parse_anno.isExistObject(f"{img_dir}/{f}.xml") == True]

    calc_map:CalcMAP = None

    is_exist_anno = False
    val_img_list  = []
    val_anno_list = []
    
    if len(val_file_list) > 0:
        # [アノテーションデータがある場合] 
        is_exist_anno = True
        val_img_list  = [f"{img_dir}/{f}.jpg" for f in val_file_list]
        val_anno_list = [f"{img_dir}/{f}.xml" for f in val_file_list]

        log_eval_anno = LogEvalAnno(True)
        log_eval_anno.openLogFile(ssd_model.device_.type, ssd_model.net_type_, output_imgdir_name)

        calc_map = CalcMAP(val_img_list, ssd_model.voc_classes_, iou_thres=eval_iou_th)

    else:
        # [画像データのみの場合] 
        val_img_list  = [f"{img_dir}/{f}.jpg" for f in val_file_all]
        log_eval_anno = LogEvalAnno(False)

    num_img = len(val_img_list)

    for idx, img_fpath in enumerate(val_img_list):

        # 入力画像読み込み
        img_org = cv2.imread(img_fpath) 

        if img_org is not None:

            print(f"[{idx}/{num_img}] proc: {img_fpath}...", )

            # SSD物体検出
            det_results = ssd_model.predict([img_proc], [img_org], conf, overlap)

            # 検出結果描画
            img_org = ImageProc.drawResultDet(img_org, det_results[0], DrawPen((128,255,255), 1, 0.4))

            if is_exist_anno == True:
                # アノテーションデータ取得＆描画
                img_h, img_w, _ = img_org.shape

                anno_data = getAnnoData(val_anno_list[idx], parse_anno, ssd_model.voc_classes_, img_w, img_h)
                img_org   = ImageProc.drawAnnoData(img_org, anno_data, DrawPen((128,255,128), 1, 0.4))

                calc_map.add(img_fpath, det_results[0], anno_data)

                for anno_cur in anno_data:
                    # アノテーションデータの評価結果をログ出力
                    (pos_det, jaccard_val) = anno_cur.extractPositiveDetResult(det_results[0], eval_iou_th)
                    log_eval_anno.writeLog(img_fpath, anno_cur, pos_det, jaccard_val)

            # 保存
            img_fname = os.path.splitext(os.path.basename(img_fpath))[0]
            img_out_fpath = f"{output_imgdir_path}/{img_fname}_result.jpg"
            cv2.imwrite(img_out_fpath, img_org)


            print(f"[{idx}/{num_img}] output: {img_out_fpath}", )

        # 表示
        cv2.imshow(output_imgdir_name, img_org)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break        

    if is_exist_anno == True:
        # mAP計算
        mAP, per_class_ap = calc_map()

        log_eval_anno.log_fp_.write(f"\n\nMean Average Precision({eval_iou_th}) =,{mAP}\n")
        print(f"\nMean Average Precision({eval_iou_th}) = {mAP}\n")

        log_eval_anno.log_fp_.write(f"== Average Precision({eval_iou_th}) per class ==\n")
        print(f"== Average Precision({eval_iou_th}) per class ==")

        for cls_name, ap_val in per_class_ap.items():
            log_eval_anno.log_fp_.write(f"{cls_name},{ap_val}\n")
            print(f"{cls_name}: {ap_val}")

    log_eval_anno.closeLogFile()

    return


def main(media_fpath:str, cfg:Dict[str,Any]):
    """ メイン（SSDモデル作成、検知実行）

    Args:
        media_fpath (str)   : 入力データパス
        cfg (Dict[str,Any]) : config
    """

    play_fps:float            = cfg["play_fps"]
    img_procs:List[ImageProc] = cfg["img_procs"]
    net_type:str              = cfg["ssd_model_net_type"]
    weight_fpath:str          = cfg["ssd_model_weight_fpath"]
    num_batch:int             = cfg["ssd_model_num_batch"]
    conf:float                = cfg["ssd_model_conf_lower_th"]
    overlap:float             = cfg["ssd_model_iou_th"]
    eval_iou_th:float         = cfg["ssd_model_eval_iou_th"]

    if (os.path.isfile(media_fpath) == False) and (os.path.isdir(media_fpath) == False):
        print("Error: ", media_fpath, " is nothing.")

    else:
        # SSDモデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print("使用デバイス：", device)

        ssd_model = SSDModelDetector(device, net_type, weight_fpath)

        media_fname:str = os.path.basename(media_fpath)
        if ".mp4" in media_fname:
            # 動画再生
            main_play_movie(img_procs, num_batch, ssd_model, media_fpath, play_fps, conf, overlap)

        elif (".jpg" in media_fname) or (".png" in media_fname):
            # 画像
            main_det_img(img_procs, ssd_model, media_fpath, conf, overlap)
        
        elif os.path.isdir(media_fpath) == True:
            # ディレクトリ
            main_play_imageset(ssd_model, media_fpath, conf, overlap, eval_iou_th)

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

        # 入力画像全域を検出範囲にする場合は以下を有効化
        # "img_procs"    : [ImageProc()],

        # (SSDモデル:VGGベース)ネットワーク種別/パラメータ/バッチ処理数(※)
        #   (※) バッチ処理数 ＝検出範囲数 x フレーム数
        # "ssd_model_net_type"     : "vgg16-ssd",
        # "ssd_model_weight_fpath" : "./weights/vgg16-ssd_best_od_cars.pth", 
        # "ssd_model_num_batch" : 32,

        # (SSDモデル:mobilenetベース)ネットワーク種別/パラメータ/バッチ処理数
        "ssd_model_net_type"     : "mb2-ssd",
        "ssd_model_weight_fpath" : "./weights/mb2-ssd_best_od_cars.pth", 
        "ssd_model_num_batch" : 64,
        # "ssd_model_num_batch" : 32,

        # (SSDモデル) 信頼度confの足切り閾値
        "ssd_model_conf_lower_th" : 0.5,

        # (SSDモデル) 重複枠削除する重なり率(iou)閾値
        "ssd_model_iou_th" : 0.4,
        # (SSDモデル(評価用)) 正解枠との重なり率(iou)閾値
        "ssd_model_eval_iou_th" : 0.4,

    }

    if len(args) < 2:
        print("Usage: ", args[0], " [movie/img file path] ([play fps])")
    else:
        if len(args) >= 3:
            cfg["play_fps"] = float(args[2])

        main(args[1], cfg)
