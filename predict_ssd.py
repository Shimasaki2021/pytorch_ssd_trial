#!/usr/bin/env python3

import sys
import cv2
import os
import time
import glob
from typing import List,Tuple,Any

from utils.ssd_model import SSD, DataTransform, VOCDataset, Anno_xml2list, nm_suppression
import numpy as np
import torch

from common_ssd import ImageProc, SSDModel, DetResult, AnnoData, DrawPen, Logger, makeVocClassesTxtFpath

# SSDモデル作成＆推論
class SSDModelDetector(SSDModel):

    def __init__(self, device:torch.device, weight_fpath:str):
        # 学習済み重みをロード
        (net_weights, voc_classes) = self.loadWeight(weight_fpath, device)

        super().__init__(device, voc_classes)

        # SSDネットワークモデル
        self.net_ = SSD(phase="inference", cfg=self.ssd_cfg_)
        self.net_.load_state_dict(net_weights)

        # ネットワークをDeviceへ
        self.net_.to(self.device_)
        return
    
    def loadWeight(self, weight_fpath:str, device:torch.device) -> Tuple[Any, List[str]]:
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

    def predict(self, img_procs:List[ImageProc], img_org:np.ndarray, data_confidence_level:float=0.5, overlap:float=0.45) -> List[DetResult]:
        
        # 複数範囲の画像バッチ化
        transform = DataTransform(self.input_size_, self.color_mean_)

        imgs_trans:List[np.ndarray] = []
        for img_proc in img_procs:
            # 検出範囲切り出し
            img_det = img_proc.clip(img_org)
            # 画像を前処理
            (img_trans, _ ,_) = transform(img_det, "val", "", "")
            imgs_trans.append(img_trans[:, :, (2, 1, 0)]) # [h,w,ch(BGR→RGB)]
        
        imgs_trans_np = np.array(imgs_trans)                         # [batch_num, h, w, ch(RGB)]
        img_batch = torch.from_numpy(imgs_trans_np).permute(0,3,1,2) # [batch_num, ch(RGB), h, w]
        img_batch = img_batch.to(self.device_)
        # print(f"img_batch = {img_batch.shape}")

        # 推論実行
        torch.backends.cudnn.benchmark = True
        self.net_.eval()
        outputs = self.net_(img_batch)

        # 結果取得
        outputs    = outputs.cpu().detach().numpy()
        find_index = np.where(outputs[:, :, :, 0] >= data_confidence_level) # (batch_num, label, top)
        outputs    = outputs[find_index]

        det_results:List[DetResult] = []

        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す

            area_no  = find_index[0][i] # 検出領域index (batch index)
            label_no = find_index[1][i] # ラベル番号

            if label_no > 0:  
                # [背景クラスでない場合] 結果を取得

                # 確信度conf
                sc = outputs[i][0]
                # クラス名
                cls_name = self.voc_classes_[label_no-1]
                # Bounding Box: 切り出し前の画像上での座標値に変換
                bb_i = img_procs[area_no].convBBox( outputs[i][1:] )

                det_results.append(DetResult(cls_name, bb_i, sc))

        if len(img_procs) > 1:
            # [検出領域が複数の場合] 重複領域での重複枠を取り除く
            det_results = self.nmSuppression(det_results, overlap)

        return det_results

    def nmSuppression(self, det_results:List[DetResult], iou:float=0.45) -> List[DetResult]:
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
    def __init__(self, is_out:bool):
        super().__init__(is_out)
        return
    
    def openLogFile(self, dev_name:str, output_imgdir_name:str):
        if self.is_out_ == True:
            super().openLogFile(dev_name, output_imgdir_name, "eval_anno.csv", "w")

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

def main_play_movie(img_procs:List[ImageProc], ssd_model:SSDModelDetector, movie_fpath:str, play_fps:float, conf:float, overlap:float):

    # 画像出力用フォルダ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, output_imgdir_name)

    # 入力動画読み込み
    cap       = cv2.VideoCapture(movie_fpath)  
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fps   = cap.get(cv2.CAP_PROP_FPS)

    frame_play_step = int((cap_fps + 0.1) / play_fps)
    if frame_play_step < 1:
        frame_play_step = 1

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

                # 検出結果描画
                for img_proc in img_procs:
                    img_org = img_proc.drawResultDet(img_org, det_results, DrawPen((255,255,255), 1, 0.4))

                time_e = time.perf_counter()

                # FPS等を描画
                img_org = img_proc.drawResultSummary(img_org, frame_no, num_frame, 
                                                     ssd_model.device_.type, 
                                                     (time_e - time_s),
                                                     DrawPen((255,255,255), 2, 0.6))

                # 保存
                frame_img_fpath = output_imgdir_path + "/F{:05}".format(frame_no) + ".jpg" 
                cv2.imwrite(frame_img_fpath, img_org)
                
                # 表示
                cv2.imshow(output_imgdir_name, img_org)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("output: ", output_imgdir_path)

    return

def main_det_img(img_procs:List[ImageProc], ssd_model:SSDModelDetector, img_fpath:str, conf:float, overlap:float):

    # 結果出力フォルダ作成
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, "")

    # 入力画像読み込み
    img_org:np.ndarray = cv2.imread(img_fpath) 

    if not (img_org is None):
        
        time_s = time.perf_counter()

        # SSD物体検出
        det_results = ssd_model.predict(img_procs, img_org, conf, overlap)

        # 検出結果描画
        for img_proc in img_procs:
            img_org = img_proc.drawResultDet(img_org, det_results, DrawPen((255,255,255), 1, 0.4))

        time_e = time.perf_counter()

        # FPS等を描画
        img_org = img_proc.drawResultSummary(img_org, 0, 0, 
                                             ssd_model.device_.type, 
                                             (time_e - time_s),
                                             DrawPen((255,255,255), 2, 0.6))

        # 保存
        img_fname = os.path.splitext(os.path.basename(img_fpath))[0]
        img_out_fpath = f"{output_imgdir_path}/{img_fname}_result.jpg"
        cv2.imwrite(img_out_fpath, img_org)

        print("output: ", img_out_fpath)

    return

def main_play_imageset(ssd_model:SSDModelDetector, img_dir:str, conf:float):
    # 検出領域は、画像全域のみサポート
    img_proc = ImageProc()

    # 画像出力用フォルダ作成
    img_dir = img_dir.replace("\\","/")
    if img_dir[-1] == "/":
        output_imgdir_name = os.path.basename(os.path.dirname(img_dir))
    else:
        output_imgdir_name = os.path.basename(img_dir)

    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, output_imgdir_name)

    # 入力画像ファイルリスト読み込み
    parse_anno = Anno_xml2list(ssd_model.voc_classes_)
    val_file_all  = [os.path.split(f)[1].split(".")[0] for f in glob.glob(f"{img_dir}/*.xml")]
    val_file_list = [f for f in val_file_all if parse_anno.isExistObject(f"{img_dir}/{f}.xml") == True]

    is_exist_anno = False
    val_img_list  = []
    val_anno_list = []
    
    if len(val_file_list) > 0:
        # [アノテーションデータがある場合] 
        is_exist_anno = True
        val_img_list  = [f"{img_dir}/{f}.jpg" for f in val_file_list]
        val_anno_list = [f"{img_dir}/{f}.xml" for f in val_file_list]

        log_eval_anno = LogEvalAnno(True)
        log_eval_anno.openLogFile(ssd_model.device_.type, output_imgdir_name)

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
            det_results = ssd_model.predict([img_proc], img_org, conf)

            # 検出結果描画
            img_org = img_proc.drawResultDet(img_org, det_results, DrawPen((128,255,255), 1, 0.4))

            if is_exist_anno == True:
                # アノテーションデータ取得＆描画
                img_h, img_w, _ = img_org.shape

                anno_data = img_proc.getAnnoData(val_anno_list[idx], parse_anno, ssd_model.voc_classes_, img_w, img_h)
                img_org = img_proc.drawAnnoData(img_org, anno_data, DrawPen((128,255,128), 1, 0.4))

                for anno_cur in anno_data:
                    # アノテーションデータの評価結果をログ出力
                    (pos_det, jaccard_val) = anno_cur.extractPositiveDetResult(det_results)
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

    log_eval_anno.closeLogFile()

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
            # 動画再生
            main_play_movie(img_procs,ssd_model, media_fpath, play_fps, conf, overlap)

        elif (".jpg" in media_fname) or (".png" in media_fname):
            # 画像
            main_det_img(img_procs, ssd_model, media_fpath, conf, overlap)
        
        elif os.path.isdir(media_fpath) == True:
            # ディレクトリ
            main_play_imageset(ssd_model, media_fpath, conf)

        else:
            print("No support ext [", media_fname, "]")
    
    return

if __name__ == "__main__":
    args = sys.argv

    weight_fpath = "./weights/ssd_best_od_cars.pth"

    # 検出範囲
    #   (1280x720を)300x300/350x350に切り出し
    img_procs = [ImageProc(110, 250, 530, 600), 
                 ImageProc(480, 200, 780, 500), 
                 ImageProc(730, 200, 1030, 500), 
                 ImageProc(930, 250, 1280, 600)] 

    # 入力画像全域を検出範囲にする場合は以下を有効化
    # img_procs = [ImageProc()] 

    conf     = 0.5
    overlap  = 0.2
    play_fps = -1.0

    if len(args) < 2:
        print("Usage: ", args[0], " [movie/img file path] ([play fps])")
    else:
        if len(args) >= 3:
            play_fps = float(args[2])

        main(args[1], play_fps, weight_fpath, img_procs, conf, overlap)
