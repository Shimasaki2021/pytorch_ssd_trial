#!/usr/bin/env python3

import sys
import cv2
import os
import time
import glob
import copy
from typing import List,Tuple,Dict,Any

from utils.ssd_model import SSD, DataTransform, VOCDataset, Anno_xml2list, nm_suppression
import numpy as np
from tqdm import tqdm
import torch
import torchvision

from common_ssd import ImageProc, MovieLoader, SSDModel, DetResult, AnnoData, DrawPen, Logger, makeVocClassesTxtFpath

# SSDモデル作成＆推論
class SSDModelDetector(SSDModel):

    def __init__(self, device:torch.device, weight_fpath:str):
        # 学習済み重みをロード
        (net_weights, voc_classes) = self.loadWeight(weight_fpath, device)

        super().__init__(device, voc_classes)

        # SSDネットワークモデル
        self.net_ = SSD(phase="inference", cfg=self.ssd_cfg_)
        self.net_.load_state_dict(net_weights)
        self.net_.eval()

        # ネットワークをDeviceへ
        self.net_.to(self.device_)

        # 前処理を行うクラス(DataTransform)のインスタンス作成
        self.transform_ = DataTransform(self.input_size_, self.color_mean_)

        # (torch.compile) Windows+anaconda環境では使用不可。WSL2+Ubuntu環境では使用可。
        #   GPU=GTX1660SUPERでは、以下Warningが出るが一応実行は可能。効果はわずか（1分の動画の処理時間が、12分7秒 → 11分57秒）
        #     W0316 07:20:49.853000 15169 torch/_inductor/utils.py:1137] [0/0_1] Not enough SMs to use max_autotune_gemm mode
        # self.net_ = torch.compile(self.net_, mode="default")
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

    def transImage(self, img:np.ndarray) -> np.ndarray:
        (img_trans, _ ,_) = self.transform_(img, "val", "", "")
        return img_trans
    
    def predict(self, img_procs:List[ImageProc], imgs:List[np.ndarray], data_confidence_level:float=0.5, overlap:float=0.45) -> List[List[DetResult]]:
        
        num_img  = len(imgs)

        det_results:List[List[DetResult]] = []
        for img_no in range(num_img):
            det_results.append(list())

        if num_img > 0:
            num_area = len(img_procs)


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
            find_index = np.where(outputs[:, :, :, 0] >= data_confidence_level) # (batch_num, label, top200)
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

    def nmSuppression(self, det_results:List[DetResult], iou:float=0.45) -> List[DetResult]:
        # 重複枠の削除
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

def main_play_movie(img_procs:List[ImageProc], num_batch:int, ssd_model:SSDModelDetector, movie_fpath:str, play_fps:float, conf:float, overlap:float):

    num_batch_frame = int(num_batch / len(img_procs))
    if num_batch_frame < 1:
        num_batch_frame = 1

    # 画像出力用フォルダ作成
    output_imgdir_name = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path = Logger.createOutputDir(ssd_model.device_.type, output_imgdir_name)

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
                time_e = time.perf_counter()

                time_per_batch = (time_e - time_s) / len(batch_imgs)

                # フレーム毎の処理
                for batch_frame_no, img_org, det_result in zip(batch_frame_nos, batch_imgs, det_results):
                    movie_iter.set_description(f"[{batch_frame_no}/{num_frame}]") # 進捗表示

                    # 検出範囲描画
                    for img_proc in img_procs:
                        img_org = img_proc.drawDetArea(img_org, DrawPen((255,255,255), 1, 0.4))

                    # 検出結果描画
                    img_org = ImageProc.drawResultDet(img_org, det_result, DrawPen((255,255,255), 1, 0.4))

                    # FPS等を描画
                    img_org = ImageProc.drawResultSummary(img_org, batch_frame_no, num_frame, 
                                                        ssd_model.device_.type, 
                                                        time_per_batch,
                                                        DrawPen((255,255,255), 2, 0.6))

                    # 保存
                    frame_img_fpath = f"{output_imgdir_path}/F{batch_frame_no:05}.jpg" 
                    cv2.imwrite(frame_img_fpath, img_org)
                    
                    # 表示
                    cv2.imshow(output_imgdir_name, img_org)

                    key = cv2.waitKey(int(1000.0 / play_fps)) & 0xFF
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

    if img_org is not None:
        
        time_s = time.perf_counter()

        # SSD物体検出
        det_results = ssd_model.predict(img_procs, [img_org], conf, overlap)

        # 検出範囲描画
        for img_proc in img_procs:
            img_org = img_proc.drawDetArea(img_org, DrawPen((255,255,255), 1, 0.4))

        # 検出結果描画
        img_org = ImageProc.drawResultDet(img_org, det_results[0], DrawPen((255,255,255), 1, 0.4))

        time_e = time.perf_counter()

        # FPS等を描画
        img_org = ImageProc.drawResultSummary(img_org, 0, 0, 
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
            det_results = ssd_model.predict([img_proc], [img_org], conf)

            # 検出結果描画
            img_org = ImageProc.drawResultDet(img_org, det_results[0], DrawPen((128,255,255), 1, 0.4))

            if is_exist_anno == True:
                # アノテーションデータ取得＆描画
                img_h, img_w, _ = img_org.shape

                anno_data = ImageProc.getAnnoData(val_anno_list[idx], parse_anno, ssd_model.voc_classes_, img_w, img_h)
                img_org = ImageProc.drawAnnoData(img_org, anno_data, DrawPen((128,255,128), 1, 0.4))

                for anno_cur in anno_data:
                    # アノテーションデータの評価結果をログ出力
                    (pos_det, jaccard_val) = anno_cur.extractPositiveDetResult(det_results[0])
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


def main(media_fpath:str, cfg:Dict[str,Any]):

    play_fps:float            = cfg["play_fps"]
    weight_fpath:str          = cfg["ssd_model_weight_fpath"]
    img_procs:List[ImageProc] = cfg["img_procs"]
    num_batch:int             = cfg["ssd_model_num_batch"]
    conf:float                = cfg["ssd_model_conf_lower_th"]
    overlap:float             = cfg["ssd_model_iou_th"]

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
            main_play_movie(img_procs, num_batch, ssd_model, media_fpath, play_fps, conf, overlap)

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

    cfg = {
        # 動画再生fps (負値＝入力動画のfpsそのまま)
        "play_fps"     : -1.0,

        # 検出範囲(1280x720、真ん中or右車線走行シーン、駐車場シーン用)
        # "img_procs"    : [ImageProc(0, 250, 350, 600), 
        #                   ImageProc(250, 200, 550, 500), 
        #                   ImageProc(480, 200, 780, 500), 
        #                   ImageProc(730, 200, 1030, 500), 
        #                   ImageProc(930, 250, 1280, 600)],

        # 検出範囲(1280x720、左車線走行シーン用)
        "img_procs"    : [ImageProc(480, 200, 780, 500), 
                          ImageProc(730, 200, 1030, 500), 
                          ImageProc(930, 250, 1280, 600)], 

        # 入力画像全域を検出範囲にする場合は以下を有効化
        # "img_procs"    : [ImageProc()],

        # (SSDモデル)パラメータ
        "ssd_model_weight_fpath" : "./weights/ssd_best_od_cars.pth", 

        # (SSDモデル) 信頼度confの足切り閾値
        "ssd_model_conf_lower_th" : 0.5,

        # (SSDモデル) 重複枠削除する重なり率(iou)閾値
        "ssd_model_iou_th" : 0.5,

        # (SSDモデル) バッチ処理数（＝検出範囲数 x フレーム数）
        "ssd_model_num_batch" : 32,
    }

    if len(args) < 2:
        print("Usage: ", args[0], " [movie/img file path] ([play fps])")
    else:
        if len(args) >= 3:
            cfg["play_fps"] = float(args[2])

        main(args[1], cfg)
