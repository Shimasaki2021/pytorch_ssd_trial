#!/usr/bin/env python3

import sys
import cv2
import os
import time
import glob
from typing import List,Tuple,Any

from utils.ssd_model import SSD, DataTransform, VOCDataset, Anno_xml2list
import numpy as np
import torch

from common_ssd import ImageProc, SSDModel, DetResult, DrawPen, makeVocClassesTxtFpath

# SSDモデル作成＆推論
class SSDModelDetector(SSDModel):

    def __init__(self, device:torch.device, weight_fpath:str):
        # 学習済み重みをロード
        (net_weights, voc_classes) = self.loadWeight(weight_fpath)

        super().__init__(device, voc_classes)

        # SSDネットワークモデル
        self.net_ = SSD(phase="inference", cfg=self.ssd_cfg_)
        self.net_.load_state_dict(net_weights)

        # ネットワークをDeviceへ
        self.net_.to(self.device_)
        return
    
    def loadWeight(self, weight_fpath:str) -> Tuple[Any, List[str]]:
        # ネットワーク重みをロード
        net_weights = torch.load(weight_fpath, weights_only=True) # FutureWarning: You are using torch.load..対策

        # クラス名リスト（voc_classes）をロード
        voc_classes:List[str] = []

        voc_classes_fpath = makeVocClassesTxtFpath(weight_fpath)
        voc_classes_fp = open(voc_classes_fpath,"r")
        if voc_classes_fp is not None:
            voc_classes = [name.strip() for name in voc_classes_fp.readlines()]
            voc_classes_fp.close()

        return (net_weights, voc_classes)
    
    def predict(self, img:np.ndarray, data_confidence_level:float=0.5) -> Tuple[List[np.ndarray], List[int], List[float]]:
        # 画像を前処理
        img_height, img_width, _ = img.shape

        transform = DataTransform(self.input_size_, self.color_mean_)
        (img_trans, _ ,_) = transform(img, "val", "", "")
        img_torch = torch.from_numpy(img_trans[:, :, (2, 1, 0)]).permute(2, 0, 1) # BGR→RGB, (H,W,C)→(C,H,W)

        img_batch = img_torch.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])            
        img_batch = img_batch.to(self.device_)

        # 推論実行
        torch.backends.cudnn.benchmark = True
        self.net_.eval()
        outputs = self.net_(img_batch)

        # 結果取得
        predict_bbox:List[np.ndarray]  = []
        pre_dict_label_index:List[int] = []
        scores:List[float]             = []    
        outputs = outputs.cpu().detach().numpy()

        find_index = np.where(outputs[:, 0:, :, 0] >= data_confidence_level)
        outputs    = outputs[find_index]

        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す

            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc   = outputs[i][0]  # 確信度
                bbox = outputs[i][1:] * [img_width, img_height, img_width, img_height]

                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)    

        return (predict_bbox, pre_dict_label_index, scores)


def main_play_movie(img_procs:List[ImageProc], ssd_model:SSDModelDetector, movie_fpath:str, play_fps:float, conf:float):

    # 画像出力用フォルダ作成
    output_imgdir_name:str = os.path.splitext(os.path.basename(movie_fpath))[0]
    output_imgdir_path:str = f"./output.{ssd_model.device_.type}/" + output_imgdir_name

    if os.path.isdir(output_imgdir_path) == False:
        os.makedirs(output_imgdir_path)

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

                for img_proc in img_procs:
                    # 検出範囲切り出し
                    img_det = img_proc.clip(img_org)

                    # SSD物体検出
                    (predict_bbox, pre_dict_label_index, scores) = ssd_model.predict(img_det, conf)
                    det_results = img_proc.convDetResult(ssd_model.voc_classes_, predict_bbox, pre_dict_label_index, scores)

                    # 検出結果描画
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

def main_det_img(img_procs:List[ImageProc], ssd_model:SSDModelDetector, img_fpath:str, conf:float):

    # 結果出力フォルダ作成
    output_imgdir_path:str = f"./output.{ssd_model.device_.type}/"
    if os.path.isdir(output_imgdir_path) == False:
        os.makedirs(output_imgdir_path)

    # 入力画像読み込み
    img_org:np.ndarray = cv2.imread(img_fpath) 

    if not (img_org is None):
        
        time_s = time.perf_counter()

        for img_proc in img_procs:
            # 検出範囲切り出し
            img_det = img_proc.clip(img_org)

            # SSD物体検出
            (predict_bbox, pre_dict_label_index, scores) = ssd_model.predict(img_det, conf)
            det_results = img_proc.convDetResult(ssd_model.voc_classes_, predict_bbox, pre_dict_label_index, scores)

            # 検出結果描画
            img_org = img_proc.drawResultDet(img_org, det_results, DrawPen((255,255,255), 1, 0.4))

        time_e = time.perf_counter()

        # FPS等を描画
        img_org = img_proc.drawResultSummary(img_org, 0, 0, 
                                             ssd_model.device_.type, 
                                             (time_e - time_s),
                                             DrawPen((255,255,255), 2, 0.6))

        # 保存
        img_fname = os.path.splitext(os.path.basename(img_fpath))[0]
        img_out_fpath = f"{output_imgdir_path}{img_fname}_result.jpg"
        cv2.imwrite(img_out_fpath, img_org)

        print("output: ", img_out_fpath)

    return

def main_play_imageset(ssd_model:SSDModelDetector, img_dir:str, conf:float):
    # 検出領域は、画像全域のみサポート
    img_proc = ImageProc()

    # 画像出力用フォルダ作成
    if img_dir[-1] == "/":
        output_imgdir_name = os.path.basename(os.path.dirname(img_dir))
    else:
        output_imgdir_name = os.path.basename(img_dir)

    output_imgdir_path = f"./output.{ssd_model.device_.type}/{output_imgdir_name}"

    if os.path.isdir(output_imgdir_path) == False:
        os.makedirs(output_imgdir_path)

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

    else:
        # [画像データのみの場合] 
        val_img_list    = [f"{img_dir}/{f}.jpg" for f in val_file_all]

    num_img = len(val_img_list)

    for idx, img_fpath in enumerate(val_img_list):

        # 入力画像読み込み
        img_org = cv2.imread(img_fpath) 

        if img_org is not None:
            
            time_s = time.perf_counter()

            # SSD物体検出
            (predict_bbox, pre_dict_label_index, scores) = ssd_model.predict(img_org, conf)
            det_results = img_proc.convDetResult(ssd_model.voc_classes_, predict_bbox, pre_dict_label_index, scores)

            # 検出結果描画
            img_org = img_proc.drawResultDet(img_org, det_results, DrawPen((128,255,255), 1, 0.4))

            time_e = time.perf_counter()

            # アノテーションデータ取得＆描画
            if is_exist_anno == True:
                img_h, img_w, _ = img_org.shape
                # anno_list = parse_anno(val_anno_list[idx], img_w, img_h)
                anno_data = img_proc.getAnnoData(val_anno_list[idx], parse_anno, ssd_model.voc_classes_, img_w, img_h)
                img_org = img_proc.drawAnnoData(img_org, anno_data, DrawPen((128,255,128), 1, 0.4))


            # FPS等を描画
            # img_org = img_proc.drawResultSummary(img_org, 0, 0, 
            #                                      ssd_model.device_.type, 
            #                                      (time_e - time_s),
            #                                      DrawPen((255,255,255), 1, 0.4))

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

    return


def main(media_fpath:str, play_fps:float, weight_fpath:str, img_procs:List[ImageProc]):

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
            main_play_movie(img_procs,ssd_model, media_fpath, play_fps, 0.5)

        elif (".jpg" in media_fname) or (".png" in media_fname):
            # 画像
            main_det_img(img_procs, ssd_model, media_fpath, 0.5)
        
        elif os.path.isdir(media_fpath) == True:
            # ディレクトリ
            main_play_imageset(ssd_model, media_fpath, 0.5)

        else:
            print("No support ext [", media_fname, "]")
    
    return

if __name__ == "__main__":
    args = sys.argv

    weight_fpath = "./weights/ssd_best_od_cars.pth"

    # 検出範囲
    img_procs = [ImageProc(180, 250, 780, 550), ImageProc(680, 250, 1280, 550)] # (1280x720を)600x300に切り出し(左右2領域)
    # img_procs = [ImageProc()] # 全域

    play_fps:float = -1.0

    if len(args) < 2:
        print("Usage: ", args[0], " [movie/img file path] ([play fps])")
    else:
        if len(args) >= 3:
            play_fps = float(args[2])

        main(args[1], play_fps, weight_fpath, img_procs)
