#!/usr/bin/env python3

# パッケージのimport
import os
import sys
import time
from typing import List
from io import TextIOWrapper

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision

from utils.ssd_model import SSD, MultiBoxLoss

from common_ssd import SSDModel, VocDataSetMng, makeVocClassesTxtFpath


class SSDModelTrainerDebug:

    def __init__(self, is_debug_out:bool, max_epoch:int):
        self.is_debug_output_      = is_debug_out
        self.debug_out_max_epoch_  = max_epoch
        self.debug_outdir_:str     = ""
        self.res_weight_fpath_:str = ""
        self.debug_log_fp_:TextIOWrapper = None
        return

    def openLogFile(self, dev_name:str, res_weight_fpath:str, fname:str):
        if self.is_debug_output_ == True:
            self.res_weight_fpath_ = res_weight_fpath

            res_weight_fname   = os.path.splitext(os.path.basename(res_weight_fpath))[0]
            self.debug_outdir_ = f"./output.{dev_name}/debug/train_{res_weight_fname}"

            if os.path.isdir(self.debug_outdir_) == False:
                os.makedirs(self.debug_outdir_)

            self.debug_log_fp_ = open(self.debug_outdir_ + "/" + fname, "w")
        return
    
    def isOutputLog(self) -> bool:
        ret = False
        if (self.is_debug_output_ == True) and (not (self.debug_log_fp_ is None)):
            ret = True
        return ret
    
    def closeLogFile(self):
        if self.isOutputLog() == True:
            self.debug_log_fp_.close()
            self.debug_log_fp_ = None
        return

    def outputLogSummary(self,dev_name:str, learn_data_path:str, epoch:int, val_loss:float):
        if self.isOutputLog() == True:
            self.debug_log_fp_.write("\n == summary ==\n")
            self.debug_log_fp_.write(f",device:,,{dev_name}\n")
            self.debug_log_fp_.write(f",learn data:,,{learn_data_path}\n")
            self.debug_log_fp_.write(f",out weight file:,,{self.res_weight_fpath_}\n")
            self.debug_log_fp_.write(f",epoch at out weight:,,{epoch}\n")
            self.debug_log_fp_.write(f",val loss at out weight:,,{val_loss}\n")
        return
    
    def outputLogEpochSummary(self, epoch:int, time_sec:float, train_loss:float, val_loss:float, train_iter:int, val_iter:int):
        if self.isOutputLog() == True:
            self.debug_log_fp_.write(f"epoch,{epoch},{time_sec:.4f},sec,train_Loss,{train_loss:.4f},(it:,{train_iter},),val_Loss,{val_loss:.4f},(it:,{val_iter},)\n")
        return
    
    def outputLogDataSetSummary(self, voc_dataset:VocDataSetMng):
        if self.isOutputLog() == True:
            self.debug_log_fp_.write("\n == dataset info ==\n")

            for phase in ["train", "val"]:
                image_num   = voc_dataset.getImageNum(phase)

                self.debug_log_fp_.write(f"{phase}\n")
                self.debug_log_fp_.write(f",image_num=,,{image_num}\n")

                for class_name in voc_dataset.voc_classes_:
                    (obj_num, obj_w_ave, obj_h_ave) = voc_dataset.getAnnoInfo(phase, class_name)
                    self.debug_log_fp_.write(f",class:,{class_name}\n")
                    self.debug_log_fp_.write(f",,obj_num=,,,{obj_num}\n")
                    self.debug_log_fp_.write(f",,obj_size(ave)(w x h)=,,,{int(obj_w_ave)},{int(obj_h_ave)}\n")

        return

    def dumpInputImage(self, epoch:int, images:List[torch.Tensor], phase:str, batch_idx:int):
        if (self.is_debug_output_ == True) and (epoch <= self.debug_out_max_epoch_):
            for image_no in range(images.shape[0]):
                debug_out_fpath = self.debug_outdir_ + "/epoch" + str(epoch) + "_" + phase + "_b" + str(batch_idx) + "_" + str(image_no) + ".jpg"
                torchvision.utils.save_image(images[image_no], debug_out_fpath)
        return

# SSDモデル作成＆学習
class SSDModelTrainer(SSDModel):

    def __init__(self, device:torch.device, voc_classes:List[str]):
        super().__init__(device, voc_classes)

        # SSDネットワークモデル
        self.net_ = SSD(phase="train", cfg=self.ssd_cfg_)

        # SSDのvggに、初期の重みを設定
        vgg_weights = torch.load("./weights/vgg16_reducedfc.pth", weights_only=True) # FutureWarning: You are using torch.load..対策
        self.net_.vgg.load_state_dict(vgg_weights)

        # SSDのextras, loc, confには、Heの初期値を適用
        self.net_.extras.apply(SSDModelTrainer.initWeight)
        self.net_.loc.apply(SSDModelTrainer.initWeight)
        self.net_.conf.apply(SSDModelTrainer.initWeight)

        # ネットワークをDeviceへ
        self.net_.to(device)

        # 損失関数の設定
        self.criterion_ = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

        # 最適化手法の設定
        #self.optimizer_ = optim.SGD(self.net_.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        #self.optimizer_ = optim.Adam(self.net_.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_ = optim.AdamW(self.net_.parameters(), lr=0.001)
        return
    
    @staticmethod
    def initWeight(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:  # バイアス項がある場合
                nn.init.constant_(m.bias, 0.0)
        return

    def train(self, voc_dataset:VocDataSetMng, num_epochs:int, res_weight_fpath:str):
        
        dataloaders_dict = voc_dataset.dataloaders_dict_

        # debug
        debug_train = SSDModelTrainerDebug(True, 1)
        debug_train.openLogFile(self.device_.type, res_weight_fpath, "train_log.csv")

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        # イテレーションカウンタをセット
        iteration        = 1
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss   = 0.0  # epochの損失和
        epoch_at_min_loss = 0   # val_loss最小時のepoch
        min_loss         = 9999.0
        #logs             = []

        # epochのループ
        for epoch in range(num_epochs+1):

            # 開始時刻を保存
            t_epoch_start = time.time()
            #t_iter_start  = time.time()

            iter_per_epoch_train = 0
            iter_per_epoch_val   = 0

            # epochごとの訓練と検証のループ
            for phase in ["train", "val"]:
                if phase == "train":
                    self.net_.train()
                else:
                    self.net_.eval()

                # データローダーからminibatchずつ取り出すループ
                with tqdm(dataloaders_dict[phase], desc=phase, file=sys.stdout) as iterator:
                    for batch_idx, (images, targets) in enumerate(iterator):
                        
                        # GPUが使えるならGPUにデータを送る
                        images  = images.to(self.device_)
                        targets = [ann.to(self.device_)
                                for ann in targets]  # リストの各要素のテンソルをGPUへ

                        # debug:入力画像ダンプ
                        debug_train.dumpInputImage(epoch, images, phase, batch_idx)

                        # optimizerを初期化
                        self.optimizer_.zero_grad()

                        # 順伝搬（forward）計算
                        with torch.set_grad_enabled(phase == "train"):
                            # 順伝搬（forward）計算
                            outputs = self.net_(images)

                            # 損失の計算
                            loss_l, loss_c = self.criterion_(outputs, targets)
                            loss = loss_l + loss_c

                            # 訓練時はバックプロパゲーション
                            if phase == "train":
                                loss.backward()
                                nn.utils.clip_grad_value_(self.net_.parameters(), clip_value=2.0)
                                self.optimizer_.step()
                                epoch_train_loss += loss.item()
                                iteration += 1
                                iter_per_epoch_train += 1
                            # 検証時
                            else:
                                epoch_val_loss += loss.item()
                                iter_per_epoch_val += 1

            # epochのphaseごとのlossと正解率(lossは、iteration数で正規化)
            t_epoch_finish = time.time()

            epoch_train_loss /= float(iter_per_epoch_train)
            epoch_val_loss   /= float(iter_per_epoch_val)

            print(f"epoch {epoch}/{num_epochs} {(t_epoch_finish - t_epoch_start):.4f}sec || train_Loss:{epoch_train_loss:.4f}(it:{iter_per_epoch_train}) val_Loss:{epoch_val_loss:.4f}(it:{iter_per_epoch_val})")

            # debug:ログ出力
            debug_train.outputLogEpochSummary(epoch, (t_epoch_finish - t_epoch_start), epoch_train_loss, epoch_val_loss, iter_per_epoch_train, iter_per_epoch_val)

            t_epoch_start  = time.time()

            # vallossが小さい、ネットワークを保存する
            if min_loss > epoch_val_loss:
                min_loss          = epoch_val_loss
                epoch_at_min_loss = epoch
                self.saveWeight(res_weight_fpath)

            epoch_train_loss = 0.0  # epochの損失和
            epoch_val_loss   = 0.0  # epochの損失和

        # debug
        debug_train.outputLogSummary(self.device_.type, voc_dataset.data_path_, epoch_at_min_loss, min_loss)
        debug_train.outputLogDataSetSummary(voc_dataset)
        debug_train.closeLogFile()

        return
    
    def saveWeight(self, weight_fpath:str):
        # ネットワーク重みをセーブ
        torch.save(self.net_.state_dict(), weight_fpath) 

        # クラス名リスト（voc_classes）をセーブ
        voc_classes_fpath = makeVocClassesTxtFpath(weight_fpath)

        voc_classes_fp = open(voc_classes_fpath,"w")
        if voc_classes_fp is not None:
            for cls_name in self.voc_classes_:
                voc_classes_fp.write(f"{cls_name}\n")
            voc_classes_fp.close()

        return


def main(num_epochs:int, data_path:str, voc_classes:List[str]):

    if os.path.isdir(data_path) == False:
        print("Error: ", data_path, " is nothing.")
    
    else:
        # SSDモデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)

        ssd_model = SSDModelTrainer(device, voc_classes)
    
        # データセット作成
        voc_dataset = VocDataSetMng(data_path, voc_classes, ssd_model.color_mean_, ssd_model.input_size_)
        for phase in ["train", "val"]:
            for class_name in voc_dataset.voc_classes_:
                (obj_num, _, _) = voc_dataset.getAnnoInfo(phase, class_name)
                print(f"DataSet({phase}): num={obj_num} (class:{class_name})")

        # SSDモデルの学習
        res_weight_fpath = "./weights/ssd_best_" + os.path.basename(data_path) + ".pth"
        ssd_model.train(voc_dataset, num_epochs, res_weight_fpath)

    return

if __name__ == "__main__":
    args = sys.argv

    data_path   = "./data/od_cars"
    voc_classes = ["car"]

    num_epochs = 400

    if len(args) >= 2:
        num_epochs = int(args[1])

    main(num_epochs, data_path, voc_classes)
