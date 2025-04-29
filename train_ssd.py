#!/usr/bin/env python3

# パッケージのimport
import os
import sys
import time
import numpy as np
from typing import List,Dict,Any
from io import TextIOWrapper
import datetime

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision

from utils.ssd_model import SSD as SSD_vgg
from utils.ssd_model import MultiBoxLoss

from common_ssd import SSDModel, VocDataSetMng, Logger, makeVocClassesTxtFpath

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.ssd import SSD as SSD_mb2
# from vision.ssd.ssd import _xavier_init_

class SSDModelTrainerDebug(Logger):
    """ ログ生成（SSDモデル学習）
    """
    def __init__(self, is_out:bool, max_epoch:int):
        """ コンストラクタ

        Args:
            is_out (bool)   : 出力有無(Falseならログ出力しない)
            max_epoch (int) : 詳細ログ出力（画像等）するepoch（最大値）
        """
        super().__init__(is_out)

        self.debug_out_max_epoch_  = max_epoch
        self.res_weight_fpath_:str = ""

        t_delta   = datetime.timedelta(hours=9)
        self.JST_ = datetime.timezone(t_delta, 'JST')
        return

    def openLogFile(self, dev_name:str, net_type:str, res_weight_fpath:str, fname:str):
        """ ログファイルopen

        Args:
            dev_name (str)          : デバイス(cpu or cuda)
            net_type (str)          : SSD種別（vgg or mobilenet）
            res_weight_fpath (str)  : 出力パラメータファイル名
            fname (str)             : ログファイル名
        """
        if self.is_out_ == True:
            self.res_weight_fpath_ = res_weight_fpath

            res_weight_fname   = os.path.splitext(os.path.basename(res_weight_fpath))[0]
            super().openLogFile(dev_name, net_type, f"train.{res_weight_fname}", fname, "w")
        return

    def outputLogModuleInfo(self, mod_name:str, mod_list:nn.ModuleList):
        """ SSDモデルモジュール情報出力

        Args:
            mod_name (str)          : モジュール名
            mod_list (nn.ModuleList): モジュールリスト
        """
        if self.isOutputLog() == True:
            for idx,module in enumerate(mod_list):
                self.log_fp_.write(f"[{mod_name}{idx}] {module}")

                grads = [p.requires_grad for p in module.parameters()]

                if len(grads) > 0:              
                    self.log_fp_.write(" requires_grad=")
                    for grad in grads: 
                        self.log_fp_.write(f"{grad},")

                self.log_fp_.write("\n")
        return
    
    def outputLogNetInfo(self, net):
        """ SSDモデル情報出力

        Args:
            net (_type_): SSDモデル
        """
        if self.isOutputLog() == True:
            if isinstance(net, SSD_mb2) == True:
                self.outputLogNetInfoSSDmb2(net)
            else:
                self.outputLogNetInfoSSDvgg(net)
        return

    def outputLogNetInfoSSDvgg(self, net:SSD_vgg):
        """ SSDモデル情報出力（VGGベースSSD）

        Args:
            net (SSD_vgg): SSDモデル（VGGベース）
        """
        self.log_fp_.write("\n == net(vgg) ==\n")
        self.outputLogModuleInfo("vgg",    net.vgg)
        self.outputLogModuleInfo("extras", net.extras)
        self.outputLogModuleInfo("loc",    net.loc)
        self.outputLogModuleInfo("conf",   net.conf)
        self.log_fp_.write("\n")
        return
    def outputLogNetInfoSSDmb2(self, net:SSD_mb2):
        """ SSDモデル情報出力（mobilenetベースSSD）

        Args:
            net (SSD_mb2): SSDモデル（mobilenetベース）
        """
        self.log_fp_.write("\n == net(mobile net v2 lite) ==\n")
        self.outputLogModuleInfo("basenet",                net.base_net)
        self.outputLogModuleInfo("extras",                 net.extras)
        self.outputLogModuleInfo("regression_headers",     net.regression_headers)
        self.outputLogModuleInfo("classification_headers", net.classification_headers)
        self.log_fp_.write("\n")
        return
    
    def outputLogSummary(self,dev_name:str, learn_data_path:str, epoch:int, val_loss:float):
        """ 学習概要を出力

        Args:
            dev_name (str)       : デバイス(cpu or cuda)
            learn_data_path (str): 学習データパス（ディレクトリ）
            epoch (int)          : epoch数
            val_loss (float)     : パラメータ出力時の損失（val loss）
        """
        if self.isOutputLog() == True:
            self.log_fp_.write("\n == summary ==\n")
            self.log_fp_.write(f",device:,,{dev_name}\n")
            self.log_fp_.write(f",learn data:,,{learn_data_path}\n")
            self.log_fp_.write(f",out weight file:,,{self.res_weight_fpath_}\n")
            self.log_fp_.write(f",epoch at out weight:,,{epoch}\n")
            self.log_fp_.write(f",val loss at out weight:,,{val_loss}\n")
        return
    
    def outputLogEpochSummary(self, epoch:int, time_sec:float, train_loss:float, val_loss:float, train_iter:int, val_iter:int):
        """ epoch毎の結果出力

        Args:
            epoch (int)         : epoch
            time_sec (float)    : 処理時間[sec]
            train_loss (float)  : 損失(train loss)
            val_loss (float)    : 損失(val loss)
            train_iter (int)    : iteration数（train）
            val_iter (int)      : iteration数（val）
        """
        if self.isOutputLog() == True:
            now     = datetime.datetime.now(self.JST_)
            now_str = now.strftime("%Y%m%d_%H%M%S")
            self.log_fp_.write(f"epoch,{epoch},{time_sec:.4f},sec,train_Loss,{train_loss:.4f},(it:,{train_iter},),val_Loss,{val_loss:.4f},(it:,{val_iter},),{now_str}\n")
        return
    
    def outputLogDataSetSummary(self, voc_dataset:VocDataSetMng):
        """ 学習データ概要を出力

        Args:
            voc_dataset (VocDataSetMng): 学習データセット
        """
        if self.isOutputLog() == True:
            self.log_fp_.write("\n == dataset info ==\n")
            self.log_fp_.write(f"batch_size: train={voc_dataset.batch_size_}, val={voc_dataset.batch_size_val_}\n\n")

            for phase in ["train", "val"]:
                image_num   = voc_dataset.getImageNum(phase)

                self.log_fp_.write(f"{phase}\n")
                self.log_fp_.write(f",image_num=,,{image_num}\n")

                for class_name in voc_dataset.voc_classes_:
                    (obj_num, obj_w_ave, obj_h_ave) = voc_dataset.getAnnoInfo(phase, class_name)
                    self.log_fp_.write(f",class:,{class_name}\n")
                    self.log_fp_.write(f",,obj_num=,,,{obj_num}\n")
                    self.log_fp_.write(f",,obj_size(ave)(w x h)=,,,{int(obj_w_ave)},{int(obj_h_ave)}\n")

        return

    def dumpInputImage(self, epoch:int, images:List[torch.Tensor], phase:str, batch_idx:int):
        """  詳細ログ出力（学習画像（前処理後）のダンプ）

        Args:
            epoch (int)                 : epoch
            images (List[torch.Tensor]) : 画像（前処理後）
            phase (str)                 : train or val
            batch_idx (int)             : バッチindex
        """
        if (self.is_out_ == True) and (epoch <= self.debug_out_max_epoch_):
            for image_no in range(images.shape[0]):
                debug_out_fpath = self.outdir_ + "/epoch" + str(epoch) + "_" + phase + "_b" + str(batch_idx) + "_" + str(image_no) + ".jpg"
                torchvision.utils.save_image(images[image_no], debug_out_fpath)
        return

# SSDモデル作成＆学習
class SSDModelTrainer(SSDModel):
    """ SSDモデル（学習用）
    """

    def __init__(self, device:torch.device, net_type:str, weight_fpath:str, voc_classes:List[str], freeze_layer:int):
        """ コンストラクタ

        Args:
            device (torch.device)   : デバイス（cpu or cuda)
            net_type (str)          : SSD種別（vgg or mobilenet）
            weight_fpath (str)      : パラメータ(重み)ファイルのパス
            voc_classes (List[str]) : クラス
            freeze_layer (int)      : パラメータ更新freezeレイヤ（0～freeze_layerまでをfreeze)
        """
        super().__init__(device, voc_classes, net_type)

        # SSDネットワークモデル
        if self.net_type_ == "mb2-ssd":
            # mobilenet-v2-liteベースSSD
            self.net_ = self.createSSDmb2(self.num_classes_, weight_fpath, freeze_layer)
        else:
            # VGGベースSSD
            self.net_ = self.createSSDvgg(weight_fpath, self.ssd_vgg_cfg_, freeze_layer)

        self.net_.to(device)

        # 損失関数の設定
        self.criterion_ = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

        # 最適化手法の設定
        #self.optimizer_ = optim.SGD(self.net_.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        #self.optimizer_ = optim.Adam(self.net_.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_ = optim.AdamW(self.net_.parameters(), lr=0.001)

        return
    
    @staticmethod
    def createSSDvgg(weight_fpath:str, cfg_vgg:Dict[str,Any], freeze_layer:int) -> SSD_vgg:
        """ VGG16ベースSSD作成＆パラメータ初期値設定

        Args:
            weight_fpath (str)      : ベースネット(VGG16)パラメータファイル
            cfg_vgg (Dict[str,Any]) : SSD用config
            freeze_layer (int)      : パラメータ更新freezeレイヤ（0～freeze_layerまでをfreeze)

        Returns:
            SSD_vgg: SSDモデル(VGG16ベース)
        """
        # SSD作成
        net = SSD_vgg(phase="train", cfg=cfg_vgg)

        # ベースnetの初期パラメータ設定＆freeze
        weight_data = torch.load(weight_fpath, weights_only=True) # FutureWarning: You are using torch.load..対策
        net.vgg.load_state_dict(weight_data) 

        for idx,module in enumerate(net.vgg):
            if idx <= freeze_layer:
                for param in module.parameters():
                    param.requires_grad = False

        # Heの初期値を適用
        net.extras.apply(SSDModelTrainer.initWeight)
        net.loc.apply(SSDModelTrainer.initWeight)
        net.conf.apply(SSDModelTrainer.initWeight)

        return net

    @staticmethod
    def createSSDmb2(num_classes:int, weight_fpath:str, freeze_layer:int) -> SSD_mb2:
        """ mobilenet-v2-liteベースSSD作成＆パラメータ初期値設定

        Args:
            num_classes (int)   : クラス数
            weight_fpath (str)  : ベースネット(mobilenet-v2-lite)パラメータファイル
            freeze_layer (int)  : パラメータ更新freezeレイヤ（0～freeze_layerまでをfreeze)

        Returns:
            SSD_mb2: SSDモデル(mobilenet-v2-liteベース)
        """
        # SSD作成
        net = create_mobilenetv2_ssd_lite(num_classes)

        # ベースnetの初期パラメータ設定＆freeze
        weight_data = torch.load(weight_fpath, weights_only=True) # FutureWarning: You are using torch.load..対策
        net.base_net.load_state_dict(weight_data) 

        for idx,module in enumerate(net.base_net):
            if idx <= freeze_layer:
                for param in module.parameters():
                    param.requires_grad = False

        # Xavierの初期値を適用
        # net.source_layer_add_ons.apply(_xavier_init_)
        # net.extras.apply(_xavier_init_)
        # net.classification_headers.apply(_xavier_init_)
        # net.regression_headers.apply(_xavier_init_)

        # Heの初期値を適用
        net.source_layer_add_ons.apply(SSDModelTrainer.initWeight)
        net.extras.apply(SSDModelTrainer.initWeight)
        net.classification_headers.apply(SSDModelTrainer.initWeight)
        net.regression_headers.apply(SSDModelTrainer.initWeight)
        return net


    @staticmethod
    def initWeight(m):
        """ Heの初期化

        Args:
            m (_type_): 初期化するモジュール
        """
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:  # バイアス項がある場合
                nn.init.constant_(m.bias, 0.0)
        return

    def train(self, voc_dataset:VocDataSetMng, num_epochs:int, res_weight_fpath:str):
        """ SSDモデル学習

        Args:
            voc_dataset (VocDataSetMng) : 学習データ
            num_epochs (int)            : epoch数
            res_weight_fpath (str)      : 出力パラメータファイル名
        """
        dataloaders_dict = voc_dataset.dataloaders_dict_

        # debug
        debug_train = SSDModelTrainerDebug(True, 1)
        debug_train.openLogFile(self.device_.type, self.net_type_, res_weight_fpath, "train_log.csv")

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
                        targets = [target.to(self.device_) for target in targets]  

                        # debug:入力画像ダンプ
                        debug_train.dumpInputImage(epoch, images, phase, batch_idx)

                        # optimizerを初期化
                        self.optimizer_.zero_grad()

                        # 順伝搬（forward）計算
                        with torch.set_grad_enabled(phase == "train"):
                            # 順伝搬（forward）/損失の計算
                            outputs        = self.net_(images)
                            loss_l, loss_c = self.criterion_(outputs, targets)
                            loss           = loss_l + loss_c

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
        debug_train.outputLogNetInfo(self.net_)
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


def main(cfg:Dict[str,Any]):
    """ メイン（SSDモデル作成、学習実行）

    Args:
        cfg (Dict[str,Any]): config
    """

    num_epochs:int        = cfg["train_num_epoch"]
    data_path:str         = cfg["train_data_path"]
    voc_classes:List[str] = cfg["train_voc_classes"]
    test_rate:float       = cfg["train_data_test_rate"]
    batch_size:int        = cfg["train_batch_size"]

    freeze_layer:int      = cfg["ssd_model_freeze_layer"]
    net_type:str          = cfg["ssd_model_net_type"]
    weight_fpath:str      = cfg["ssd_model_init_weight_fpath"]

    if os.path.isdir(data_path) == False:
        print("Error: ", data_path, " is nothing.")
    
    else:
        # SSDモデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)

        ssd_model = SSDModelTrainer(device, net_type, weight_fpath, voc_classes, freeze_layer)
    
        # データセット作成
        voc_dataset = VocDataSetMng(data_path, voc_classes, ssd_model.input_size_, ssd_model.color_mean_, ssd_model.color_std_, batch_size, test_rate)
        for phase in ["train", "val"]:
            for class_name in voc_dataset.voc_classes_:
                (obj_num, _, _) = voc_dataset.getAnnoInfo(phase, class_name)
                print(f"DataSet({phase}): num={obj_num} (class:{class_name})")

        # SSDモデルの学習
        res_weight_fpath = f"./weights/{net_type}_best_{os.path.basename(data_path)}.pth"
        ssd_model.train(voc_dataset, num_epochs, res_weight_fpath)

    return

if __name__ == "__main__":
    args = sys.argv

    cfg = {
        # 学習データを置いたディレクトリ
        "train_data_path"    : "./data/od_cars",

        # 学習クラス名
        "train_voc_classes"  : ["car","number"],

        # 検証用画像の割合（全体のtest_rateを検証用画像にする）
        "train_data_test_rate" : 0.1,

        # epoch数（引数指定なしの場合のDefault値）
        "train_num_epoch" : 500,

        # バッチ処理数（＝検出範囲数 x フレーム数）
        "train_batch_size" : 16,

        # (SSDモデル:VGGベース)ネットワーク種別/ベースnet初期パラメータ/freezeレイヤー(※)
        #    (※) 入力層～freeze_layer層までの重みは更新しない
        # "ssd_model_net_type"          : "vgg16-ssd",
        # "ssd_model_init_weight_fpath" : "./weights/vgg16_reducedfc.pth", 
        # "ssd_model_freeze_layer"      : 5,

        # (SSDモデル:mobilenetベース)ネットワーク種別/ベースnet初期パラメータ
        "ssd_model_net_type"          : "mb2-ssd",
        "ssd_model_init_weight_fpath" : "./weights/mb2-imagenet-71_8.pth", 
        "ssd_model_freeze_layer"      : 0,

    }

    if len(args) >= 2:
        cfg["train_num_epoch"] = int(args[1])

    main(cfg)
