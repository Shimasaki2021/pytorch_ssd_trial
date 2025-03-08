# pytorch_ssd_trial
pytorch ssdの転移学習を実行するソース一式です。

## ファイル構成

| ディレクトリ,ファイル | 説明 |
|---|---|
| utils/ | SSDモデル実装等のソース（モデルはVGG16ベースのSSD300）  |
| common_ssd.py | 推論/学習共通部分のソース |
| train_ssd.py | 学習実行ソース |
| predict_ssd.py | 検知（推論）実行ソース |
| movie_player.py | 動画(mp4)から学習用画像切り出しツール(ソース) |
| data/od_cars_org_F00000.jpg | テストデータ（推論用画像） |
| data/od_cars_sample/ | テストデータ（学習用画像、アノテーションデータ） |
| weights/ssd_best_od_cars.pth | 学習済みSSD重みデータ(車、ナンバープレートを学習済) |
| weights/ssd_best_od_cars_classes.txt | ssd_best_od_cars.pthで学習したクラス名 |
| python_version.txt | 動作確認したpythonバージョン |
| python_module_version.txt | 動作確認したpythonモジュールのバージョン |

utils以下のソースは、以下掲載ソースをベースに、上述のpythonバージョンで動作するよう一部修正したものです。

https://github.com/hituji1012/od_test

参考までに、ソースのクラス図です（主要クラスのみ。Debugクラス等は省略）。
![クラス図](./fig/soft_structure.png)

## 検知（推論）実行方法

ターミナルで以下を実行します。
```
./predict_ssd.py [動画(mp4) or 画像ファイルパス]
```

結果は、output.cuda or output.cpuディレクトリ以下に出力されます。

実行結果例
```
./predict_ssd.py data/od_cars_org_F00000.jpg
```
![実行結果例](./fig/od_cars_org_F00000_result.jpg)

入力画像/動画の解像度は1280x800、検出対象は車＆ナンバープレートの想定で、predict_ssd.py末尾の以下コードで検出範囲を設定しています（車がいそうな領域を4分割）。

異なる解像度の画像や、検出対象がいそうな領域が違う場合は、以下の検出範囲を適宜編集してご利用ください。現状、境界値チェックを入れれておらず、はみ出すと落ちてしまうので、ご注意ください。

```predict_ssd.py
# 検出範囲
#   (1280x720を)300x300/350x350に切り出し
img_procs = [ImageProc(180, 250, 530, 600), 
             ImageProc(480, 200, 780, 500), 
             ImageProc(730, 200, 1030, 500), 
             ImageProc(930, 250, 1280, 600)] 
```

なお、学習済みSSD重みデータ（ssd_best_od_cars.pth）は、車を検出対象に入れているものの、側面は意図的に外してます（※）。そのため、真横の車は検出できないことがあるので、ご了承ください。

（※）∵側面を入れると防音壁の誤検出が取れない。また、ナンバープレートを自動でぼかすアプリを作りたくて学習したデータのため、側面は不要。車の検出結果は補助的に使う想定。

## 学習実行方法

以下を実行します。

1. VGG16の学習済み重みをダウンロード
1. 学習データ作成（アノテーション）
1. 学習実行

### 1. VGG16の学習済み重みをダウンロード

以下から、vgg16_reducedfc.pthファイルをダウンロードし、weightsディレクトリに置いてください。
https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

### 2. 学習データ作成（アノテーション）

検出対象が映っている画像をできるだけたくさん集め、検出対象を囲む枠をxmlファイルで作成します。私は、labelImgで作成しました。

学習済みSSD重みデータ（weights/ssd_best_od_cars.pth）を学習したときのデータの一部をdata/od_cars_sample/　に置きましたので、参考にしていただけたらと思います（ナンバープレートが映った画像が多く含まれるので全部は公開できません。。）。

data/od_cars_sampleは、labelImgでopenすればアノテーション結果を確認／編集できます（下図）。xmlファイルにはファイルの絶対パスが書かれてますが、このパス以外に置いても問題なく開けます。

labelImgは、以下リンクのファイル一式から構築可能なdockerコンテナにインストール済みです。今回のソースもそのまま動かせますので、よかったらご活用ください。

https://github.com/Shimasaki2021/docker_pytorch

![labelImg例](./fig/labelImg_open_od_cars_sample.png)

画像を動画(mp4)から収集する場合は、movie_player.py を使って画像切り出しができます。

```movie_player.py
./movie_player.py [動画(mp4)ファイルパス] ([fps])

※実行例(動画ファイルNNF_230504-092531.mp4から、0.5fpsで画像切り出し)
./movie_player.py data/NNF_230504-092531.mp4 0.5
```

切り出し領域は、movie_player.py 末尾付近の以下を適宜編集してご利用ください。predict_ssd.pyの検出範囲と同じ設定にする必要はありません。

```movie_player.py
# 切り出し領域
img_procs = [ImageProc(180, 150, 530, 500), 
             ImageProc(580, 200, 880, 500), 
             ImageProc(930, 250, 1280, 600)]
```

### 3. 学習実行

train_ssd.pyの以下を編集します。

検出対象に合わせて編集が必須なのは上２つ（「学習データを置いたディレクトリパス」「学習クラス名」）で、あとは（おそらく）そのままでも使えるかと思います。

```train_ssd.py
# 学習データを置いたディレクトリパス
data_path    = "./data/od_cars"

# 学習クラス名
voc_classes  = ["car","number"]

# SSDモデルで重みを更新しないレイヤー（入力層～freeze_layerまでの重みは更新しない）
freeze_layer = 5

# epoch数（引数指定なしの場合のDefault値）
num_epochs = 500

# バッチサイズ
batch_size = 16

# 検証用画像の割合（全体のtest_rateを検証用画像にする）
test_rate  = 0.1
```

編集後は、ターミナルで以下を実行します。
実行後は、weights/以下に、pthファイルと、学習クラス名が書かれたtxtが出力されます。

```
./train_ssd.py ([epoch数])
```

学習にかかる時間は、PC環境や学習データ数に大きく依存します。

参考までに、以下PC環境、学習データ数で、ssd_best_od_cars.pthを学習するのに、約3時間かかりました。

- PC環境
  - CPU: AMD Ryzen 7 3700X (3.60 GHz)
  - GPU: NVIDIA GeForce GTX 1660 SUPER
  - OS: Windows 11 Home (24H2） , WSL2 + Ubuntu24.04

- 学習データ数（※検証用（1割）込み）
  - 画像数: 530
  - 物体数（枠の数）: 1394　（車：857、ナンバープレート：537）

