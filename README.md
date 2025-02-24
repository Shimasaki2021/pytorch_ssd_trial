# pytorch_ssd_trial
pytorch ssdの転移学習を実行するソース一式です。

## ファイル構成

| ディレクトリ,ファイル | 説明 |
|---|---|
| utils/ | SSDモデル実装等のソース（モデルはVGG16ベースのSSD300）  |
| common_ssd.py | 推論/学習共通部分のソース |
| train_ssd.py | 学習実行ソース |
| predict_ssd.py | 推論実行ソース |
| movie_player.py | 動画(mp4)から学習用画像切り出しツール(ソース) |
| data/od_cars_org_F00000.jpg | テストデータ（推論用画像） |
| data/od_cars_sample/ | テストデータ（学習用画像、アノテーションデータ） |
| weights/ssd_best_od_cars.pth | 学習済みSSD重みデータ(車、ナンバープレートを学習済) |
| weights/ssd_best_od_cars_classes.txt | ssd_best_od_cars.pthで学習したクラス名 |
| python_version.txt | 動作確認したpythonバージョン |
| requires.txt | 動作確認したpythonモジュールのバージョン |

SSDモデル実装等のソース(utils以下のソース)は、以下掲載ソースをベースに、上述のpythonバージョンで動作するよう一部修正したものです。

https://github.com/YutaroOgawa/pytorch_advanced/tree/master/2_objectdetection/utils

## 検知（推論）実行方法

ターミナルで以下を実行
```
./predict.py [動画(mp4) or 画像ファイルパス]
```

結果は、output.cuda or output.cpuディレクトリ以下に出力されます。

実行結果例
```
./predict.py data/od_cars_org_F00000.jpg
```
![実行結果例](./fig/od_cars_org_F00000_result.jpg)

入力画像/動画の解像度は1280x800、検出対象は車＆ナンバープレートの想定で、predict_ssd.py末尾の以下コードで検出範囲を設定しています（車がいそうな領域を4分割）。

異なる解像度の画像や、検出対象がいそうな領域が違う場合は、以下を適宜編集してご利用ください。現状、境界値チェックを入れれておらず、はみ出すと落ちてしまうので、ご注意ください。

```predict.py
# 検出範囲
#   (1280x720を)300x300/350x350に切り出し
img_procs = [ImageProc(110, 250, 530, 600), 
             ImageProc(480, 200, 780, 500), 
             ImageProc(730, 200, 1030, 500), 
             ImageProc(930, 250, 1280, 600)] 
```

なお、学習済みSSD重みデータ（weights/ssd_best_od_cars.pth）は、車を検出対象に入れているものの、側面は意図的に外してます（※）。そのため、真横の車は検出できないことがあるので、ご了承ください。

（※）∵側面を入れると防音壁の誤検出が取れない。また、ナンバープレートを自動でぼかすアプリを作りたくて学習したデータのため、側面は不要。

## 学習実行方法

