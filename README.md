# JetRacer Minimal — アノテーション手順マニュアル

このリポジトリで使う「車体マスクの生成」と「自動アノテーション」の手順をまとめたマニュアルです。

## 目次
- 前提
- 車体マスクの作り方
- 自動アノテーション (OneFormer)
- ラベルの説明
- 可視化と出力場所
- トラブルシューティング

---

## 前提
- Python 3.10 以上、必要なライブラリは `requirements.txt` を参照してください。
- 仮想環境を使うことを推奨します（例: `python -m venv .venv && source .venv/bin/activate`）。

## 車体マスクの作り方
セッションごとに車体（常に画面下部に映る自車の車体）を検出してバイナリマスクを作成します。

例：`data/raw_images/session_YYYYmmdd_HHMMSS` に画像が格納されている場合、以下を実行します。

```bash
python scripts/generate_vehicle_mask.py \
  --input data/raw_images/session_20251127_153358 \
  --output data/vehicle_masks/session_20251127_153358.png \
  --num-samples 6 \
  --visualize
```

- 出力:
  - `data/vehicle_masks/session_XXXX.png` (車体マスク, 0/255)
  - `data/vehicle_masks/session_XXXX_visualization.jpg` (デバッグ用オーバーレイ)

## 自動アノテーション (OneFormer)
OneFormer を使ってセマンティックセグメンテーションを実行し、ADE20K のクラスを JetRacer 用のクラスに変換します。
生成した `vehicle_mask` が存在すると、自動的に各画像へ適用して車体領域を「対象外（ラベル3）」に上書きします。

セッション単位で実行する例:

```bash
python scripts/auto_annotate.py \
  --input data/raw_images/session_20251127_153358 \
  --output data/annotations/auto_masks/session_20251127_153358 \
  --visualize
```

フォルダ単位（全セッション）で一括処理する例:

```bash
python scripts/auto_annotate.py --input data/raw_images --output data/annotations/auto_masks --visualize
```

注意点:
- `data/vehicle_masks/<session>.png` を自動検出します。セッションディレクトリは `session_` で始まる名前にしてください。

## ラベルの説明
- `0` : Background（背景）
- `1` : Road（道路／床）
- `2` : Obstacle（障害物）
- `3` : Vehicle / Ignored（車体／対象外） — 生成された `vehicle_mask` の領域はこのラベルに上書きされます。

保存方法: マスクは値を保持するため `PIL.Image.fromarray(...).save(...)` 経由で保存しています。

## 可視化と出力場所
- マスク: `data/annotations/auto_masks/<session>/masks/*.png` (ラベル画像)
- 可視化: `data/annotations/auto_masks/<session>/visualizations/*.jpg` (オーバーレイ表示)
- 色割り当て: Background=灰、Road=緑、Obstacle=赤、Vehicle(対象外)=濃い灰

## トラブルシューティング
- 車体マスクが適用されない／反映されない
  - `data/vehicle_masks/<session>.png` が存在するか確認してください。
  - セッション名は `session_` で始まっていますか？スクリプトは画像パスの親ディレクトリから `session_` を探してマスクを割り当てます。
  - マスクは 0/255 の2値画像です。必要なら `cv2.IMREAD_GRAYSCALE` で読み込み、`mask > 127` で2値化してください。

- 可視化が壊れている（極小画像や黒一色になる）
  - 以前のバージョンでは `cv2.imwrite` に誤った引数を渡していました。現在は `cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)` を使って BGR に変換して保存しています。

- マスクのラベルが消える／3が保存されない
  - マスクの保存は `PIL` を使っています。もし `cv2.imwrite` を使うとラベル値が変換される場合があるため注意してください。

## 開発者向けメモ
- デバッグ出力は `scripts/auto_annotate.py` に一時的に入れています。確認後はログ出力（`logging`）に移すか削除してください。

---

必要があれば、この README を元に `docs/` 下へ詳細フローや画像付きガイドを作成できます。どの形式が良いか教えてください。
# JetRacer Minimal - Phase 1

自律走行システムのデータ収集と学習パイプライン

## 使い方

```bash
./start_jetracer.sh
```

ブラウザ: http://192.168.1.65:8080

## データ収集

1. Start New Session
2. CAPTURE を連打（50-100回）
3. End Session

保存先: `data/raw_images/session_YYYYMMDD_HHMMSS/`
