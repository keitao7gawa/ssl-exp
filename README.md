# 自己教師あり学習実験

このプロジェクトは，SimCLRフレームワークを用いた自己教師あり学習実験の実装です．対照学習による表現学習と，その後の線形評価プロトコルによる性能評価を行うことを主な目的としています．

## プロジェクト構成

```
project/
├── src/
│   ├── train.py                   ← 学習スクリプト
│   ├── model_wrapper.py           ← モデルラッパー
│   ├── logger.py                  ← ロギング機能
│   ├── dataset.py                 ← データセット定義
│   ├── models/
│   │   ├── resnet.py              ← ResNetベースエンコーダー
│   │   └── simclr.py              ← SimCLRモデル定義
│   ├── utils/
│   │   ├── losses.py              ← 損失関数
│   │   ├── transform_dict.py      ← データ拡張定義
│   │   └── transform/             ← データ拡張実装
│   └── optimizer/                 ← 最適化手法
├── notebooks/                     ← 分析用ノートブック
├── runs/                          ← 実験結果・ログの保存先
├── configs/                       ← 設定ファイル
│   └── default.yaml               ← デフォルト設定
├── data/                          ← データセット
├── .gitignore
├── README.md
└── environment.yml                ← Anaconda環境設定ファイル
```

## 主な機能

- SimCLRフレームワークによる自己教師あり学習
- ResNet18をベースとしたエンコーダーと射影ヘッド
- 対照学習用のデータ拡張

## セットアップ

1. 環境の作成とアクティベート：
```bash
# 環境の作成
conda env create -f environment.yml

# 環境のアクティベート
conda activate sslexp
```

2. 学習の実行：
```bash
# カスタム設定ファイルを使用して学習を実行
python src/train.py --config configs/default.yaml

# チェックポイントから学習を再開
python src/train.py --resume runs/experiment_name/checkpoints/latest.pth
```