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
├── environment.yml                ← Linux環境用Anaconda設定
└── environment_cross_platform.yml ← クロスプラットフォーム用Anaconda設定
```

## 主な機能

- SimCLRフレームワークによる自己教師あり学習
- ResNet18をベースとしたエンコーダーと射影ヘッド
- 対照学習用のデータ拡張

## セットアップ

1. 環境の作成とアクティベート：
```bash
# Linux環境の場合
conda env create -f environment.yml

# その他の環境の場合
conda env create -f environment_cross_platform.yml

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

## 環境設定について

このプロジェクトでは，2つの環境設定ファイルを提供しています：

1. `environment.yml`
   - Linux環境向けの詳細な設定
   - プラットフォーム固有の依存関係を含む
   - 既存の環境との互換性を保持

2. `environment_cross_platform.yml`
   - クロスプラットフォーム対応の設定
   - プラットフォーム依存のパッケージを最小限に
   - 開発ツール（black，flake8，mypy）を含む

必要に応じて適切な環境設定ファイルを選択してください．また，CUDAバージョンなどのプラットフォーム固有の設定は，環境に合わせて調整してください．