# Transform Utils

このディレクトリには画像変換に関連するユーティリティが含まれています．

## 実装されている変換

- `cutout_transforms.py`: CutOutデータ拡張の実装
  - 画像の一部をランダムにマスクする手法
  - 論文: https://arxiv.org/abs/1708.04552

- `simclr_transforms.py`: SimCLRの画像変換の実装
  - 自己教師あり学習のための2つの異なる画像変換を生成
  - ランダムクロップ，色変換，グレースケール変換などを含む
  - 論文: https://arxiv.org/abs/2002.05709

- `transform_dict.py`: 変換関数のマッピング
  - 設定ファイルで指定された変換名と実際の変換クラスのマッピングを管理
  - 現在サポートされている変換:
    - SimCLRTransform
    - CutOut

