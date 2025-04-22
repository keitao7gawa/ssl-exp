# Transform Utils

このディレクトリには画像変換に関連するユーティリティが含まれています．

## 実装されている変換

- `cutout_transforms.py`: CutOutデータ拡張の実装
  - 画像の一部をランダムにマスクする手法
  - 論文: https://arxiv.org/abs/1708.04552

- `simclr_transforms.py`: SimCLRの画像変換の実装
  - 自己教師あり学習のための任意の数の異なる画像変換を生成
  - ランダムクロップ，色変換，グレースケール変換などを含む
  - 論文: https://arxiv.org/abs/2002.05709

- `moco_transforms.py`: MoCoV1の画像変換の実装
  - 自己教師あり学習のための任意の数の異なる画像変換を生成
  - 論文: https://arxiv.org/abs/1911.05722

- `hs_colorjitter.py`: HS画像用の色歪み変換クラス
  - MoCoや SimCLRのデータ拡張に必要な ColorJitterの自作変換クラス
  - `torchvision.transforms.ColorJitter`は HS画像には使えない

- `hs_transforms.py`: HS画像用の画像変換の実装
  - HS画像用に SimCLRや MoCoV1の画像変換を生成
  - `hs_colorjitter.py`の自作変換クラスを使用

## 変換マッピング定義
設定ファイルで指定された変換名と実際の変換クラスとの対応関係を定義するマッピングが含まれています．
- `transform_dict.py`: 変換関数のマッピング
  - 設定ファイルで指定された変換名と実際の変換クラスのマッピングを管理
  - 現在サポートされている変換:
    - SimCLRTransform
    - CutOut
    - MoCoTransform
    - HSSimCLRTransform
    - HSMoCoTransform

