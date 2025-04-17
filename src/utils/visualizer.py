import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class SimCLRVisualizer:
    """SimCLRの学習過程を可視化するクラス
    
    学習中のメトリクスをプロットし，結果を画像として保存します．
    
    Attributes:
        save_dir (Path): 画像の保存先ディレクトリ
    """
    
    def __init__(self, save_dir: str):
        """SimCLRVisualizerの初期化
        
        Args:
            save_dir (str): 画像の保存先ディレクトリ
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_metrics(
        self,
        metrics_path: str,
        save_name: Optional[str] = None,
        save: bool = True
    ) -> None:
        """学習メトリクスをプロットします
        
        Args:
            metrics_path (str): metrics.csvファイルのパス
            save_name (Optional[str]): 保存するファイル名
            save (bool): 画像を保存するかどうか
        """
        # CSVファイルの読み込み
        df = pd.read_csv(metrics_path, header=None, names=['epoch', 'train_loss', 'val_loss', 'lr', 'temperature'])
        
        plt.figure(figsize=(12, 8))
        
        # 損失のプロット
        plt.subplot(2, 1, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        # 学習率のプロット
        plt.subplot(2, 1, 2)
        plt.plot(df['epoch'], df['lr'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 画像の保存
        if save:
            save_name = save_name or f'metrics_plot.png'
            plt.savefig(self.save_dir / save_name)
            plt.close()