import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

class ExperimentVisualizer:
    """実験結果を可視化するクラス
    
    学習中のメトリクスをプロットし，複数の実験結果を比較・分析するための機能を提供します．
    
    Attributes:
        save_dir (Path): 画像の保存先ディレクトリ（指定しない場合はカレントディレクトリ）
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """ExperimentVisualizerの初期化
        
        Args:
            save_dir (Optional[str]): 画像の保存先ディレクトリ（デフォルト: None）
        """
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        if save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # プロットのスタイル設定
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズを大きく
        plt.rcParams['figure.titlesize'] = 18
        
        # カラーパレットの設定
        self.colors = ['#2E86C1', '#E67E22', '#27AE60', '#8E44AD', '#C0392B']

    def _load_metrics(self, metrics_path: str) -> pd.DataFrame:
        """メトリクスファイルを読み込みます
        
        Args:
            metrics_path (str): metrics.csvファイルのパス
            
        Returns:
            pd.DataFrame: 読み込んだメトリクスデータ
        """
        # CSVファイルを読み込み（ヘッダーあり）
        df = pd.read_csv(metrics_path)
        # epochでソート
        df = df.sort_values('epoch')
        return df

    def _format_axis(self, ax: plt.Axes, metric: str) -> None:
        """軸の表示形式を設定します
        
        Args:
            ax (plt.Axes): 対象の軸
            metric (str): メトリクスの名前
        """
        # y軸のフォーマット設定
        if metric in ['lr', 'momentum']:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        elif metric in ['temperature']:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        elif metric in ['queue_size']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # グリッドの設定
        ax.grid(True, alpha=0.3)
        
        # 軸ラベルの設定
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # 軸の目盛り数を調整
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # x軸の範囲を設定
        ax.set_xlim(left=0)

    def _format_legend(self, ax: plt.Axes, outside: bool = True) -> None:
        """凡例の表示形式を設定します
        
        Args:
            ax (plt.Axes): 対象の軸
            outside (bool): 凡例をグラフの外に表示するかどうか
        """
        if outside:
            # グラフの右側に凡例を表示
            ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=14,  # 凡例のフォントサイズ
                markerscale=2,  # マーカーのサイズを2倍に
                handlelength=3,  # 凡例の線の長さを調整
                borderpad=1,  # 凡例の枠とテキストの間隔
                labelspacing=1.2,  # 凡例の項目間の間隔
            )
        else:
            # グラフ内に凡例を表示
            ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                loc='best',
                fontsize=14,
                markerscale=2,
                handlelength=3,
                borderpad=1,
                labelspacing=1.2,
            )

    def _save_or_show(self, save: bool, save_name: Optional[str], default_name: str) -> None:
        """プロットを保存または表示します
        
        Args:
            save (bool): 保存するかどうか
            save_name (Optional[str]): 保存するファイル名
            default_name (str): デフォルトのファイル名
        """
        if save:
            save_name = save_name or default_name
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_experiment(
        self,
        experiment_path: Union[str, Tuple[str, str]],
        metrics: List[str] = ['train_loss', 'val_loss', 'lr'],
        figsize: Tuple[int, int] = (15, 10),
        save: bool = False,
        save_name: Optional[str] = None,
        legend_outside: bool = True
    ) -> None:
        """単一の実験結果をプロットします
        
        Args:
            experiment_path (Union[str, Tuple[str, str]]): 
                実験のmetrics.csvファイルのパス，または（実験名，パス）のタプル
            metrics (List[str]): 表示するメトリクスのリスト
            figsize (Tuple[int, int]): 図のサイズ
            save (bool): 画像を保存するかどうか（デフォルト: False）
            save_name (Optional[str]): 保存するファイル名
            legend_outside (bool): 凡例をグラフの外に表示するかどうか
        """
        if isinstance(experiment_path, tuple):
            exp_name, path = experiment_path
        else:
            exp_name = Path(experiment_path).parent.name
            path = experiment_path

        df = self._load_metrics(path)
        
        # 凡例を外に表示する場合は，グラフ領域を調整
        if legend_outside:
            figsize = (figsize[0] + 3, figsize[1])  # 右側に余白を追加
        
        fig = plt.figure(figsize=figsize)
        
        for i, metric in enumerate(metrics, 1):
            ax = plt.subplot(len(metrics), 1, i)
            plt.plot(df['epoch'], df[metric], color=self.colors[0], 
                    linewidth=2, label=f'{exp_name} - {metric}')
            
            self._format_axis(ax, metric)
            plt.title(f'{metric.replace("_", " ").title()}', pad=15)
            self._format_legend(ax, legend_outside)
        
        plt.suptitle(f'Experiment: {exp_name}', y=1.02)
        plt.tight_layout()
        
        self._save_or_show(save, save_name, f'{exp_name}_metrics.png')

    def plot_experiments_individually(
        self,
        experiment_paths: List[Tuple[str, str]],
        metrics: List[str] = ['train_loss', 'val_loss', 'lr'],
        figsize: Tuple[int, int] = (15, 10),
        save: bool = False
    ) -> None:
        """複数の実験結果を個別にプロットします
        
        Args:
            experiment_paths (List[Tuple[str, str]]): 実験名とmetrics.csvのパスのタプルのリスト
            metrics (List[str]): 表示するメトリクスのリスト
            figsize (Tuple[int, int]): 図のサイズ
            save (bool): 画像を保存するかどうか（デフォルト: False）
        """
        for exp_name, path in experiment_paths:
            self.plot_experiment((exp_name, path), metrics, figsize, save)

    def compare_experiments(
        self,
        experiment_paths: List[Tuple[str, str]],
        metrics: List[str] = ['train_loss', 'val_loss', 'lr'],
        figsize: Tuple[int, int] = (15, 10),
        save: bool = False,
        save_name: Optional[str] = None,
        legend_outside: bool = True
    ) -> None:
        """複数の実験結果を比較してプロットします
        
        Args:
            experiment_paths (List[Tuple[str, str]]): 実験名とmetrics.csvのパスのタプルのリスト
            metrics (List[str]): 比較するメトリクスのリスト
            figsize (Tuple[int, int]): 図のサイズ
            save (bool): 画像を保存するかどうか（デフォルト: False）
            save_name (Optional[str]): 保存するファイル名
            legend_outside (bool): 凡例をグラフの外に表示するかどうか
        """
        # 凡例を外に表示する場合は，グラフ領域を調整
        if legend_outside:
            figsize = (figsize[0] + 3, figsize[1])  # 右側に余白を追加
            
        fig = plt.figure(figsize=figsize)
        
        for i, metric in enumerate(metrics, 1):
            ax = plt.subplot(len(metrics), 1, i)
            
            for j, (exp_name, path) in enumerate(experiment_paths):
                df = self._load_metrics(path)
                plt.plot(df['epoch'], df[metric], color=self.colors[j % len(self.colors)], 
                        linewidth=2, label=exp_name)
            
            self._format_axis(ax, metric)
            plt.title(f'{metric.replace("_", " ").title()} Comparison', pad=15)
            self._format_legend(ax, legend_outside)
        
        plt.tight_layout()
        
        self._save_or_show(save, save_name, 'experiment_comparison.png')