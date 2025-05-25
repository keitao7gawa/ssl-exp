import os
import csv
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class BaseLogger:
    """ロガーの基底クラス
    
    Attributes:
        log_dir (Path): ログを保存するディレクトリ
        csv_file (Path): CSVログファイルのパス
        csv_writer (csv.writer): CSVライター
        csv_file_handle: CSVファイルハンドル
    """
    
    def __init__(self, base_dir: str = "runs", config_path: Optional[str] = None) -> None:
        """初期化
        
        Args:
            base_dir (str): ログのベースディレクトリ
            config_path (Optional[str]): 設定ファイルのパス
        """
        # 実験ごとのディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(base_dir) / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定ファイルをコピー
        if config_path is not None:
            self._copy_config(config_path)
        
        # CSVファイルの設定
        self.csv_file = self.log_dir / "metrics.csv"
        self.csv_file_handle = open(self.csv_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file_handle)
        
    def _copy_config(self, config_path: str) -> None:
        """設定ファイルをログディレクトリにコピー
        
        Args:
            config_path (str): 設定ファイルのパス
            
        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            PermissionError: 設定ファイルへのアクセス権限がない場合
        """
        try:
            config_path = Path(config_path)
            
            # 設定ファイルの存在確認
            if not config_path.exists():
                raise FileNotFoundError(f"設定ファイル {config_path} が見つかりません")
            
            # 設定ファイルの読み取り権限確認
            if not os.access(config_path, os.R_OK):
                raise PermissionError(f"設定ファイル {config_path} への読み取り権限がありません")
            
            # コピー先のファイル名を元のファイル名と同じにする
            target_path = self.log_dir / config_path.name
            
            # コピー実行
            shutil.copy2(config_path, target_path)
            print(f"設定ファイルをコピーしました: {target_path}")
            
        except FileNotFoundError as e:
            print(f"エラー: {e}")
            raise
        except PermissionError as e:
            print(f"エラー: {e}")
            raise
        except Exception as e:
            print(f"設定ファイルのコピー中にエラーが発生しました: {e}")
            raise
            
    def log_message(self, message: str) -> None:
        """メッセージをログに記録
        
        Args:
            message (str): ログメッセージ
        """
        print(message)
        
    def close(self) -> None:
        """ロガーを閉じる"""
        try:
            if hasattr(self, "csv_file_handle"):
                self.csv_file_handle.flush()
                self.csv_file_handle.close()
        except Exception as e:
            print(f"ロガーの閉じる中でエラーが発生しました: {e}")
            raise
        
    def __del__(self) -> None:
        """デストラクタ"""
        if hasattr(self, "csv_file_handle"):
            self.csv_file_handle.close()
            
class SimCLRLogger(BaseLogger):
    """SimCLR用のロガー
    
    Attributes:
        log_dir (Path): ログを保存するディレクトリ
        csv_file (Path): CSVログファイルのパス
        csv_writer (csv.writer): CSVライター
        csv_file_handle: CSVファイルハンドル
    """
    
    def __init__(self, base_dir: str = "runs", config_path: Optional[str] = None) -> None:
        """初期化
        
        Args:
            base_dir (str): ログのベースディレクトリ
            config_path (Optional[str]): 設定ファイルのパス
        """
        super().__init__(base_dir, config_path)
        
        # ヘッダーを書き込み
        self.csv_writer.writerow([
            "epoch", 
            "train_loss", 
            "val_loss", 
            "lr",  # 学習率
            "temperature"  # 温度パラメータ
        ])
        
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """メトリクスをログに記録
        
        Args:
            epoch (int): エポック数
            metrics (Dict[str, float]): メトリクスの辞書
        """
        # CSVに記録
        self.csv_writer.writerow([
            epoch,
            metrics.get("train_loss", 0.0),
            metrics.get("val_loss", 0.0),
            metrics.get("lr", 0.0),  # 学習率
            metrics.get("temperature", 0.0)  # 温度パラメータ
        ])
        
        # コンソールに出力
        print(f"Epoch {epoch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

class MoCoLogger(BaseLogger):
    """MoCo用のロガー
    
    Attributes:
        log_dir (Path): ログを保存するディレクトリ
        csv_file (Path): CSVログファイルのパス
        csv_writer (csv.writer): CSVライター
        csv_file_handle: CSVファイルハンドル
    """
    
    def __init__(self, base_dir: str = "runs", config_path: Optional[str] = None) -> None:
        """初期化
        
        Args:
            base_dir (str): ログのベースディレクトリ
            config_path (Optional[str]): 設定ファイルのパス
        """
        super().__init__(base_dir, config_path)
        
        # ヘッダーを書き込み
        self.csv_writer.writerow([
            "epoch", 
            "train_loss", 
            "lr",  # 学習率
            "temperature",  # 温度パラメータ
            "queue_size",  # キューのサイズ
            "momentum"  # モーメンタム係数
        ])
        
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """メトリクスをログに記録
        
        Args:
            epoch (int): エポック数
            metrics (Dict[str, float]): メトリクスの辞書
        """
        # CSVに記録
        self.csv_writer.writerow([
            epoch,
            metrics.get("train_loss", 0.0),
            metrics.get("lr", 0.0),  # 学習率
            metrics.get("temperature", 0.0),  # 温度パラメータ
            metrics.get("queue_size", 0.0),  # キューのサイズ
            metrics.get("momentum", 0.0)  # モーメンタム係数
        ])
        
        # コンソールに出力
        print(f"Epoch {epoch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

class MAELogger(BaseLogger):
    """MAE用のロガー
    
    MAEの訓練に特化したロガークラス．
    進捗表示機能を備えています．
    
    Attributes:
        log_dir (Path): ログを保存するディレクトリ
        csv_file (Path): CSVログファイルのパス
        csv_writer (csv.writer): CSVライター
        csv_file_handle: CSVファイルハンドル
        current_epoch (int): 現在のエポック
        total_epochs (int): 総エポック数
        current_batch (int): 現在のバッチ
        total_batches (int): 総バッチ数
        current_loss (float): 現在の損失
    """
    
    def __init__(self, base_dir: str = "runs", config_path: Optional[str] = None) -> None:
        """初期化
        
        Args:
            base_dir (str): ログのベースディレクトリ
            config_path (Optional[str]): 設定ファイルのパス
        """
        super().__init__(base_dir, config_path)
        
        # ヘッダーを書き込み
        self.csv_writer.writerow([
            "epoch", 
            "train_loss", 
            "val_loss", 
            "lr",  # 学習率
            "mask_ratio"  # マスク率
        ])
        
        # 進捗表示用の変数
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_loss = 0.0
        
    def set_epoch_info(self, current_epoch: int, total_epochs: int) -> None:
        """エポック情報を設定
        
        Args:
            current_epoch (int): 現在のエポック
            total_epochs (int): 総エポック数
        """
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
    def set_batch_info(self, current_batch: int, total_batches: int) -> None:
        """バッチ情報を設定
        
        Args:
            current_batch (int): 現在のバッチ
            total_batches (int): 総バッチ数
        """
        self.current_batch = current_batch
        self.total_batches = total_batches
        
    def update_progress(self, loss: float) -> None:
        """進捗を更新
        
        Args:
            loss (float): 現在の損失
        """
        self.current_loss = loss
        self._print_progress()
        
    def _print_progress(self) -> None:
        """進捗を表示"""
        # 進捗の計算
        progress = self.current_batch / self.total_batches
        
        # 進捗情報の表示（シンプルな形式）
        progress_str = (f'E{self.current_epoch+1}/{self.total_epochs} '
                       f'B{self.current_batch}/{self.total_batches} '
                       f'({progress*100:.1f}%) '
                       f'L:{self.current_loss:.4f}')
        
        # 進捗情報を出力
        print(f'\r{progress_str}', end='', flush=True)
        
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """メトリクスをログに記録
        
        Args:
            epoch (int): エポック数
            metrics (Dict[str, float]): メトリクスの辞書
        """
        # 改行を入れて新しい行に表示
        print()
        
        # CSVに記録
        self.csv_writer.writerow([
            epoch,
            metrics.get("train_loss", 0.0),
            metrics.get("val_loss", 0.0),
            metrics.get("lr", 0.0),  # 学習率
            metrics.get("mask_ratio", 0.75)  # マスク率
        ])
        
        # コンソールに出力
        print(f"Epoch {epoch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

def setup_logger(logger_name: str, base_dir: str = "runs", config_path: Optional[str] = None) -> BaseLogger:
    """ロガーを設定します．
    
    Args:
        logger_name (str): ロガーの名前
        base_dir (str): ログのベースディレクトリ
        config_path (Optional[str]): 設定ファイルのパス
        
    Returns:
        BaseLogger: 設定されたロガー
    """
    if logger_name == "simclr":
        return SimCLRLogger(base_dir, config_path)
    elif logger_name == "moco":
        return MoCoLogger(base_dir, config_path)
    elif logger_name == "mae":
        return MAELogger(base_dir, config_path)
    else:
        raise ValueError(f"サポートされていないロガー: {logger_name}") 