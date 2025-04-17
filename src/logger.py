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
        self.csv_file_handle.close()
        
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