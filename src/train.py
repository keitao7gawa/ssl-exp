import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset import MyCIFAR10, HFD100
from model_wrapper import SimCLRWrapper, MoCoWrapper, MAEWrapper
from logger import SimCLRLogger, MoCoLogger, MAELogger
from models.simclr import SimCLR
from models.moco import MoCo, MoCoV2
from models.resnet import ResNetWrapper
from optimizer.lars import LARS
from models.mae import MaskedAutoencoderViT

class BaseTrainer:
    """訓練の基底クラス
    
    Attributes:
        config (Dict[str, Any]): 設定
        device (str): 使用デバイス
        output_dir (Path): 出力ディレクトリ
        logger: ロガー
        config_path (str): 設定ファイルのパス
    """
    
    def __init__(self, config: Dict[str, Any], config_path: str) -> None:
        """初期化
        
        Args:
            config (Dict[str, Any]): 設定
            config_path (str): 設定ファイルのパス
        """
        self.config = config
        self.config_path = config_path
        self.device = config.training.device
        
        # 出力ディレクトリの作成
        self.output_dir = Path(config.training.output_dir) / config.training.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Early Stoppingの初期化
        self._setup_early_stopping()
        
        # データセットとデータローダーの設定
        self.DATASET_DICT = {
            "CIFAR10": MyCIFAR10,
            "HFD100": HFD100
        }
        self._setup_data()
        
        # モデルの設定
        self._setup_model()
        
        # 最適化アルゴリズムと学習率スケジューラの設定
        self._setup_optimizer()
        
        # ロガーの設定
        self._setup_logger()
        
    def _setup_early_stopping(self) -> None:
        """Early Stoppingの初期化"""
        self.early_stopping_enabled = self.config.training.early_stopping.enabled
        if self.early_stopping_enabled:
            self.patience = self.config.training.early_stopping.patience
            self.min_delta = self.config.training.early_stopping.min_delta
            self.mode = self.config.training.early_stopping.mode
            self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
            self.counter = 0
            self.early_stop = False
            
    def _check_early_stopping(self, metric: float) -> bool:
        """Early Stoppingのチェック
        
        Args:
            metric (float): 現在の評価指標
            
        Returns:
            bool: 学習を停止すべきかどうか
        """
        if not self.early_stopping_enabled:
            return False
            
        if self.mode == 'min':
            if metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if metric > self.best_metric + self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            self.logger.log_message(f'Early Stopping: {self.patience}エポック間改善が見られませんでした')
            return True
            
        return False
        
    def _setup_data(self) -> None:
        """データセットとデータローダーの設定"""
        raise NotImplementedError
        
    def _setup_model(self) -> None:
        """モデルの設定"""
        raise NotImplementedError
        
    def _setup_optimizer(self) -> None:
        """最適化アルゴリズムと学習率スケジューラの設定"""
        raise NotImplementedError
        
    def _setup_logger(self) -> None:
        """ロガーの設定"""
        raise NotImplementedError
        
    def train(self) -> None:
        """訓練ループ"""
        raise NotImplementedError
        
    def _train_epoch(self, epoch: int) -> float:
        """1エポックの訓練
        
        Args:
            epoch (int): 現在のエポック
            
        Returns:
            float: 平均損失
        """
        raise NotImplementedError
        
    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """チェックポイントの保存
        
        Args:
            epoch (int): 現在のエポック
            loss (float): 損失
        """
        checkpoint_dir = self.logger.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 最新のモデルを保存
        self.model_wrapper.save_checkpoint(
            checkpoint_dir / 'latest.pth',
            epoch + 1,
            loss
        )
        
        # 最良のモデルを保存
        if loss < self.best_loss:
            self.best_loss = loss
            self.model_wrapper.save_checkpoint(
                checkpoint_dir / 'best.pth',
                epoch + 1,
                loss
            )
            self.logger.log_message(f'最良のモデルを保存しました: 損失 = {loss:.4f}')

class SimCLRTrainer(BaseTrainer):
    """SimCLRの訓練クラス"""
    
    def _setup_data(self) -> None:
        """データセットとデータローダーの設定"""
        # データセット
        dataset_params = self.config.dataset
        dataset_params = {k: v for k, v in dataset_params.items() if k != "name"}
        self.train_dataset = self.DATASET_DICT[self.config.dataset.name](
            train = True,
            **dataset_params)
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=self.config.training.drop_last
        )
        
    def _setup_model(self) -> None:
        """モデルの設定"""
        # エンコーダー
        encoder = ResNetWrapper(
            model_name=self.config.model.name,
            pretrained=self.config.model.pretrained,
            num_classes=self.config.model.hidden_dim,
            input_layer_params=self.config.model.input_layer_params
        )
        
        # SimCLRモデル
        simclr_model = SimCLR(
            encoder=encoder,
            projection_dim=self.config.model.projection_dim,
            hidden_dim=self.config.model.hidden_dim
        )
        
        # SimCLRWrapper
        self.model_wrapper = SimCLRWrapper(
            model=simclr_model,
            temperature=self.config.model.temperature
        )
        
    def _setup_optimizer(self) -> None:
        """最適化アルゴリズムと学習率スケジューラの設定"""
        # 最適化アルゴリズム
        optimizer_name = self.config.training.optimizer.name
        optimizer_params = self.config.training.optimizer.params
        
        if optimizer_name == 'LARS':
            from optimizer.lars import LARS
            optimizer = LARS(
                self.model_wrapper.parameters(),
                lr=optimizer_params.lr,
                weight_decay=optimizer_params.weight_decay,
                momentum=optimizer_params.momentum,
                eta=optimizer_params.eta,
                trust_coef=optimizer_params.trust_coef
            )
        else:
            raise ValueError(f'サポートされていない最適化アルゴリズム: {optimizer_name}')
        
        # 学習率スケジューラ
        scheduler_name = self.config.training.scheduler.name
        scheduler_params = self.config.training.scheduler.params
        
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs * self.config.training.batch_size,
                eta_min=scheduler_params.eta_min
            )
        else:
            raise ValueError(f'サポートされていないスケジューラ: {scheduler_name}')
        
        # モデルに設定
        self.model_wrapper.optimizer = optimizer
        self.model_wrapper.scheduler = scheduler
        
    def _setup_logger(self) -> None:
        """ロガーの設定"""
        self.logger = SimCLRLogger(
            base_dir=self.output_dir,
            config_path=self.config_path
        )
        
    def train(self) -> None:
        """訓練ループ"""
        self.model_wrapper.to(self.device)
        self.best_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            # 1エポックの訓練
            train_loss = self._train_epoch(epoch)
            
            # チェックポイントの保存
            self._save_checkpoint(epoch, train_loss)
            
            # Early Stoppingのチェック
            if self._check_early_stopping(train_loss):
                break
            
    def _train_epoch(self, epoch: int) -> float:
        """1エポックの訓練
        
        Args:
            epoch (int): 現在のエポック
            
        Returns:
            float: 平均損失
        """
        self.model_wrapper.train()
        train_loss = 0.0
        train_total = 0
        
        for batch in self.train_loader:
            result = self.model_wrapper.training_step(batch, self.device)
            train_loss += result['loss']
            train_total += 1
        
        train_loss = train_loss / train_total
        
        # ログ記録
        metrics = {
            'train_loss': train_loss,
            'lr': self.model_wrapper.optimizer.param_groups[0]['lr'],
            'temperature': self.model_wrapper.temperature
        }
        self.logger.log_metrics(epoch + 1, metrics)
        
        # 学習率の更新
        self.model_wrapper.update_scheduler()
        
        return train_loss

class MoCoTrainer(BaseTrainer):
    """MoCoの訓練クラス"""
    
    def _setup_data(self) -> None:
        """データセットとデータローダーの設定"""
        # データセット
        dataset_params = self.config.dataset
        dataset_params = {k: v for k, v in dataset_params.items() if k != "name"}
        self.train_dataset = self.DATASET_DICT[self.config.dataset.name](
            train = True,
            **dataset_params)
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=self.config.training.drop_last,
            persistent_workers=self.config.training.persistent_workers,
            multiprocessing_context=self.config.training.multiprocessing_context
        )
        
    def _setup_model(self) -> None:
        """モデルの設定"""
        # エンコーダー
        encoder = ResNetWrapper(
            model_name=self.config.model.name,
            pretrained=self.config.model.pretrained,
            num_classes=self.config.model.dim,
            input_layer_params=self.config.model.input_layer_params
        )
        
        # MoCoモデル
        if self.config.framework == "moco":
            moco_model = MoCo(
                encoder=encoder,
                dim=self.config.model.dim,
                K=self.config.model.K,
                m=self.config.model.m,
                T=self.config.model.T,
                input_layer_params=self.config.model.input_layer_params
            )
        elif self.config.framework == "mocov2":
            moco_model = MoCoV2(
                encoder=encoder,
                dim=self.config.model.dim,
                mlp_dim=self.config.model.mlp_dim,
                K=self.config.model.K,
                m=self.config.model.m,
                T=self.config.model.T,
                input_layer_params=self.config.model.input_layer_params
            )
        else:
            raise ValueError(f'サポートされていないフレームワーク: {self.config.framework}')
        
        # MoCoWrapper
        self.model_wrapper = MoCoWrapper(
            model=moco_model,
            temperature=self.config.model.T
        )
        
    def _setup_optimizer(self) -> None:
        """最適化アルゴリズムと学習率スケジューラの設定"""
        # 最適化アルゴリズム
        optimizer_name = self.config.training.optimizer.name
        optimizer_params = self.config.training.optimizer.params
        
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model_wrapper.parameters(),
                lr=optimizer_params.lr,
                momentum=optimizer_params.momentum,
                weight_decay=optimizer_params.weight_decay
            )
        else:
            raise ValueError(f'サポートされていない最適化アルゴリズム: {optimizer_name}')
        
        # 学習率スケジューラ
        scheduler_name = self.config.training.scheduler.name
        scheduler_params = self.config.training.scheduler.params
        
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs * self.config.training.batch_size,
                eta_min=scheduler_params.eta_min
            )
        else:
            raise ValueError(f'サポートされていないスケジューラ: {scheduler_name}')
        
        # モデルに設定
        self.model_wrapper.optimizer = optimizer
        self.model_wrapper.scheduler = scheduler
        
    def _setup_logger(self) -> None:
        """ロガーの設定"""
        self.logger = MoCoLogger(
            base_dir=self.output_dir,
            config_path=self.config_path
        )
        
    def train(self, start_epoch: int = 0) -> None:
        """訓練ループ"""
        self.model_wrapper.to(self.device)
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.training.epochs):
            print(f"エポック {epoch + 1} / {self.config.training.epochs}")
            # 1エポックの訓練
            train_loss = self._train_epoch(epoch)
            
            # チェックポイントの保存
            self._save_checkpoint(epoch, train_loss)
            
            # Early Stoppingのチェック
            if self._check_early_stopping(train_loss):
                break
            
    def _train_epoch(self, epoch: int) -> float:
        """1エポックの訓練
        
        Args:
            epoch (int): 現在のエポック
            
        Returns:
            float: 平均損失
        """
        self.model_wrapper.train()
        train_loss = 0.0
        train_total = 0
        
        for batch in self.train_loader:
            result = self.model_wrapper.training_step(batch, self.device)
            train_loss += result['loss']
            train_total += 1
        
        train_loss = train_loss / train_total
        
        # ログ記録
        metrics = {
            'train_loss': train_loss,
            'lr': self.model_wrapper.optimizer.param_groups[0]['lr'],
            'temperature': self.model_wrapper.temperature,
            'queue_size': self.model_wrapper.model.K,
            'momentum': self.model_wrapper.model.m
        }
        self.logger.log_metrics(epoch + 1, metrics)
        
        # 学習率の更新
        self.model_wrapper.update_scheduler()
        
        return train_loss

class MAETrainer(BaseTrainer):
    """MAEの訓練クラス
    
    MAEの事前学習を管理するクラス．
    マスク付き自己符号化器の訓練を実行します．
    
    Attributes:
        config (Dict[str, Any]): 設定
        device (str): 使用デバイス
        output_dir (Path): 出力ディレクトリ
        logger (MAELogger): ロガー
        config_path (str): 設定ファイルのパス
    """
    
    def _setup_data(self) -> None:
        """データセットとデータローダーの設定"""
        # データセット
        dataset_params = self.config.dataset
        dataset_params = {k: v for k, v in dataset_params.items() if k != "name"}
        self.train_dataset = self.DATASET_DICT[self.config.dataset.name](
            train = True,
            **dataset_params)
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=self.config.training.drop_last,
            persistent_workers=self.config.training.persistent_workers,
            multiprocessing_context=self.config.training.multiprocessing_context
        )
        
    def _setup_model(self) -> None:
        """モデルの設定"""
        # MAEモデル
        mae_model = MaskedAutoencoderViT(
            img_size=self.config.model.img_size,
            patch_size=self.config.model.patch_size,
            in_chans=self.config.model.in_chans,
            embed_dim=self.config.model.embed_dim,
            depth=self.config.model.depth,
            num_heads=self.config.model.num_heads,
            decoder_embed_dim=self.config.model.decoder_embed_dim,
            decoder_depth=self.config.model.decoder_depth,
            decoder_num_heads=self.config.model.decoder_num_heads,
            mlp_ratio=self.config.model.mlp_ratio,
            norm_layer=nn.LayerNorm,
        )
        
        # ロガーの設定
        self._setup_logger()
        
        # MAEWrapper
        self.model_wrapper = MAEWrapper(
            model=mae_model,
            norm_pix_loss=self.config.model.norm_pix_loss,
            mask_ratio=self.config.model.mask_ratio,
            logger=self.logger
        )
        
    def _setup_optimizer(self) -> None:
        """最適化アルゴリズムと学習率スケジューラの設定"""
        # 最適化アルゴリズム
        optimizer_name = self.config.training.optimizer.name
        optimizer_params = self.config.training.optimizer.params
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model_wrapper.parameters(),
                lr=optimizer_params.lr,
                weight_decay=optimizer_params.weight_decay,
                betas=(optimizer_params.beta1, optimizer_params.beta2)
            )
        else:
            raise ValueError(f'サポートされていない最適化アルゴリズム: {optimizer_name}')
        
        # 学習率スケジューラ
        scheduler_name = self.config.training.scheduler.name
        scheduler_params = self.config.training.scheduler.params
        
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs * self.config.training.batch_size,
                eta_min=scheduler_params.eta_min
            )
        else:
            raise ValueError(f'サポートされていないスケジューラ: {scheduler_name}')
        
        # モデルに設定
        self.model_wrapper.optimizer = optimizer
        self.model_wrapper.scheduler = scheduler
        
    def _setup_logger(self) -> None:
        """ロガーの設定"""
        self.logger = MAELogger(
            base_dir=self.output_dir,
            config_path=self.config_path
        )
        
    def train(self, start_epoch: int = 0) -> None:
        """訓練ループ"""
        torch.cuda.empty_cache()
        self.model_wrapper.to(self.device)
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.training.epochs):
            print(f"エポック {epoch + 1} / {self.config.training.epochs}")
            self.logger.set_epoch_info(epoch, self.config.training.epochs)
            # 1エポックの訓練
            train_loss = self._train_epoch(epoch)
            
            # チェックポイントの保存
            self._save_checkpoint(epoch, train_loss)
            
            # Early Stoppingのチェック
            if self._check_early_stopping(train_loss):
                break
            
    def _train_epoch(self, epoch: int) -> float:
        """1エポックの訓練
        
        Args:
            epoch (int): 現在のエポック
            
        Returns:
            float: 平均損失
        """
        self.model_wrapper.train()
        train_loss = 0.0
        train_total = 0
        
        # ロガーにバッチ情報を設定
        self.logger.set_batch_info(0, len(self.train_loader))
        
        for batch_idx, batch in enumerate(self.train_loader):
            result = self.model_wrapper.training_step(batch, self.device)
            train_loss += result['loss']
            train_total += 1
            
            # 進捗の更新（バッチインデックスを1から始める）
            
            self.logger.set_batch_info(batch_idx + 1, len(self.train_loader))
            self.logger.update_progress(result['loss'])
            
            # バッファをフラッシュ
            if hasattr(self.logger, 'csv_file_handle'):
                self.logger.csv_file_handle.flush()
        
        train_loss = train_loss / train_total
        
        # ログ記録
        metrics = {
            'train_loss': train_loss,
            'lr': self.model_wrapper.optimizer.param_groups[0]['lr'],
            'mask_ratio': self.model_wrapper.mask_ratio
        }
        self.logger.log_metrics(epoch + 1, metrics)
        
        # 学習率の更新
        self.model_wrapper.update_scheduler()
        
        return train_loss

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します．
    
    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(description='自己教師あり学習')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='設定ファイルのパス')
    parser.add_argument('--resume', type=str, default=None,
                      help='チェックポイントのパス')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """設定ファイルを読み込みます．
    
    Args:
        config_path (str): 設定ファイルのパス
        
    Returns:
        dict: 設定
    """
    return OmegaConf.load(config_path)

def main():
    # 引数の解析
    args = parse_args()
    
    # 設定の読み込み
    config = load_config(args.config)
    
    # 訓練クラスの選択
    if config.framework == 'simclr':
        trainer = SimCLRTrainer(config, args.config)
    elif config.framework in ['moco', 'mocov2']:
        trainer = MoCoTrainer(config, args.config)
    elif config.framework == 'mae':
        trainer = MAETrainer(config, args.config)
    else:
        raise ValueError(f'サポートされていないフレームワーク: {config.framework}')
    
    # チェックポイントの読み込み
    start_epoch = 0
    if args.resume:
        checkpoint_info = trainer.model_wrapper.load_checkpoint(args.resume)
        start_epoch = checkpoint_info['epoch']
        trainer.best_loss = checkpoint_info['train_loss']
        print(f'チェックポイントを読み込みました: エポック = {start_epoch}, 損失 = {trainer.best_loss:.4f}')
        
    # 訓練の実行
    trainer.train(start_epoch=start_epoch)
    
    # ロガーを閉じる
    trainer.logger.close()

if __name__ == '__main__':
    main() 