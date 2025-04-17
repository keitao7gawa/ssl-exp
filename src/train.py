import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset import MyCIFAR10
from model_wrapper import SimCLRWrapper
from logger import SimCLRLogger
from models.simclr import SimCLR
from models.resnet import ResNetWrapper
from optimizer.lars import LARS

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します．
    
    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(description='SimCLR 事前学習')
    parser.add_argument('--config', type=str, default='configs/simclr.yaml',
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

def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """設定に基づいて最適化アルゴリズムを作成します．
    
    Args:
        model (nn.Module): 最適化するモデル
        config (dict): 最適化アルゴリズムの設定
        
    Returns:
        optim.Optimizer: 最適化アルゴリズム
    """
    optimizer_name = config.training.optimizer.name
    optimizer_params = config.training.optimizer.params
    
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == 'LARS':
        return LARS(
            model.parameters(),
            lr=optimizer_params.lr,
            weight_decay=optimizer_params.weight_decay,
            momentum=optimizer_params.momentum,
            eta=optimizer_params.eta,
            trust_coef=optimizer_params.trust_coef
        )
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

def get_scheduler(optimizer: optim.Optimizer, config: dict) -> Optional[optim.lr_scheduler._LRScheduler]:
    """設定に基づいて学習率スケジューラを作成します．
    
    Args:
        optimizer (optim.Optimizer): 最適化アルゴリズム
        config (dict): 学習率スケジューラの設定
        
    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: 学習率スケジューラ
    """
    scheduler_name = config.training.scheduler.name
    scheduler_params = config.training.scheduler.params
    
    if scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == 'None':
        return None
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_name}')

def train(
    model: SimCLRWrapper,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    logger: SimCLRLogger,
    output_dir: Path,
    start_epoch: int = 0,
    best_loss: float = float('inf')
) -> None:
    """SimCLRモデルの事前学習を行います．
    
    Args:
        model (SimCLRWrapper): 学習するモデル
        train_loader (DataLoader): 学習データローダー
        device (str): 使用デバイス
        epochs (int): エポック数
        logger (SimCLRLogger): ロガー
        output_dir (Path): 出力ディレクトリ
        start_epoch (int): 開始エポック
        best_loss (float): 最良の損失
    """
    model.to(device)
    
    for epoch in range(start_epoch, epochs):
        # 学習
        train_loss = 0.0
        train_total = 0
        
        for batch in train_loader:
            result = model.training_step(batch, device)
            train_loss += result['loss']
            train_total += 1
        
        train_loss = train_loss / train_total
        
        # ログ記録
        metrics = {
            'train_loss': train_loss,
            'lr': model.optimizer.param_groups[0]['lr'],
            'temperature': model.temperature
        }
        logger.log_metrics(epoch + 1, metrics)
        
        # モデルの保存
        checkpoint_dir = logger.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 最新のモデルを保存
        model.save_checkpoint(
            checkpoint_dir / 'latest.pth',
            epoch + 1,
            train_loss
        )
        
        # 最良のモデルを保存
        if train_loss < best_loss:
            best_loss = train_loss
            model.save_checkpoint(
                checkpoint_dir / 'best.pth',
                epoch + 1,
                train_loss
            )
            logger.log_message(f'最良のモデルを保存しました: 損失 = {train_loss:.4f}')
        
        # 学習率の更新
        model.update_scheduler()

def main():
    # 引数の解析
    args = parse_args()
    
    # 設定の読み込み
    config = load_config(args.config)
    
    # 出力ディレクトリの作成
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データセットの準備
    train_dataset = MyCIFAR10(
        train=True,
        root=config.dataset.root,
        download=config.dataset.download,
        transform_cfg=config.dataset.train_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=config.training.drop_last
    )
    
    # モデルの準備
    encoder = ResNetWrapper(
        model_name=config.model.name,
        pretrained=False,
        num_classes=config.model.hidden_dim
    )
    
    simclr_model = SimCLR(
        encoder=encoder,
        projection_dim=config.model.projection_dim,
        hidden_dim=config.model.hidden_dim
    )
    
    # SimCLRWrapperの初期化
    model_wrapper = SimCLRWrapper(
        model=simclr_model,
        temperature=config.model.temperature
    )
    
    # 最適化アルゴリズムと学習率スケジューラの準備
    optimizer = get_optimizer(model_wrapper, config)
    scheduler = get_scheduler(optimizer, config)
    
    # モデルに最適化アルゴリズムと学習率スケジューラを設定
    model_wrapper.optimizer = optimizer
    model_wrapper.scheduler = scheduler
    
    # チェックポイントの読み込み
    start_epoch = 0
    best_loss = float('inf')
    
    # ロガーの準備
    logger = SimCLRLogger(base_dir=config.training.output_dir, config_path=args.config)
    
    if args.resume:
        model_wrapper.to(config.training.device)
        checkpoint_info = model_wrapper.load_checkpoint(args.resume)
        start_epoch = checkpoint_info['epoch']
        best_loss = checkpoint_info['train_loss']
        logger.log_message(f'チェックポイントを読み込みました: {args.resume}')
        logger.log_message(f'エポック: {start_epoch}, 損失: {best_loss:.4f}')
    
    # 学習の実行
    train(
        model=model_wrapper,
        train_loader=train_loader,
        device=config.training.device,
        epochs=config.training.epochs,
        logger=logger,
        output_dir=output_dir,
        start_epoch=start_epoch,
        best_loss=best_loss
    )
    
    # ロガーを閉じる
    logger.close()

if __name__ == '__main__':
    main() 