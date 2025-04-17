import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from utils.losses import NTXentLoss

class BaseWrapper(nn.Module):
    """モデルをラップする基底クラス
    
    モデルの学習・評価・推論の基本的なロジックを提供します．
    
    Attributes:
        model (nn.Module): ベースとなるモデル
        criterion (nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        **kwargs: Any
    ):
        """BaseWrapperの初期化
        
        Args:
            model (nn.Module): ベースとなるモデル
            criterion (Optional[nn.Module]): 損失関数
            optimizer (Optional[torch.optim.Optimizer]): オプティマイザ
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
            **kwargs: その他のパラメータ
        """
        super().__init__()
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル
            
        Returns:
            torch.Tensor: モデルの出力
        """
        return self.model(x)
    
    def update_scheduler(self) -> None:
        """学習率スケジューラを更新します．"""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def save_checkpoint(self, path: str, epoch: int, train_loss: float) -> None:
        """チェックポイントを保存します．
        
        Args:
            path (str): 保存先のパス
            epoch (int): エポック数
            train_loss (float): 学習損失
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss
        }
        
        # ディレクトリが存在しない場合は作成
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """チェックポイントを読み込みます．
        
        Args:
            path (str): チェックポイントのパス
            
        Returns:
            Dict[str, Any]: チェックポイントの情報
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # チェックポイント復元後にoptimizerのstateを正しいデバイスに移す
        device = next(self.model.parameters()).device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss']
        }

class SimCLRWrapper(BaseWrapper):
    """SimCLR用のモデルラッパー
    
    SimCLRの訓練に特化したラッパークラス．
    NT-Xentロスを使用し，2つの拡張画像を処理します．
    
    Attributes:
        model (nn.Module): ベースとなるモデル
        criterion (NTXentLoss): NT-Xent損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
        temperature (float): NT-Xentロスの温度パラメータ
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        temperature: float = 0.07,
        **kwargs: Any
    ):
        """SimCLRWrapperの初期化
        
        Args:
            model (nn.Module): ベースとなるモデル
            optimizer (Optional[torch.optim.Optimizer]): オプティマイザ
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
            temperature (float): NT-Xentロスの温度パラメータ
            **kwargs: その他のパラメータ
        """
        super().__init__(
            model=model,
            criterion=NTXentLoss(temperature=temperature),
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )
        self.temperature = temperature
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], device: str) -> Dict[str, float]:
        """SimCLRの学習ステップ
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): 2つの拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 学習結果（損失）
        """
        self.train()
        # batchは既にタプルとして渡される
        x1, x2 = batch[0]  # バッチの最初の要素がタプル batchは(images, labels)の形式で渡される
        x1, x2 = x1.to(device), x2.to(device)
        
        self.optimizer.zero_grad()
        z1 = self(x1)
        z2 = self(x2)
        loss = self.criterion(z1, z2)
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item()
        }
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], device: str) -> Dict[str, float]:
        """SimCLRの検証ステップ
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): 2つの拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 検証結果（損失）
        """
        self.eval()
        # batchは既にタプルとして渡される
        x1, x2 = batch[0]  # バッチの最初の要素がタプル　タプル batchは(images, labels)の形式で渡される
        x1, x2 = x1.to(device), x2.to(device)
        
        with torch.no_grad():
            z1 = self(x1)
            z2 = self(x2)
            loss = self.criterion(z1, z2)
            
        return {
            'loss': loss.item()
        }
        
    def save_checkpoint(self, path: str, epoch: int, train_loss: float) -> None:
        """チェックポイントを保存します．
        
        Args:
            path (str): 保存先のパス
            epoch (int): エポック数
            train_loss (float): 学習損失
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'temperature': self.temperature
        }
        
        # ディレクトリが存在しない場合は作成
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """チェックポイントを読み込みます．
        
        Args:
            path (str): チェックポイントのパス
            
        Returns:
            Dict[str, Any]: チェックポイントの情報
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # SimCLR特有のパラメータを読み込み
        self.temperature = checkpoint.get('temperature', 0.07)
        # 温度パラメータが変更された場合は損失関数を再初期化
        if self.temperature != self.criterion.temperature:
            self.criterion = NTXentLoss(temperature=self.temperature)
            
        # チェックポイント復元後にoptimizerのstateを正しいデバイスに移す
        device = next(self.model.parameters()).device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss']
        }

class FineTuneWrapper(BaseWrapper):
    """Fine-tuning用のモデルラッパー
    
    Fine-tuningに特化したラッパークラス．
    モデルの学習率を調整し，モデルのパラメータを更新します．
    
    
    """
    def __init__(
            self,
            encoder: nn.Module,
            num_classes: int,
            freeze_encoder: bool = True,
            criterion: Optional[nn.Module] = None,
            optimizer: Optional[optim.Optimizer] = None,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            **kwargs: Any
    ):
        # 線形分類器
        classifier = nn.Linear(encoder.model.fc.in_features, num_classes)
        model = nn.Sequential(encoder, classifier)
        super().__init__(
            model=model,
            criterion=criterion or nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )

        if freeze_encoder:
            for param in self.model[0].parameters():
                param.requires_grad = False
        