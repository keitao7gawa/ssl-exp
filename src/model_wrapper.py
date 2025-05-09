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
        self.device = next(self.model.parameters()).device
    
    def to(self, device: str) -> 'BaseWrapper':
        """モデルとそのコンポーネントを指定されたデバイスに移動します．
        
        Args:
            device (str): 移動先のデバイス
            
        Returns:
            BaseWrapper: 自身のインスタンス
        """
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        self.device = device
        return self
    
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
        # チェックポイントをCPUにロード
        checkpoint = torch.load(path, map_location="cpu")
        
        # モデルの状態をロード
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # モデルをデバイスに転送
        self.model = self.model.to(self.device)
        
        # オプティマイザの状態をロード
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # オプティマイザの状態をデバイスに転送
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss']
        }

class SimCLRWrapper(BaseWrapper):
    """SimCLR用のモデルラッパー
    
    SimCLRの訓練に特化したラッパークラス．
    NT-Xentロスを使用し，複数の拡張画像を処理します．
    
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
        
    def training_step(self, batch: Tuple[torch.Tensor, ...], device: str) -> Dict[str, float]:
        """SimCLRの学習ステップ
        
        Args:
            batch (Tuple[torch.Tensor, ...]): 拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 学習結果（損失）
        """
        self.train()
        # batchは既にタプルとして渡される
        views = batch[0]  # バッチの最初の要素がタプル batchは(images, labels)の形式で渡される
        views = [view.to(device) for view in views]
        
        self.optimizer.zero_grad()
        z = [self(view) for view in views]
        loss = self.criterion(z)  # リストとして渡す
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item()
        }
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], device: str) -> Dict[str, float]:
        """SimCLRの検証ステップ
        
        Args:
            batch (Tuple[torch.Tensor, ...]): 拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 検証結果（損失）
        """
        self.eval()
        # batchは既にタプルとして渡される
        views = batch[0]  # バッチの最初の要素がタプル batchは(images, labels)の形式で渡される
        views = [view.to(device) for view in views]
        
        with torch.no_grad():
            z = [self(view) for view in views]
            loss = self.criterion(z)  # リストとして渡す
            
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
        
        # モデルの状態をロードしてGPUに転送
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # オプティマイザの状態をロード
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # オプティマイザの状態をGPUに転送
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
                    
        # SimCLR特有のパラメータを読み込み
        self.temperature = checkpoint.get('temperature', 0.07)
        # 温度パラメータが変更された場合は損失関数を再初期化
        if self.temperature != self.criterion.temperature:
            self.criterion = NTXentLoss(temperature=self.temperature)
            
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
        
class MoCoWrapper(BaseWrapper):
    """MoCo用のモデルラッパー
    
    MoCoの訓練に特化したラッパークラス．
    キューとモーメンタムエンコーダーを管理し，InfoNCE損失を使用します．
    
    Attributes:
        model (nn.Module): ベースとなるモデル
        criterion (nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
        temperature (float): InfoNCEロスの温度パラメータ
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        temperature: float = 0.07,
        **kwargs: Any
    ):
        """MoCoWrapperの初期化
        
        Args:
            model (nn.Module): ベースとなるモデル
            optimizer (Optional[torch.optim.Optimizer]): オプティマイザ
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
            temperature (float): InfoNCEロスの温度パラメータ
            **kwargs: その他のパラメータ
        """
        super().__init__(
            model=model,
            criterion=nn.CrossEntropyLoss(),  # InfoNCE損失はCrossEntropyLossで実装
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )
        self.temperature = temperature
        
    def training_step(self, batch: Tuple[torch.Tensor, ...], device: str) -> Dict[str, float]:
        """MoCoの学習ステップ
        
        Args:
            batch (Tuple[torch.Tensor, ...]): 拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 学習結果（損失）
        """
        self.train()
        # batchは既にタプルとして渡される
        im_q, im_k = batch[0]  # バッチの最初の要素がタプル batchは(images, labels)の形式で渡される
        im_q, im_k = im_q.to(device), im_k.to(device)
        
        self.optimizer.zero_grad()
        logits, labels = self.model(im_q, im_k)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item()
        }
    
    def validation_step(self, batch: Tuple[torch.Tensor, ...], device: str) -> Dict[str, float]:
        """MoCoの検証ステップ
        
        Args:
            batch (Tuple[torch.Tensor, ...]): 拡張画像のバッチ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 検証結果（損失）
        """
        self.eval()
        # batchは既にタプルとして渡される
        im_q, im_k = batch[0]  # バッチの最初の要素がタプル batchは(images, labels)の形式で渡される
        im_q, im_k = im_q.to(device), im_k.to(device)
        
        with torch.no_grad():
            logits, labels = self.model(im_q, im_k)
            loss = self.criterion(logits, labels)
            
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
        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        
        # モデルの状態をロードしてGPUに転送
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # オプティマイザの状態をロード
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # オプティマイザの状態をGPUに転送
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        # MoCo特有のパラメータを読み込み
        self.temperature = checkpoint.get('temperature', 0.07)
            
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss']
        }

class MAEWrapper(BaseWrapper):
    """MAE用のモデルラッパー
    
    MAEの訓練に特化したラッパークラス．
    マスク付き自己符号化器の訓練を管理します．
    
    Attributes:
        model (nn.Module): ベースとなるMAEモデル
        criterion (nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
        norm_pix_loss (bool): ピクセル正規化損失を使用するかどうか
        mask_ratio (float): マスク率（デフォルトは0.75）
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        norm_pix_loss: bool = False,
        mask_ratio: float = 0.75,
        **kwargs: Any
    ):
        """MAEWrapperの初期化
        
        Args:
            model (nn.Module): ベースとなるMAEモデル
            criterion (Optional[nn.Module]): 損失関数
            optimizer (Optional[torch.optim.Optimizer]): オプティマイザ
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): 学習率スケジューラ
            norm_pix_loss (bool): ピクセル正規化損失を使用するかどうか
            mask_ratio (float): マスク率（デフォルトは0.75）
            **kwargs: その他のパラメータ
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs
        )
        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """損失関数
        
        Args:
            imgs (torch.Tensor): 入力画像 [N, C, H, W]
            pred (torch.Tensor): 予測結果 [N, L, p*p*C]
            mask (torch.Tensor): マスク [N, L], 0は保持，1は削除
            
        Returns:
            torch.Tensor: 損失
        """
        target = self.model.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], パッチごとの平均損失
        
        loss = (loss * mask).sum() / mask.sum()  # 削除されたパッチの平均損失
        return loss
        
    def training_step(self, batch: Tuple[torch.Tensor, ...], device: str) -> Dict[str, float]:
        """MAEの学習ステップ
        
        Args:
            batch (Tuple[torch.Tensor, ...]): バッチデータ
            device (str): 使用デバイス
            
        Returns:
            Dict[str, float]: 学習結果（損失）
        """
        self.train()
        imgs = batch[0].to(device)
        
        # 順伝播（マスク率を指定）
        pred, mask = self.model(imgs, self.mask_ratio)
        
        # 損失の計算
        loss = self.forward_loss(imgs, pred, mask)
        
        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
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
            'mask_ratio': self.mask_ratio
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
        
        # モデルの状態をロードしてGPUに転送
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # オプティマイザの状態をロード
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # オプティマイザの状態をGPUに転送
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # マスク率を読み込み
        self.mask_ratio = checkpoint.get('mask_ratio', 0.75)
            
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss']
        }
