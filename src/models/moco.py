import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .resnet import ResNetWrapper

class MoCo(nn.Module):
    """MoCo v1モデル
    
    ResNetをエンコーダーとして使用し，キューとモーメンタムエンコーダーを実装したモデル．
    
    Attributes:
        encoder_q (nn.Module): クエリエンコーダー
        encoder_k (nn.Module): キーエンコーダー
        queue (torch.Tensor): キュー
        queue_ptr (int): キューのポインタ
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "resnet18",
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        **kwargs
    ):
        """MoCoモデルの初期化
        
        Args:
            encoder (Optional[nn.Module]): カスタムエンコーダー．Noneの場合はResNetWrapperを使用
            encoder_name (str): 使用するResNetモデルの名前
            dim (int): 出力次元
            K (int): キューのサイズ
            m (float): モーメンタム係数
            T (float): 温度パラメータ
            **kwargs: エンコーダーの追加パラメータ
        """
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # クエリエンコーダーの設定
        if encoder is None:
            self.encoder_q = ResNetWrapper(
                model_name=encoder_name,
                pretrained=False,
                num_classes=dim,
                **kwargs
            )
        else:
            self.encoder_q = encoder
            
        # キーエンコーダーの設定（初期状態はクエリエンコーダーと同じ）
        self.encoder_k = ResNetWrapper(
            model_name=encoder_name,
            pretrained=False,
            num_classes=dim,
            **kwargs
        )
        
        # キーエンコーダーのパラメータをクエリエンコーダーと同じに初期化
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # キーエンコーダーは勾配を計算しない
            
        # キューの初期化
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        # キューのポインタ
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """キーエンコーダーのモーメンタム更新"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """キューを更新
        
        Args:
            keys (torch.Tensor): 新しいキー [batch_size, dim]
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # キューのサイズはバッチサイズの倍数である必要がある
        
        # キューを更新
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # ポインタを移動
        
        self.queue_ptr[0] = ptr
        
    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播
        
        Args:
            im_q (torch.Tensor): クエリ画像 [batch_size, 3, 32, 32]
            im_k (torch.Tensor): キー画像 [batch_size, 3, 32, 32]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: クエリとキーの出力
        """
        # クエリの特徴量を計算
        q = self.encoder_q(im_q)  # [batch_size, dim]
        q = nn.functional.normalize(q, dim=1)
        
        # キーの特徴量を計算（勾配を計算しない）
        with torch.no_grad():
            self._momentum_update_key_encoder()  # キーエンコーダーを更新
            
            k = self.encoder_k(im_k)  # [batch_size, dim]
            k = nn.functional.normalize(k, dim=1)
            
        # 正例の類似度を計算
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [batch_size, 1]
        
        # 負例の類似度を計算
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [batch_size, K]
        
        # ログを計算
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, K+1]
        
        # 温度パラメータでスケーリング
        logits /= self.T
        
        # ラベル（正例はインデックス0）
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # キューを更新
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """エンコーダーのみを使用して特徴量を抽出
        
        Args:
            x (torch.Tensor): 入力テンソル [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: エンコーダーの出力 [B, dim]
        """
        return self.encoder_q(x)

class MoCoV2(MoCo):
    """MoCo v2モデル
    
    MoCo v1をベースに，2層のMLPヘッドを追加したモデル．
    
    Attributes:
        encoder_q (nn.Module): クエリエンコーダー
        encoder_k (nn.Module): キーエンコーダー
        queue (torch.Tensor): キュー
        queue_ptr (int): キューのポインタ
        projection_q (nn.Sequential): クエリの射影ヘッド
        projection_k (nn.Sequential): キーの射影ヘッド
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "resnet18",
        dim: int = 128,
        mlp_dim: int = 512,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        **kwargs
    ):
        """MoCoV2モデルの初期化
        
        Args:
            encoder (Optional[nn.Module]): カスタムエンコーダー．Noneの場合はResNetWrapperを使用
            encoder_name (str): 使用するResNetモデルの名前
            dim (int): 出力次元
            mlp_dim (int): MLPの中間層の次元
            K (int): キューのサイズ
            m (float): モーメンタム係数
            T (float): 温度パラメータ
            **kwargs: エンコーダーの追加パラメータ
        """
        super().__init__(
            encoder=encoder,
            encoder_name=encoder_name,
            dim=dim,
            K=K,
            m=m,
            T=T,
            **kwargs
        )
        
        # クエリの射影ヘッド
        self.projection_q = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        # キーの射影ヘッド
        self.projection_k = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
        
        # キーの射影ヘッドのパラメータをクエリの射影ヘッドと同じに初期化
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # キーの射影ヘッドは勾配を計算しない
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """キーエンコーダーと射影ヘッドのモーメンタム更新"""
        # エンコーダーの更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        # 射影ヘッドの更新
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播
        
        Args:
            im_q (torch.Tensor): クエリ画像 [batch_size, 3, 32, 32]
            im_k (torch.Tensor): キー画像 [batch_size, 3, 32, 32]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: クエリとキーの出力
        """
        # クエリの特徴量を計算
        q = self.encoder_q(im_q)  # [batch_size, dim]
        q = self.projection_q(q)  # [batch_size, dim]
        q = nn.functional.normalize(q, dim=1)
        
        # キーの特徴量を計算（勾配を計算しない）
        with torch.no_grad():
            self._momentum_update_key_encoder()  # キーエンコーダーと射影ヘッドを更新
            
            k = self.encoder_k(im_k)  # [batch_size, dim]
            k = self.projection_k(k)  # [batch_size, dim]
            k = nn.functional.normalize(k, dim=1)
            
        # 正例の類似度を計算
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [batch_size, 1]
        
        # 負例の類似度を計算
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [batch_size, K]
        
        # ログを計算
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, K+1]
        
        # 温度パラメータでスケーリング
        logits /= self.T
        
        # ラベル（正例はインデックス0）
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # キューを更新
        self._dequeue_and_enqueue(k)
        
        return logits, labels
