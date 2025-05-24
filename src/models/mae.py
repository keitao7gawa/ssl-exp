import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import numpy as np
from functools import partial
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Block

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone
    
    MAEは，画像の一部をマスクし，そのマスクされた部分を予測する自己教師あり学習モデルです．
    
    Attributes:
        patch_embed (PatchEmbed): パッチ埋め込み
        cls_token (nn.Parameter): CLSトークン
        pos_embed (nn.Parameter): 位置埋め込み
        blocks (nn.ModuleList): Transformerブロック
        norm (nn.LayerNorm): 正規化層
        decoder_embed (nn.Linear): デコーダーの埋め込み層
        mask_token (nn.Parameter): マスクトークン
        decoder_pos_embed (nn.Parameter): デコーダーの位置埋め込み
        decoder_blocks (nn.ModuleList): デコーダーのブロック
        decoder_norm (nn.LayerNorm): デコーダーの正規化層
        decoder_pred (nn.Linear): デコーダーの予測層
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False
    ):
        """MAEモデルの初期化
        
        Args:
            img_size (int): 入力画像のサイズ
            patch_size (int): パッチのサイズ
            in_chans (int): 入力チャンネル数
            embed_dim (int): 埋め込み次元
            depth (int): エンコーダーの深さ
            num_heads (int): エンコーダーのヘッド数
            decoder_embed_dim (int): デコーダーの埋め込み次元
            decoder_depth (int): デコーダーの深さ
            decoder_num_heads (int): デコーダーのヘッド数
            mlp_ratio (float): MLPの比率
            norm_layer (nn.Module): 正規化層
            use_checkpoint (bool): 勾配チェックポイントを使用するかどうか
        """
        super().__init__()
        
        # モデルのパラメータを保存
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.use_checkpoint = use_checkpoint
        
        # パッチ埋め込み
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLSトークン
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置埋め込み
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        # エンコーダーのブロック
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # エンコーダーの正規化層
        self.norm = norm_layer(embed_dim)
        
        # デコーダーの埋め込み層
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # マスクトークン
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # デコーダーの位置埋め込み
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        # デコーダーのブロック
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        # デコーダーの正規化層
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # デコーダーの予測層
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)
        
        # 初期化
        self.initialize_weights()
        
    def initialize_weights(self) -> None:
        """重みの初期化"""
        # 位置埋め込みの初期化
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # CLSトークンの初期化
        nn.init.normal_(self.cls_token, std=0.02)
        
        # マスクトークンの初期化
        nn.init.normal_(self.mask_token, std=0.02)
        
        # その他の重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module) -> None:
        """重みの初期化
        
        Args:
            m (nn.Module): 初期化するモジュール
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """画像をパッチに分割
        
        Args:
            imgs (torch.Tensor): 入力画像 [B, C, H, W]
            
        Returns:
            torch.Tensor: パッチ化された画像 [B, L, p*p*C]
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x
        
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """パッチを画像に戻す
        
        Args:
            x (torch.Tensor): パッチ化された画像 [B, L, p*p*C]
            
        Returns:
            torch.Tensor: 元の画像 [B, C, H, W]
        """
        p = self.patch_embed.patch_size[0]
        h = w = int((x.shape[1])**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
        
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ランダムマスキング
        
        Args:
            x (torch.Tensor): 入力テンソル [B, L, D]
            mask_ratio (float): マスク率
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: マスクされたテンソル，マスク，インデックス
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))  # 保持するパッチ数
        
        noise = torch.rand(N, L, device=x.device)
        
        # シャッフル
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # マスク
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # マスクの生成（1がマスク，0が保持）
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """エンコーダーの順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [B, C, H, W]
            mask_ratio (float): マスク率
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: エンコードされたテンソル，マスク，インデックス
        """
        # パッチ埋め込み
        x = self.patch_embed(x)
        
        # 位置埋め込みの追加
        x = x + self.pos_embed[:, 1:, :]
        
        # マスキング
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # CLSトークンの追加
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformerブロック
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore
        
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """デコーダーの順伝播
        
        Args:
            x (torch.Tensor): エンコードされたテンソル [B, L, D]
            ids_restore (torch.Tensor): 復元用インデックス [B, L]
            
        Returns:
            torch.Tensor: デコードされたテンソル [B, L, p*p*C]
        """
        # 埋め込み層
        x = self.decoder_embed(x)
        
        # マスクトークンの追加
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1], x_], dim=1)
        
        # 位置埋め込みの追加
        x = x + self.decoder_pos_embed
        
        # Transformerブロック
        if self.use_checkpoint:
            for blk in self.decoder_blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        
        x = self.decoder_norm(x)
        
        # 予測層
        x = self.decoder_pred(x)
        
        # CLSトークンを削除
        x = x[:, 1:, :]
        
        return x
        
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [B, C, H, W]
            mask_ratio (float): マスク率
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 予測結果，マスク
        """
        # エンコーダー
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # デコーダー
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    """MAE ViT-Baseモデル
    
    Args:
        **kwargs: 追加の引数
        
    Returns:
        MaskedAutoencoderViT: MAE ViT-Baseモデル
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    """MAE ViT-Largeモデル
    
    Args:
        **kwargs: 追加の引数
        
    Returns:
        MaskedAutoencoderViT: MAE ViT-Largeモデル
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    """MAE ViT-Hugeモデル
    
    Args:
        **kwargs: 追加の引数
        
    Returns:
        MaskedAutoencoderViT: MAE ViT-Hugeモデル
    """
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

# 推奨アーキテクチャ
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """2次元の正弦・余弦位置埋め込みを生成
    
    Args:
        embed_dim (int): 埋め込み次元
        grid_size (int): グリッドサイズ
        cls_token (bool): CLSトークンを含めるかどうか
        
    Returns:
        np.ndarray: 位置埋め込み
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """グリッドから2次元の正弦・余弦位置埋め込みを生成
    
    Args:
        embed_dim (int): 埋め込み次元
        grid (np.ndarray): グリッド
        
    Returns:
        np.ndarray: 位置埋め込み
    """
    assert embed_dim % 2 == 0
    
    # グリッドの正規化
    grid = grid.reshape([2, -1])
    grid = grid / grid.max()
    
    # 位置埋め込みの生成
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """グリッドから1次元の正弦・余弦位置埋め込みを生成
    
    Args:
        embed_dim (int): 埋め込み次元
        pos (np.ndarray): 位置
        
    Returns:
        np.ndarray: 位置埋め込み
    """
    assert embed_dim % 2 == 0
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb 