import numpy as np
import torch

class RandomHSColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0, hue=0, vis_range=None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.vis_range = vis_range # HSデータ上の可視光領域

        self.transforms = [
            self.brightness_transform,
            self.contrast_transform,
            self.saturation_transform,
            self.hue_transform
        ]
        

    def brightness_transform(self, img):
        if isinstance(self.brightness, tuple):
            brightness_factor = np.random.uniform(self.brightness[0], self.brightness[1])
        else:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        black_img = torch.zeros_like(img)
        return img * brightness_factor + black_img * (1 - brightness_factor)
    
    def contrast_transform(self, img):
        if isinstance(self.contrast, tuple):
            contrast_factor = np.random.uniform(self.contrast[0], self.contrast[1])
        else:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)

        # グレースケール化 (C, H, W) -> (1, H, W)
        gray_img = img.mean(dim=0, keepdim=True)
        # 画像全体の平均輝度値（明るさの平均）を計算
        mean = gray_img.mean()
        # 平均輝度で imgと同じ shapeの画像を作成
        mean_img = torch.ones_like(img) * mean
        # コントラストを変更
        return img * contrast_factor + mean_img * (1 - contrast_factor)
    
    def saturation_transform(self, img):
        if isinstance(self.saturation, tuple):
            saturation_factor = np.random.uniform(self.saturation[0], self.saturation[1])
        else:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)

        # グレースケール化 (C, H, W) -> (1, H, W)
        gray_img = img.mean(dim=0, keepdim=True)
        # グレースケール画像のサイズを変更 (1, H, W) -> (C, H, W)
        gray_img = gray_img.repeat(img.shape[0], 1, 1)
        # 彩度を変更
        return img * saturation_factor + gray_img * (1 - saturation_factor)
    
    def hue_transform(self, img):
        if isinstance(self.hue, tuple):
            hue_factor = np.random.uniform(self.hue[0], self.hue[1])
        else:
            hue_factor = np.random.uniform(-self.hue, self.hue)
        
        C, H, W = img.shape
        hue_factor = int(hue_factor * C)

        # (C, H, W) -> Cをずらす
        return torch.roll(img, shifts=hue_factor, dims=0)
    
    def __call__(self, x):
        x = x.to(torch.float32)
        # ランダムに順序を入れ替え
        transforms = self.transforms[:]  # リストのシャローコピーを作成
        np.random.shuffle(transforms)
        
        if self.vis_range is not None:
            # 可視光領域のみを抽出
            x = x[self.vis_range[0]:self.vis_range[1], :, :]
            # 非可視光領域を抽出 (後で元に戻す)
            x_non_vis_before = x[:self.vis_range[0], :, :]
            x_non_vis_after = x[self.vis_range[1]:, :, :]

        # 順番に変換を適用
        for transform in transforms:
            x = transform(x)
        if self.vis_range is not None:
            # 非可視光領域を元に戻す
            x = torch.cat([x_non_vis_before, x, x_non_vis_after], dim=0)
            
        return x
            
class RandomHSGrayscale:
    def __init__(self):
        pass

    def __call__(self, x):
        # グレースケール化 (C, H, W) -> (1, H, W)
        gray_img = x.mean(dim=0, keepdim=True)
        # グレースケール画像のサイズを変更 (1, H, W) -> (C, H, W)
        gray_img = gray_img.repeat(x.shape[0], 1, 1)

        return gray_img