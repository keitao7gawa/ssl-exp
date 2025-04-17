import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    Args:
        params: 最適化するパラメータ
        lr: 学習率
        momentum: モーメンタム係数
        weight_decay: 重み減衰係数
        eta: LARS係数
        trust_coef: 信頼係数
    """
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        eta=0.001,
        trust_coef=0.001
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta value: {eta}")
        if trust_coef < 0.0:
            raise ValueError(f"Invalid trust_coef value: {trust_coef}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            trust_coef=trust_coef
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """パラメータを更新します．
        
        Args:
            closure: 損失を計算するクロージャ
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            trust_coef = group['trust_coef']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 勾配を計算
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # パラメータと勾配のノルムを計算
                param_norm = torch.norm(p)
                grad_norm = torch.norm(d_p)

                # 信頼係数を計算
                trust_ratio = 1.0
                if param_norm > 0 and grad_norm > 0:
                    trust_ratio = trust_coef * param_norm / (grad_norm + weight_decay * param_norm + eta)
                    trust_ratio = min(trust_ratio, 1.0)

                # モーメンタムを適用
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                # パラメータを更新
                p.add_(d_p, alpha=-group['lr'] * trust_ratio)

        return loss 