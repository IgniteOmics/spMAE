import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossModalCLIPLoss(nn.Module):
    def __init__(self, initial_temperature=0.07):
        """
        跨模态 CLIP 对比损失函数
        initial_temperature: 初始温度系数，0.07 是 OpenAI 论文里的默认最佳起点
        """
        super(CrossModalCLIPLoss, self).__init__()
        # 把温度系数变成可学习的参数 (Learnable Parameter)
        # 模型会在训练中自动找到最适合当前数据的拉扯力度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temperature))
        # self.logit_scale = 0.1

    def forward(self, z_rna, z_atac):
        """
        输入:
        z_rna: 干净未打乱的 RNA 潜变量 [Batch, Hidden_Dim]
        z_atac: 干净未打乱的 ATAC 潜变量 [Batch, Hidden_Dim]
        """
        # 1. 强制 L2 归一化 (把特征投影到单位超球面上，只计算方向夹角)
        z_rna = F.normalize(z_rna, dim=-1)
        z_atac = F.normalize(z_atac, dim=-1)

        # 2. 获取当前动态学习到的温度缩放因子，并限制上限防止梯度爆炸
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100.0)

        # 3. 计算相似度矩阵 (Logits)
        # 结果是一个 [Batch, Batch] 的矩阵
        logits = logit_scale * torch.matmul(z_rna, z_atac.T)

        # 4. 生成对角线标签 (Batch 内只有同一个细胞的 RNA 和 ATAC 是正样本)
        batch_size = z_rna.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=z_rna.device)

        # 5. 计算双向对称交叉熵损失
        # RNA 找自己的 ATAC
        loss_r2a = F.cross_entropy(logits, labels)
        # ATAC 找自己的 RNA
        loss_a2r = F.cross_entropy(logits.T, labels)

        # 6. 取平均值
        loss = (loss_r2a + loss_a2r) / 2.0
        
        return loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
        """
        gamma: 难度聚焦系数，通常设为 2.0
        alpha: 类别平衡系数，因为你是 40% mask，0.5 就很合适（或者设为 0.4 给正样本稍微提提权）
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits 是你 Mask Predictor 的输出 (没有经过 sigmoid 的原始值)
        # targets 是 0 和 1 的真实 mask
        
        # 1. 计算基础的 BCE Loss (不求均值)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 获取预测概率 p
        p = torch.sigmoid(logits)
        
        # 3. 计算 p_t (如果是正样本就是 p，如果是负样本就是 1-p)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 4. 计算 Focal Loss 权重
        # 答得越好 (p_t 越接近1)，权重越小
        focal_weight = (1 - p_t) ** self.gamma
        
        # 5. 加入 alpha 权重 (可选)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight
        
        # 6. 最终 Loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss