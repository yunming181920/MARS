from __future__ import print_function

import torch
import torch.nn as nn

import logging

logger = logging.getLogger("logger")


def check_nan(variable, var_name):
    """Utility function to check if a variable contains NaN and log its stats."""
    if torch.isnan(variable).any():
        logger.error(f"{var_name} contains NaN values!")
        print(f"{var_name} contains NaN values!")
    else:
        logger.info(f"{var_name} is valid.")
        print(f"{var_name} is valid.")
    print(
        f"{var_name} stats - min: {variable.min().item()}, max: {variable.max().item()}, mean: {variable.mean().item()}")


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning:
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, scale_weight=1, fac_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda:1')
                  if features.is_cuda
                  else torch.device('cpu'))

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

            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)

            for ind, label in enumerate(labels.view(-1)):
                if label == fac_label:
                    mask_scale[ind, :] = mask[ind, :] * scale_weight

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # check_nan(logits, "logits")

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        #check_nan(exp_logits, "exp_logits")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #check_nan(log_prob, "log_prob")
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)

        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        #check_nan(mean_log_prob_pos, "mean_log_prob_pos")
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        
        return loss