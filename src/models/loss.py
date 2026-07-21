from torch import nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.to(targets.device).gather(0, targets)
            ce_loss = at * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class F1MacroCallback(pl.Callback):
    def __init__(self, val_loader):
        super().__init__()
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        val_preds = []
        val_targets = []
        pl_module.eval()
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(pl_module.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(pl_module.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(pl_module.device)
            labels = batch['labels'].to(pl_module.device)
            with torch.no_grad():
                logits = pl_module(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
        f1_macro = f1_score(val_targets, val_preds, average='macro')
        trainer.logger.log_metrics({'val_f1_macro': f1_macro}, step=trainer.global_step)
        pl_module.log('val_f1_macro', f1_macro, prog_bar=True)


# # F1 metric logging (Lightning callback)
# class F1MacroCallback(pl.Callback):
#     def on_validation_epoch_end(self, trainer, pl_module):
#         val_preds = []
#         val_targets = []
#         pl_module.eval()
#         for batch in val_loader:
#             input_ids = batch['input_ids'].to(pl_module.device)
#             attention_mask = batch.get('attention_mask', None)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(pl_module.device)
#             token_type_ids = batch.get('token_type_ids', None)
#             if token_type_ids is not None:
#                 token_type_ids = token_type_ids.to(pl_module.device)
#             labels = batch['labels'].to(pl_module.device)
#             with torch.no_grad():
#                 logits = pl_module(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#                 preds = torch.argmax(logits, dim=1)
#             val_preds.extend(preds.cpu().numpy())
#             val_targets.extend(labels.cpu().numpy())
#         f1_macro = f1_score(val_targets, val_preds, average='macro')
#         trainer.logger.log_metrics({'val_f1_macro': f1_macro}, step=trainer.global_step)
#         pl_module.log('val_f1_macro', f1_macro, prog_bar=True)
