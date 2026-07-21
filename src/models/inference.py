from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm


def get_model_predictions(pl_model: torch.nn.Module, data_loader: Any) -> Dict[str, np.ndarray]:
    """
    Returns logits, targets, and predicted labels for a given model and dataloader.
    """
    pl_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl_model.to(device)
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = pl_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    all_logits_np = np.concatenate(all_logits)
    all_targets_np = np.concatenate(all_targets)
    all_preds_np = np.argmax(all_logits_np, axis=1)

    return {
        'logits': all_logits_np,
        'targets': all_targets_np,
        'preds': all_preds_np,
    }
