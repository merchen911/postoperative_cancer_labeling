from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

from . import FocalLoss





class SupervisedTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_col='검사결과결론내용', target_col='target', max_length=256): # type: ignore
        self.texts = dataframe[text_col].astype(str).tolist()

        if target_col in dataframe.columns:
            self.targets = dataframe[target_col].tolist()
        else:
            self.targets = None

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.st_token = [self.tokenizer.special_tokens_map[i] for i in ['cls_token','bos_token'] if i in self.tokenizer.special_tokens_map][0] # type: ignore
        self.ed_token = [self.tokenizer.special_tokens_map[i] for i in ['sep_token','eos_token'] if i in self.tokenizer.special_tokens_map][0] # type: ignore

    def tokenize(self, text):
        return self.tokenizer(
            self.st_token + text + self.ed_token, # type: ignore
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        item = {k: v.squeeze(0) for k, v in tokens.items()}

        if self.targets is not None:
            target = self.targets[idx]
            item['labels'] = torch.tensor(target, dtype=torch.long)
        
        return item





class SimpleClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=768, num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits




class SupervisedPLModel(pl.LightningModule):
    def __init__(self, encoder, hidden_dim=768, num_classes=3, lr=1e-4, gamma=2.0, alpha=None, criterion='focal'):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.lr = lr
        
        # criterion 설정
        if criterion == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:  # 기본값은 focal
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(
            batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            token_type_ids=batch.get('token_type_ids')
        )
        loss = self.criterion(logits, batch['labels'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            token_type_ids=batch.get('token_type_ids')
        )
        loss = self.criterion(logits, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)