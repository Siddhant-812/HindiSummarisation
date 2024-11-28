import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel

class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        # Load IndicBERT model
        self.encoder = AutoModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id
        
        # LoRA Configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Feature extraction for embeddings
            inference_mode=False,
            r=8,  # LoRA rank
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1, # Dropout probability for LoRA
            target_modules=["query", "value"]
        )
        
        # Apply LoRA to the model
        self.encoder = get_peft_model(self.encoder, peft_config)

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True):
        batch_size = text_id.size(0)

        # Document Embedding
        input_mask = text_id != self.pad_token_id
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = out[:, 0, :]  # CLS token embedding

        if require_gold:
            # Gold Summary Embedding
            input_mask = summary_id != self.pad_token_id
            out = self.encoder(summary_id, attention_mask=input_mask)[0]
            summary_emb = out[:, 0, :]
            summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # Candidate Summary Embeddings
        candidate_num = candidate_id.size(0)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)

        # Candidate Scores
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)

        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score
        return output


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    TotalLoss = 0.0

    # Candidate Loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

    # Gold Summary Loss
    if not no_gold:
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(gold_margin)
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)

    return TotalLoss
