# model
from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM,
    NUM_LAYERS,
    MAX_LEN,
    DROPOUT,
    ACTIVATION,
    PAD_TOKEN_ID,
    POSITIONAL_EMBEDDING_TYPE,
    MOTIF_VOCAB_SIZE,
    PROJECTION_DIM,
)

from embeddings import TokenPositionMotifEmbedding


class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size: int, motif_vocab_size: int = MOTIF_VOCAB_SIZE):
        super().__init__()

        self.embedding = TokenPositionMotifEmbedding(
            vocab_size=vocab_size,
            max_len=MAX_LEN,
            embedding_dim=EMBED_DIM,
            pad_token_id=PAD_TOKEN_ID,
            positional_type=POSITIONAL_EMBEDDING_TYPE,
            motif_vocab_size=motif_vocab_size,
        )
        self.embedding_dropout = nn.Dropout(p=DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation=ACTIVATION,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

    def forward(
        self,
        x: torch.Tensor,                             
        attention_mask: torch.Tensor | None = None,   
        motif_flags: torch.Tensor | None = None,     
        position_ids: torch.Tensor | None = None,    
    ) -> torch.Tensor:
        if attention_mask is None:
            src_key_padding_mask = (x == PAD_TOKEN_ID)
        else:
            src_key_padding_mask = (attention_mask == 0)
        src_key_padding_mask = src_key_padding_mask.bool() 

        embeddings = self.embedding(
            x,
            attention_mask=attention_mask,
            motif_flags=motif_flags,
            position_ids=position_ids,
        )
        embeddings = self.embedding_dropout(embeddings)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask, 
        )  
        return encoded


class DualStreamEncoder(nn.Module):
    def __init__(
        self,
        kmer_vocab_size: int,
        bpe_vocab_size: int,
        motif_vocab_size: int = MOTIF_VOCAB_SIZE,
        fusion: str = "sum",        
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.fusion = fusion.lower()
        assert self.fusion in ("sum", "concat")

        self.kmer_emb = TokenPositionMotifEmbedding(
            vocab_size=kmer_vocab_size,
            max_len=MAX_LEN,
            embedding_dim=embed_dim,
            pad_token_id=PAD_TOKEN_ID,
            positional_type=POSITIONAL_EMBEDDING_TYPE,
            motif_vocab_size=motif_vocab_size,
        )
        self.bpe_emb = TokenPositionMotifEmbedding(
            vocab_size=bpe_vocab_size,
            max_len=MAX_LEN,
            embedding_dim=embed_dim,
            pad_token_id=PAD_TOKEN_ID,
            positional_type=POSITIONAL_EMBEDDING_TYPE,
            motif_vocab_size=1, 
        )

        self.dropout = nn.Dropout(DROPOUT)

        if self.fusion == "concat":
            self.fuse_proj = nn.Linear(2 * embed_dim, embed_dim)
        else:
            self.fuse_proj = nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation=ACTIVATION,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)

    def forward(
        self,
        kmer_input_ids: torch.Tensor,
        bpe_input_ids: torch.Tensor,
        kmer_attention_mask: torch.Tensor | None = None,
        bpe_attention_mask: torch.Tensor | None = None,
        kmer_position_ids: torch.Tensor | None = None,
        bpe_position_ids: torch.Tensor | None = None,
        motif_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kmer_attention_mask is None:
            k_pad = (kmer_input_ids == PAD_TOKEN_ID)
        else:
            k_pad = (kmer_attention_mask == 0)
        if bpe_attention_mask is None:
            b_pad = (bpe_input_ids == PAD_TOKEN_ID)
        else:
            b_pad = (bpe_attention_mask == 0)

        k_emb = self.kmer_emb(
            kmer_input_ids,
            attention_mask=(~k_pad).long(),
            motif_flags=motif_flags,
            position_ids=kmer_position_ids,
        )
        b_emb = self.bpe_emb(
            bpe_input_ids,
            attention_mask=(~b_pad).long(),
            motif_flags=None,
            position_ids=bpe_position_ids,
        )

        if self.fusion == "sum":
            fused = k_emb + b_emb
        else: 
            fused = torch.cat([k_emb, b_emb], dim=-1)
            fused = self.fuse_proj(fused)

        fused = self.dropout(fused)

        encoded = self.encoder(fused, src_key_padding_mask=k_pad.bool())
        return encoded


class MLMHead(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) 


class MaskedMotifModelingHead(nn.Module):
    def __init__(self, embedding_dim: int, motif_vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, motif_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) 

class ContrastiveMotifLearningHead(nn.Module):
    def __init__(self, embedding_dim: int = EMBED_DIM, projection_dim: int = PROJECTION_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_embedding = x[:, 0, :]
        return self.proj(cls_embedding) 


class MotifAnnotatedPretrainingHead(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_embedding = x[:, 0, :]
        return self.linear(cls_embedding) 


class MotifBoundaryPredictionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) 


class MultiTaskPretrainingModel(nn.Module):
    def __init__(self, vocab_size: int, motif_vocab_size: int = MOTIF_VOCAB_SIZE, dual_stream: bool = False):
        super().__init__()
        if not dual_stream:
            self.encoder = SimpleEncoder(vocab_size, motif_vocab_size)
        else:
            raise ValueError("For dual_stream=True, instantiate DualStreamEncoder separately.")

        self.mlm_head = MLMHead(EMBED_DIM, vocab_size)
        self.masked_motif_head = MaskedMotifModelingHead(EMBED_DIM, motif_vocab_size)
        self.cml_head = ContrastiveMotifLearningHead()
        self.map_head = MotifAnnotatedPretrainingHead(EMBED_DIM, output_dim=1)
        self.mbp_head = MotifBoundaryPredictionHead(EMBED_DIM)

        self.embedding = self.encoder.embedding
        self.embedding_dropout = self.encoder.embedding_dropout 

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        motif_flags: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        task: str = "mlm",
    ):
        encoded = self.encoder(
            x,
            attention_mask=attention_mask,
            motif_flags=motif_flags,
            position_ids=position_ids,
        )

        task = task.lower()
        if task == "mlm":
            return self.mlm_head(encoded)
        if task == "masked_motif":
            return self.masked_motif_head(encoded)
        if task == "cml":
            return self.cml_head(encoded)
        if task == "map":
            return self.map_head(encoded)
        if task == "mbp":
            return self.mbp_head(encoded)
        raise ValueError(f"Unknown task: {task}")
