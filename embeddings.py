# embeddings
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import POSITIONAL_EMBEDDING_TYPE, MAX_LEN, DROPOUT, PROJECTION_DIM


class TokenPositionMotifEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        motif_vocab_size: int = 2,
        motif_embed_dim: int = 16,
        max_len: int = MAX_LEN,
        pad_token_id: int = 0,
        positional_type: str = POSITIONAL_EMBEDDING_TYPE,
        truncate_long_sequences: bool = True,
        dropout: float = DROPOUT,
        normalize: bool = True,
        projection_dim: Optional[int] = PROJECTION_DIM,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_len = int(max_len)
        self.truncate_long_sequences = bool(truncate_long_sequences)
        self.positional_type = str(positional_type).lower()
        self.normalize = bool(normalize)
        self.pad_token_id = int(pad_token_id)

        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_id)

     
        if self.positional_type == "learned":
            self.position_embedding = nn.Embedding(self.max_len, embedding_dim)
            self._pos_is_buffer = False
        elif self.positional_type == "sinusoidal":
            sinus = self._build_sinusoidal_embeddings(self.max_len, embedding_dim) 
            self.register_buffer("position_embedding", sinus, persistent=True)
            self._pos_is_buffer = True
        else:
            raise ValueError(f"Unknown positional embedding type: {self.positional_type!r}")

        self.motif_embedding = nn.Embedding(motif_vocab_size, motif_embed_dim)

        self.combine = nn.Linear(embedding_dim + motif_embed_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        if projection_dim is not None and projection_dim > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, projection_dim),
            )
        else:
            self.projection_head = None

        nn.init.xavier_uniform_(self.combine.weight)
        if self.combine.bias is not None:
            nn.init.zeros_(self.combine.bias)

    def _build_sinusoidal_embeddings(self, max_len: int, embedding_dim: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) 
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim)
        )
        embeddings = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings.unsqueeze(0)  # (1, L, E)

    def _positions_from_len(self, *args, **kwargs) -> torch.LongTensor:
       
        if args:
            if len(args) != 3:
                raise TypeError("Expected (batch_size, seq_len, device) when using positional args.")
            batch_size, seq_len, device = args
        else:
            batch_size = kwargs.pop("batch_size", kwargs.pop("B", None))
            seq_len = kwargs.pop("seq_len", kwargs.pop("L", None))
            device = kwargs.get("device", None)
            if batch_size is None or seq_len is None or device is None:
                raise TypeError("Must provide batch_size (or B), seq_len (or L), and device")
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    def _gather_positions(self, position_ids: torch.LongTensor) -> torch.Tensor:
        
        position_ids = position_ids.clamp(min=0, max=self.max_len - 1)

        if self._pos_is_buffer:
            table = self.position_embedding 
            B, L = position_ids.shape
            E = table.size(-1)
            table_exp = table.expand(B, -1, -1)                         
            idx = position_ids.unsqueeze(-1).expand(B, L, E)           
            pos = torch.gather(table_exp, dim=1, index=idx)         
            return pos
        else:
            return self.position_embedding(position_ids) 

    @torch.no_grad()
    def _maybe_truncate(self, x: torch.Tensor, *tensors):
        B, L = x.shape
        if L <= self.max_len:
            return (x, *tensors)
        if not self.truncate_long_sequences:
            raise ValueError(f"Input sequence length {L} exceeds max_len {self.max_len}")
        sl = slice(0, self.max_len)
        out = [x[:, sl]]
        for t in tensors:
            if t is None:
                out.append(None)
            else:
                out.append(t[:, sl])
        return tuple(out)

    def encode(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        motif_flags: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_motif_embeds: bool = False,
    ) -> torch.Tensor:
        B, L = x.shape
        x, attention_mask, motif_flags, position_ids = self._maybe_truncate(x, attention_mask, motif_flags, position_ids)

        if attention_mask is None:
            attention_mask = (x != self.pad_token_id).long()
        if motif_flags is None:
            motif_flags = torch.zeros_like(x, dtype=torch.long)
        if position_ids is None:
            position_ids = self._positions_from_len(batch_size=x.size(0), seq_len=x.size(1), device=x.device)

        tok = self.token_embedding(x)                            
        pos = self._gather_positions(position_ids)                
        mot = self.motif_embedding(motif_flags)                   

        mask_f = attention_mask.to(dtype=tok.dtype).unsqueeze(-1)  
        pos = pos * mask_f
        mot = mot * mask_f

        token_plus_pos = tok + pos                                 
        combined = self.combine(torch.cat([token_plus_pos, mot], dim=-1))  
        combined = self.dropout(combined)
        if self.normalize:
            combined = F.normalize(combined, p=2, dim=-1)

        if return_motif_embeds:
            return combined, mot
        return combined

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        motif_flags: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_motif_embeds: bool = False,
    ):
        return self.encode(
            x,
            attention_mask=attention_mask,
            motif_flags=motif_flags,
            position_ids=position_ids,
            return_motif_embeds=return_motif_embeds,
        )

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        
        if self.projection_head is None:
            return F.normalize(embeddings, p=2, dim=-1)
        B, L, E = embeddings.shape
        flat = embeddings.reshape(B * L, E)  
        proj = self.projection_head(flat).reshape(B, L, -1)
        return F.normalize(proj, p=2, dim=-1)

    def project_cls(self, embeddings: torch.Tensor) -> torch.Tensor:
        
        cls_embedding = embeddings[:, 0, :] 
        if self.projection_head is None:
            return F.normalize(cls_embedding, p=2, dim=-1)
        return F.normalize(self.projection_head(cls_embedding), p=2, dim=-1)

    def encode_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        attention_mask1: Optional[torch.Tensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        motif_flags1: Optional[torch.Tensor] = None,
        motif_flags2: Optional[torch.Tensor] = None,
        position_ids1: Optional[torch.Tensor] = None,
        position_ids2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb1 = self.encode(
            x1, attention_mask=attention_mask1, motif_flags=motif_flags1, position_ids=position_ids1
        )
        emb2 = self.encode(
            x2, attention_mask=attention_mask2, motif_flags=motif_flags2, position_ids=position_ids2
        )
        return self.project_cls(emb1), self.project_cls(emb2)
