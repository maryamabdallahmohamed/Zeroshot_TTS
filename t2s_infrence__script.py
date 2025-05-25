import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from Vocab import Vocab
import pandas as pd
import librosa
@dataclass
class MaskGCTConfig:
    vocab_size_text: int = 40000  
    vocab_size_semantic: int = 1024 
    max_seq_len: int = 2048
    n_layers: int = 12
    n_heads: int = 4
    d_model: int = 512
    d_ff: int = 1408  
    dropout: float = 0.1
    eps: float = 1e-5
    theta: float = 10000.0  
    max_time_steps: int = 1000  
    max_position_embeddings=1152

class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm that accepts time step as condition"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)  
        )
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        time_out = self.time_mlp(time_emb)  
        scale, shift = time_out.chunk(2, dim=-1) 
        

        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        scale = scale.unsqueeze(1)  
        shift = shift.unsqueeze(1) 
        
        return norm * self.weight * (1 + scale) + shift

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit with GELU activation"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by n_heads"
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.rotary_emb = RotaryPositionalEmbedding(d_model=self.head_dim, max_seq_len=2048)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k, seq_len)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v) 
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        out = self.out_proj(attn_output)
        
        return out
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, d_model: int,max_seq_len=2048, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot


class TransformerBlock(nn.Module):
    """Transformer block with bidirectional attention and GLU"""
    def __init__(self, config: MaskGCTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = GatedLinearUnit(config.d_model, config.d_ff)
        self.norm1 = AdaptiveRMSNorm(config.d_model, config.eps)
        self.norm2 = AdaptiveRMSNorm(config.d_model, config.eps)
        
    def forward(self, x, time_emb, attention_mask=None):
        # Pre-norm attention
        normed_x = self.norm1(x, time_emb)
        attn_out = self.attention(normed_x, attention_mask)
        x = x + attn_out
        
        # Pre-norm feed-forward
        normed_x = self.norm2(x, time_emb)
        ff_out = self.feed_forward(normed_x)
        x = x + ff_out
        
        return x
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion steps"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, time_steps):
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time_steps.device) * -emb)
        emb = time_steps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
def create_model_config():
    """Create configuration for Arabic TTS"""
    return MaskGCTConfig(
        vocab_size_text=100, 
        vocab_size_semantic=7500,  
        max_seq_len=1024,
        n_layers=6,
        n_heads=2,
        d_model=512,
        d_ff=1408,
        dropout=0.1,
        eps=1e-5,
        theta=10000.0,
        max_time_steps=1000
)
    


class MaskGCT_T2S(nn.Module):
    """Text-to-Semantic MaskGCT Model"""
    def __init__(self, config: MaskGCTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.text_embedding = nn.Embedding(config.vocab_size_text, config.d_model)
        self.semantic_embedding = nn.Embedding(config.vocab_size_semantic, config.d_model)
        
        # Time embedding for diffusion
        self.time_embedding = TimeEmbedding(config.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output head
        self.output_norm = AdaptiveRMSNorm(config.d_model, config.eps)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size_semantic)
        
        # Special tokens
        self.mask_token_id = config.vocab_size_semantic - 1
        self.pad_token_id = 0
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, text_len, prompt_len, target_len):
        """Create attention mask for [text, prompt, target] sequence"""
        total_len = text_len + prompt_len + target_len
        mask = torch.ones(total_len, total_len)
        
        # Text can attend to itself
        mask[:text_len, :text_len] = 1
        
        # Prompt can attend to text and itself
        mask[text_len:text_len+prompt_len, :text_len+prompt_len] = 1
        
        # Target can attend to text, prompt, and itself (bidirectional)
        mask[text_len+prompt_len:, :] = 1
        
        return mask.unsqueeze(0)  # Add batch dimension
    
    def forward(self, 
                text_tokens: torch.Tensor,
                semantic_prompt: torch.Tensor,
                semantic_target: torch.Tensor,
                time_steps: torch.Tensor,
                mask_ratio: float = 0.15):
        """
        Forward pass for training
        
        Args:
            text_tokens: Text token sequence (batch, text_len)
            semantic_prompt: Prompt semantic tokens (batch, prompt_len)
            semantic_target: Target semantic tokens (batch, target_len)
            time_steps: Diffusion time steps (batch,)
            mask_ratio: Ratio of tokens to mask
        """
        batch_size = text_tokens.shape[0]
        text_len = text_tokens.shape[1]
        prompt_len = semantic_prompt.shape[1]
        target_len = semantic_target.shape[1]
        
        # Create masked target
        masked_target = semantic_target.clone()
        mask = torch.rand(batch_size, target_len) < mask_ratio
        masked_target[mask] = self.mask_token_id
        
        # Embed tokens
        text_emb = self.text_embedding(text_tokens)
        prompt_emb = self.semantic_embedding(semantic_prompt)
        target_emb = self.semantic_embedding(masked_target)
        
        # Concatenate sequences: [text, prompt, target]
        x = torch.cat([text_emb, prompt_emb, target_emb], dim=1)
        
        # Time embedding
        time_emb = self.time_embedding(time_steps)
        time_emb = self.time_mlp(time_emb)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(text_len, prompt_len, target_len)
        attention_mask = attention_mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, time_emb, attention_mask)
        
        # Apply output normalization and projection
        x = self.output_norm(x, time_emb)
        logits = self.output_proj(x)
        
        # Return only target logits
        target_logits = logits[:, text_len + prompt_len:, :]
        
        return target_logits
    
    def generate(self,
                 text_tokens: torch.Tensor,
                 semantic_prompt: torch.Tensor,
                 target_length: int,
                 num_steps: int = 20,
                 temperature: float = 1.0,
                 top_k: int = None,
                 top_p: float = None):
        """
        Generate semantic tokens given text and prompt
        
        Args:
            text_tokens: Text token sequence (batch, text_len)
            semantic_prompt: Prompt semantic tokens (batch, prompt_len)
            target_length: Length of target sequence to generate
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Initialize with all mask tokens
        semantic_target = torch.full(
            (batch_size, target_length), 
            self.mask_token_id, 
            device=device, 
            dtype=torch.long
        )
        
        # Iterative denoising
        for step in range(num_steps):
            # Current time step
            t = torch.full((batch_size,), step / num_steps * self.config.max_time_steps, device=device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.forward_inference(text_tokens, semantic_prompt, semantic_target, t)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k and top-p filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample tokens
            probs = F.softmax(logits, dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(batch_size, target_length)
            
            # Update masked positions
            mask_positions = (semantic_target == self.mask_token_id)
            
            # Schedule: unmask some tokens each step
            num_to_unmask = max(1, int(mask_positions.sum().item() * (1 - (step + 1) / num_steps)))
            
            # Choose positions to unmask based on confidence
            confidence = probs.max(dim=-1)[0]
            confidence[~mask_positions] = -1  # Don't consider already unmasked positions
            
            # Get top confident positions to unmask
            _, top_indices = torch.topk(confidence.view(batch_size, -1), num_to_unmask, dim=-1)
            
            # Update tokens
            for b in range(batch_size):
                for idx in top_indices[b]:
                    if mask_positions[b, idx]:
                        semantic_target[b, idx] = new_tokens[b, idx]
        
        return semantic_target
    
    def forward_inference(self, text_tokens, semantic_prompt, semantic_target, time_steps):
        """Forward pass for inference (no masking)"""
        batch_size = text_tokens.shape[0]
        text_len = text_tokens.shape[1]
        prompt_len = semantic_prompt.shape[1]
        target_len = semantic_target.shape[1]
        
        # Embed tokens
        text_emb = self.text_embedding(text_tokens)
        prompt_emb = self.semantic_embedding(semantic_prompt)
        target_emb = self.semantic_embedding(semantic_target)
        
        # Concatenate sequences
        x = torch.cat([text_emb, prompt_emb, target_emb], dim=1)
        
        # Time embedding
        time_emb = self.time_embedding(time_steps)
        time_emb = self.time_mlp(time_emb)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(text_len, prompt_len, target_len)
        attention_mask = attention_mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, time_emb, attention_mask)
        
        # Apply output normalization and projection
        x = self.output_norm(x, time_emb)
        logits = self.output_proj(x)
        
        # Return only target logits
        return logits[:, text_len + prompt_len:, :]


