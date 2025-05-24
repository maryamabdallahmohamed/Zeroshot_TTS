import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim0=1, dim1=2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SemanticExtractor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=384, codebook_size=8192, codebook_dim=8):
        super().__init__()
        # Only keep the encoder and quantizer parts
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            Transpose(),                             # (B, 384, L) → (B, L, 384)
            nn.LayerNorm(hidden_dim),               # now normalizes the last dim
            Transpose(),                             # (B, L, 384) → (B, 384, L)
            *[ConvNextBlock(hidden_dim) for _ in range(6)],
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1)
        )

        self.quantizer = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=codebook_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # To (batch_size, input_dim, sequence_length)
        z = self.encoder(x)  # Get encoded features
        z = z.transpose(1, 2)  # To (batch_size, sequence_length, codebook_dim)
        _, _, indices = self.quantizer(z)  # Get semantic tokens
        return indices

class ConvNextBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)  # Depthwise
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, dim, seq_len)
        residual = x
        x = self.conv(x)
        # Transpose for LayerNorm
        x = x.transpose(1, 2)  # (batch, seq_len, dim)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Transpose back
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        return x + residual

# Vector Quantization Layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=8, commitment_cost=0.25):
        
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_count', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', self.embeddings.clone())

    def forward(self, x):

        flat_x = x.reshape(-1, self.embedding_dim)
        distances = torch.cdist(flat_x, self.embeddings)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings[encoding_indices].reshape(x.shape)
        codebook_loss = F.mse_loss(quantized.detach(), x)
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, x.detach())
        loss = codebook_loss + commitment_loss
        quantized = x + (quantized - x).detach()
        
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
                self.ema_count = 0.999 * self.ema_count + 0.001 * torch.sum(one_hot, dim=0)
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + 1e-8) / (n + self.num_embeddings * 1e-8) * n
                dw = torch.matmul(one_hot.transpose(0, 1), flat_x)
                self.ema_weight = 0.999 * self.ema_weight + 0.001 * dw
                self.embeddings.data = (self.ema_weight / (self.ema_count.unsqueeze(-1) + 1e-8))
        
        return quantized, loss, encoding_indices

def load_semantic_extractor(model_path='semantic_codec_final_2.pth', device='mps'):
    model = SemanticExtractor(input_dim=512, hidden_dim=384, codebook_size=8192, codebook_dim=8)
    state_dict = torch.load(model_path, map_location=device)

    # Load encoder weights
    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state, strict=False)  # <=== set strict=False

    # Load quantizer weights
    quantizer_state = {k.replace("quantizer.", ""): v for k, v in state_dict.items() if k.startswith("quantizer.")}
    model.quantizer.load_state_dict(quantizer_state, strict=False)  

    model.to(device)
    model.eval()
    return model


def extract_semantics(audio_features, model):
    """
    Extract semantic tokens from audio features
    Args:
        audio_features: Tensor of shape (batch_size, sequence_length, 512)
        model: Loaded SemanticExtractor model
    Returns:
        semantic_tokens: Tensor of shape (batch_size, sequence_length)
    """
    with torch.no_grad():
        semantic_tokens = model(audio_features)
    return semantic_tokens



