import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import LayerNorm

class SwinTransformerBlock(nn.Module):
    """3D-aware Swin Transformer block for video processing"""
    def __init__(self, dim, num_heads=8, window_size=4, shift_size=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x):
        # Input shape: (B, C, H, W) or (B, C, T, H, W)
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> (b t) h w c')
        else:
            B, C, H, W = x.shape
            T = 1
            x = rearrange(x, 'b c h w -> b h w c')
            
        shortcut = x
        
        # Window partition and attention
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Window partition
        x_windows = rearrange(shifted_x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c',
                            p1=self.window_size, p2=self.window_size)
        
        # Multi-head attention
        x_windows = self.norm1(x_windows)
        qkv = self.qkv(x_windows).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * (self.dim // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        x_windows = rearrange((attn @ v), 'b h n d -> b n (h d)')
        
        x_windows = self.proj(x_windows)
        
        # Reverse window partition
        x = rearrange(x_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c',
                     h=H//self.window_size, w=W//self.window_size,
                     p1=self.window_size, p2=self.window_size)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        # Restore proper shape
        if T > 1:
            x = rearrange(x, '(b t) h w c -> b c t h w', t=T)
        else:
            x = rearrange(x, 'b h w c -> b c h w')
        
        return x

class SparseFeaturePyramid(nn.Module):
    """Memory-efficient feature pyramid with sparse connections"""
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, feature_dim//4, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2 = nn.Conv3d(feature_dim//4, feature_dim//2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv3 = nn.Conv3d(feature_dim//2, feature_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        self.lateral1 = nn.Conv3d(feature_dim//4, feature_dim, kernel_size=1)
        self.lateral2 = nn.Conv3d(feature_dim//2, feature_dim, kernel_size=1)
        
    def forward(self, x):
        # Forward path
        c1 = self.conv1(x)  # B, C/4, T, H/2, W/2
        c2 = self.conv2(c1)  # B, C/2, T, H/4, W/4
        c3 = self.conv3(c2)  # B, C, T, H/8, W/8
        
        # Lateral connections (sparse)
        p3 = c3  # B, C, T, H/8, W/8
        
        # Upsample p3 and add to c2
        p3_up = F.interpolate(p3, size=(c2.size(2), c2.size(3), c2.size(4)), 
                            mode='trilinear', align_corners=False)
        p2 = self.lateral2(c2)
        p2 = p2 + p3_up
        
        # Upsample p2 and add to c1
        p2_up = F.interpolate(p2, size=(c1.size(2), c1.size(3), c1.size(4)), 
                            mode='trilinear', align_corners=False)
        p1 = self.lateral1(c1)
        p1 = p1 + p2_up
        
        return p1, p2, p3


class PromptMemoryBank(nn.Module):
    """Prompt-enhanced memory bank for normal pattern learning"""
    def __init__(self, mem_size=256, feature_dim=256, num_prompts=4):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, feature_dim))
        self.prompts = nn.Parameter(torch.randn(num_prompts, feature_dim))
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.prompt_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = LayerNorm(feature_dim)
        
    def forward(self, query):
        B, N, C = query.shape
        
        # Enhance query with prompts
        prompts = self.prompt_proj(self.prompts)
        enhanced_query = query + prompts.mean(dim=0, keepdim=True)
        enhanced_query = self.norm(enhanced_query)
        
        # Memory attention with prompt guidance
        mem_out, attn = self.attention(enhanced_query, self.memory.unsqueeze(0).expand(B, -1, -1),
                                     self.memory.unsqueeze(0).expand(B, -1, -1))
        
        return mem_out, attn

class MemVADModel(nn.Module):
    def __init__(self, input_size=128, in_channels=3, num_frames=8, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Feature pyramid
        self.feature_pyramid = SparseFeaturePyramid(in_channels, feature_dim)
        
        # Swin blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(feature_dim, num_heads=8, window_size=4)
            for _ in range(3)
        ])
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(feature_dim)
        
        # Memory bank
        self.memory_bank = PromptMemoryBank(256, feature_dim)
        
        # Decoder with skip connection adjustments
        self.decoder = nn.ModuleList()
        self.skip_adjusts = nn.ModuleList()
        
        channels = [feature_dim, feature_dim//2, feature_dim//4]
        for i in range(3):
            # Decoder block
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose3d(channels[i], channels[min(i+1, len(channels)-1)], 
                                 kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
                nn.GroupNorm(8, channels[min(i+1, len(channels)-1)]),
                nn.GELU()
            ))
            # Skip connection channel adjustment
            if i < 2:  # Only need 2 skip connections
                self.skip_adjusts.append(nn.Conv3d(feature_dim, channels[min(i+1, len(channels)-1)], 
                                                 kernel_size=1))
        
        self.final_conv = nn.Conv3d(feature_dim//4, in_channels, kernel_size=(1,3,3), padding=(0,1,1))
        
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.transpose(1, 2)  # B, C, T, H, W
        
        # Multi-scale feature extraction
        p1, p2, p3 = self.feature_pyramid(x)
        
        # Process each scale with Swin blocks
        features = []
        for feat, swin in zip([p1, p2, p3], self.swin_blocks):
            feat = swin(feat)
            features.append(feat)
        
        # Temporal modeling
        x = rearrange(features[-1], 'b c t h w -> b t (h w) c')
        x = self.temporal_transformer(x)
        
        # Memory bank processing
        mem_out, mem_attn = self.memory_bank(x.reshape(-1, x.size(2), x.size(3)))
        x = mem_out.reshape(B, T, -1, self.feature_dim)
        
        # Reconstruction
        x = rearrange(x, 'b t (h w) c -> b c t h w', h=H//(2**3), w=W//(2**3))
        
        # Decoder with adjusted skip connections
        for i, (decoder, skip_adjust) in enumerate(zip(self.decoder, self.skip_adjusts)):
            x = decoder(x)
            if i < len(features)-1:
                skip = features[-(i+2)]
                # Adjust skip connection channels and size
                skip = skip_adjust(skip)
                skip = F.interpolate(skip, size=(x.size(2), x.size(3), x.size(4)),
                                  mode='trilinear', align_corners=False)
                x = x + skip
        
        # Final decoder block (no skip connection)
        x = self.decoder[-1](x)
        
        # Final reconstruction
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # B, T, C, H, W
        
        return x, mem_attn

class TemporalTransformer(nn.Module):
    """Temporal modeling with causal attention"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        # x shape: (B, T, N, C)
        B, T, N, C = x.shape
        x = rearrange(x, 'b t n c -> (b n) t c')
        
        # Causal self-attention
        x_norm = self.norm(x)
        attn_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn_mask = attn_mask.to(x.device)
        
        x = x + self.attention(x_norm, x_norm, x_norm, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.norm(x))
        
        return rearrange(x, '(b n) t c -> b t n c', b=B)
    
def compute_anomaly_score(x, recon, mem_attn, temporal_weights=None):
    """Advanced anomaly score computation with temporal weighting"""
    # Reconstruction error with perceptual loss
    recon_error = F.mse_loss(x, recon, reduction='none')
    recon_error = recon_error.mean(dim=[2,3,4])  # B, T
    
    # Memory-based score
    mem_scores = 1 - mem_attn.max(dim=-1)[0].mean(dim=-1)
    mem_scores = mem_scores.reshape_as(recon_error)
    
    # Temporal weighting
    if temporal_weights is None:
        temporal_weights = torch.ones_like(recon_error)
    
    # Combined weighted score
    anomaly_score = (0.7 * recon_error + 0.3 * mem_scores) * temporal_weights
    
    return anomaly_score