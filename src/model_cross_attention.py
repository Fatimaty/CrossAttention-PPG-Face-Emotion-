import torch
import torch.nn as nn

# === Helper ===
def chw(x): return x  # placeholder; actual CHW conversion happens in dataset

# === Convolution Blocks ===
class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.pool(self.act(self.bn(self.conv(x))))
        return x


# === CNN Branch for each modality ===
class CNNBranch(nn.Module):
    def __init__(self, in_ch, base_ch=32, embed_dim=128):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4
        self.b1 = ConvBlock2d(in_ch, c1)
        self.b2 = ConvBlock2d(c1, c2)
        self.b3 = ConvBlock2d(c2, c3)
        self.proj = nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        return self.proj(x)


# === Cross-Attention Module ===
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln  = nn.LayerNorm(embed_dim)

    def forward(self, q, kv):
        out,_ = self.mha(q, kv, kv)
        return self.ln(q + out)


# === Full Multimodal Cross-Attention Model ===
class DualBranchCNN_XAttn_Classifier(nn.Module):
    def __init__(self, cwt_in_ch=1, face_in_ch=3, use_face=True,
                 base_channels=32, embed_dim=128, num_heads=4, dropout=0.2,
                 num_classes=4, pooling='mean'):
        super().__init__()
        self.use_face = use_face
        self.pooling = pooling

        # CWT branch
        self.cwt_branch = CNNBranch(cwt_in_ch, base_ch=base_channels, embed_dim=embed_dim)

        # Facial image branch
        if use_face:
            self.face_branch = CNNBranch(face_in_ch, base_ch=base_channels, embed_dim=embed_dim)

        # Optional CLS token
        if pooling == 'cls':
            self.cls_cwt  = nn.Parameter(torch.zeros(1,1,embed_dim))
            if use_face:
                self.cls_face = nn.Parameter(torch.zeros(1,1,embed_dim))

        # Cross attention both ways
        self.xattn_cwt_to_face = CrossAttention(embed_dim, num_heads, dropout)
        if use_face:
            self.xattn_face_to_cwt = CrossAttention(embed_dim, num_heads, dropout)

        fusion_dim = embed_dim * (2 if use_face else 1)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim,256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256,64), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(64,num_classes)
        )

    @staticmethod
    def _spatial_to_tokens(x, cls_token=None):
        B,D,H,W = x.shape
        x = x.view(B,D,H*W).transpose(1,2)
        if cls_token is not None:
            cls = cls_token.expand(B,-1,-1)
            x = torch.cat([cls,x], dim=1)
        return x

    @staticmethod
    def _pool(tokens, mode='mean'):
        return tokens.mean(dim=1) if mode=='mean' else tokens[:,0,:]

    def forward(self, x_cwt, x_face=None):
        f_cwt = self.cwt_branch(x_cwt)
        t_cwt = self._spatial_to_tokens(f_cwt, getattr(self,'cls_cwt',None) if self.pooling=='cls' else None)

        if self.use_face:
            f_face = self.face_branch(x_face)
            t_face = self._spatial_to_tokens(f_face, getattr(self,'cls_face',None) if self.pooling=='cls' else None)

            t_cwt_attn  = self.xattn_cwt_to_face(t_cwt, t_face)
            t_face_attn = self.xattn_face_to_cwt(t_face, t_cwt)

            v_cwt  = self._pool(t_cwt_attn,  self.pooling)
            v_face = self._pool(t_face_attn, self.pooling)

            fused = torch.cat([v_cwt, v_face], dim=-1)
            return self.head(fused)

        else:
            t_cwt_attn = self.xattn_cwt_to_face(t_cwt, t_cwt)
            v_cwt = self._pool(t_cwt_attn, self.pooling)
            return self.head(v_cwt)
