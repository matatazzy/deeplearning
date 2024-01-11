## from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024 (bs,196patch+1cls,patch embedding)   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        # to_qkv(x) = 把dim=x -》 inner_dim*3  =512*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # 分成多个头

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots) # softmax

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # 1个encoder
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), # 多头注意力机制 PreNorm是多头之前做一个Norm
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) # 前馈神经网络，PreNorm在前馈之前做了一个Norm
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) ## 224*224
        patch_height, patch_width = pair(patch_size) ## 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) # 分割patch
        patch_dim = channels * patch_height * patch_width # 将patch拉平
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 原来维度‘b,c,(h*p1),（w*p2'） ->b,(h*w),(p1*p2*c)
            # (h*p1)原图片的高,（w*p2')原图片的宽 -> (h*w) #patch总数
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 生成所有的位置编码  num_patch+1个cls ，dim是维度，彩色照片就是三维
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # CLS的初始化参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # 定义的输入传入transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential( # 多分类
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes) #  映射到多分类
        )

    def forward(self, img):
        x = self.to_patch_embedding(img) ## img 1 3 224 224  输出形状x : 1 196 1024 
        b, n, _ = x.shape ## 

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # 复制batch_size份
        x = torch.cat((cls_tokens, x), dim=1) # 拼接token_embedding 和patch_embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



v = ViT(
    image_size = 224,  # 输入原图像大小 H,W
    patch_size = 16,   # 一个patch的大小
    num_classes = 1000,# CLS 最后map到linear层时的分类数
    dim = 1024,
    depth = 6,         # encoder堆叠个数
    heads = 16,        # 多头注意力机制多少个头，map到多少子空间上
    mlp_dim = 2048,    #
    dropout = 0.1,
    emb_dropout = 0.1  # embedding层后会过一个dropout
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)