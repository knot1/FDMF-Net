import torch
import torch.nn as nn
import torch.nn.functional as F


# class CrossFrequencyInteraction(nn.Module):
#     """
#     Cross-Frequency Interaction Module (CFI)

#     RGB_low  <-> DSM_high
#     RGB_high <-> DSM_low

#     residual interaction
#     """

#     def __init__(self, dim, num_heads=4):
#         super().__init__()

#         self.num_heads = num_heads
#         self.dim = dim
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         # RGB low -> DSM high
#         self.q1 = nn.Conv2d(dim, dim, 1)
#         self.k1 = nn.Conv2d(dim, dim, 1)
#         self.v1 = nn.Conv2d(dim, dim, 1)

#         # RGB high -> DSM low
#         self.q2 = nn.Conv2d(dim, dim, 1)
#         self.k2 = nn.Conv2d(dim, dim, 1)
#         self.v2 = nn.Conv2d(dim, dim, 1)

#         self.proj1 = nn.Conv2d(dim, dim, 1)
#         self.proj2 = nn.Conv2d(dim, dim, 1)

#         self.norm = nn.BatchNorm2d(dim)

#     def attention(self, q, k, v):

#         B, C, H, W = q.shape

#         q = q.reshape(B, self.num_heads, self.head_dim, H * W)
#         k = k.reshape(B, self.num_heads, self.head_dim, H * W)
#         v = v.reshape(B, self.num_heads, self.head_dim, H * W)

#         q = q.permute(0,1,3,2)
#         k = k.permute(0,1,2,3)
#         v = v.permute(0,1,3,2)

#         attn = torch.matmul(q, k) * self.scale
#         attn = torch.softmax(attn, dim=-1)

#         out = torch.matmul(attn, v)

#         out = out.permute(0,1,3,2).reshape(B, C, H, W)

#         return out

#     def forward(self, rgb_low, rgb_high, dsm_low, dsm_high):

#         # --------------------------------
#         # RGB_low <-> DSM_high
#         # --------------------------------

#         q = self.q1(rgb_low)
#         k = self.k1(dsm_high)
#         v = self.v1(dsm_high)

#         interaction1 = self.attention(q, k, v)
#         interaction1 = self.proj1(interaction1)

#         rgb_low = rgb_low + interaction1

#         # --------------------------------
#         # RGB_high <-> DSM_low
#         # --------------------------------

#         q = self.q2(rgb_high)
#         k = self.k2(dsm_low)
#         v = self.v2(dsm_low)

#         interaction2 = self.attention(q, k, v)
#         interaction2 = self.proj2(interaction2)

#         rgb_high = rgb_high + interaction2

#         rgb_low = self.norm(rgb_low)
#         rgb_high = self.norm(rgb_high)

#         return rgb_low, rgb_high, dsm_low, dsm_high

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrequencyInteraction(nn.Module):
    """
    Linear Cross-Frequency Interaction (Linear-CFI)

    RGB_low  <-> DSM_high
    RGB_high <-> DSM_low

    Linear attention version
    Complexity: O(N)
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # projection
        self.q1 = nn.Conv2d(dim, dim, 1)
        self.k1 = nn.Conv2d(dim, dim, 1)
        self.v1 = nn.Conv2d(dim, dim, 1)

        self.q2 = nn.Conv2d(dim, dim, 1)
        self.k2 = nn.Conv2d(dim, dim, 1)
        self.v2 = nn.Conv2d(dim, dim, 1)

        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)

        self.norm = nn.BatchNorm2d(dim)

    def linear_attention(self, q, k, v):

        B, C, H, W = q.shape
        N = H * W

        q = q.reshape(B, self.num_heads, self.head_dim, N)
        k = k.reshape(B, self.num_heads, self.head_dim, N)
        v = v.reshape(B, self.num_heads, self.head_dim, N)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhdn,bhde->bhen", q, context)

        out = out.reshape(B, C, H, W)

        return out

    def forward(self, rgb_low, rgb_high, dsm_low, dsm_high):

        # -------------------------
        # RGB_low ↔ DSM_high
        # -------------------------

        q = self.q1(rgb_low)
        k = self.k1(dsm_high)
        v = self.v1(dsm_high)

        interaction1 = self.linear_attention(q, k, v)
        interaction1 = self.proj1(interaction1)

        rgb_low = rgb_low + interaction1

        # -------------------------
        # RGB_high ↔ DSM_low
        # -------------------------

        q = self.q2(rgb_high)
        k = self.k2(dsm_low)
        v = self.v2(dsm_low)

        interaction2 = self.linear_attention(q, k, v)
        interaction2 = self.proj2(interaction2)

        rgb_high = rgb_high + interaction2

        rgb_low = self.norm(rgb_low)
        rgb_high = self.norm(rgb_high)

        return rgb_low, rgb_high, dsm_low, dsm_high