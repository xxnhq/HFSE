import torch
import torch.nn as nn
import torch.nn.functional as Fn


class HarmonicIntegration(nn.Module):
    def __init__(self, in_channels, heads=4, head_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else in_channels // heads

        assert in_channels % heads == 0, "in_channels must be divisible by heads"

        self.key_conv = nn.Conv2d(in_channels, heads * self.head_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, heads * self.head_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels, heads * self.head_dim, kernel_size=1)

        self.h_conv = nn.Conv2d(heads * self.head_dim, heads * self.head_dim, kernel_size=(1, 3), padding=(0, 1))

        self.out_conv = nn.Conv2d(heads * self.head_dim, in_channels, kernel_size=1)

        self.scale = self.head_dim ** -0.5

        self.eps = 1e-5

    def forward(self, X, Q):
        # X: (B, C, T, F)
        # Q: (B, Nc, F)
        b, c, t, f = X.size()
        h = self.heads

        K = self.key_conv(X)
        V = self.value_conv(X)

        K = K.view(b, h, self.head_dim, t, f)

        attention_scores = torch.einsum('bhdtf,bnf->bhdtn', K, Q)  # (B, H, F, Nc)
        attention_scores = attention_scores * self.scale

        attention_weights = Fn.softmax(attention_scores, dim=-1)

        harmonic_features = torch.einsum('bhdtn,bnf->bhdtf', attention_weights, Q)  # (B, H, D, T, F)

        harmonic_features = harmonic_features.reshape(b, h * self.head_dim, t, f)  # (B, heads*dim, T, F)

        harmonic_features = self.h_conv(harmonic_features)

        modulated_features = V * harmonic_features

        output = self.out_conv(modulated_features)  # (B, C, T, F)

        return output


class HarmonicAttention(nn.Module):
    def __init__(self, channels=1, hidden_dim=64, heads=4):
        super(HarmonicAttention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU()
        )
        self.hint = HarmonicIntegration(hidden_dim, heads=heads)

        self.final_conv = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, X, Q):
        # X: (B,T,F)
        X = X.unsqueeze(1)

        out = self.conv(X)

        h_out = self.hint(out, Q=Q)

        out_final = self.final_conv(h_out).squeeze(1)

        return out_final
