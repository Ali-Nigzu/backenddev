import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.models.volo import VOLO

from .cross_bottleneck_attn import CrossBottleneckAttn

__all__ = ["MiVOLOModel", "create_mivolo_d1_224"]

def get_output_size(input_shape, conv_layer):
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride

    output_size = [
        ((input_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i]) + 1 for i in range(2)
    ]
    return output_size

def get_output_size_module(input_size, stem):
    output_size = input_size

    for module in stem:
        if isinstance(module, nn.Conv2d):
            output_size = [
                (
                    (output_size[i] + 2 * module.padding[i] - module.dilation[i] * (module.kernel_size[i] - 1) - 1)
                    // module.stride[i]
                )
                + 1
                for i in range(2)
            ]

    return output_size

class PatchEmbed(nn.Module):

    def __init__(
        self, img_size=224, stem_conv=False, stem_stride=1, patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans in [3, 6]
        self.with_persons_model = in_chans == 6
        self.use_cross_attn = True

        if stem_conv:
            if not self.with_persons_model:
                self.conv = self.create_stem(stem_stride, in_chans, hidden_dim)
            else:
                self.conv = True  # just to match interface
                self.conv1 = self.create_stem(stem_stride, 3, hidden_dim)
                self.conv2 = self.create_stem(stem_stride, 3, hidden_dim)
        else:
            self.conv = None

        if self.with_persons_model:

            self.proj1 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )
            self.proj2 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

            stem_out_shape = get_output_size_module((img_size, img_size), self.conv1)
            self.proj_output_size = get_output_size(stem_out_shape, self.proj1)

            self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=self.proj_output_size)

        else:
            self.proj = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim**2

    def create_stem(self, stem_stride, in_chans, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.conv is not None:
            if self.with_persons_model:
                x1 = x[:, :3]
                x2 = x[:, 3:]

                x1 = self.conv1(x1)
                x1 = self.proj1(x1)

                x2 = self.conv2(x2)
                x2 = self.proj2(x2)

                x = torch.cat([x1, x2], dim=1)
                x = self.map(x)
            else:
                x = self.conv(x)
                x = self.proj(x)  # B, C, H, W

        return x

class MiVOLOModel(VOLO):

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        use_mix_token=False,
        pooling_scale=2,
    ):
        super().__init__(
            layers,
            img_size,
            in_chans,
            num_classes,
            global_pool,
            patch_size,
            stem_hidden_dim,
            embed_dims,
            num_heads,
            downsamples,
            outlook_attention,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            post_layers,
            use_aux_head,
            use_mix_token,
            pooling_scale,
        )

        im_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = PatchEmbed(
            img_size=im_size,
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        x = self.forward_tokens(x)

        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False, targets=None, epoch=None):
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out

        features = out
        fds_enabled = hasattr(self, "_fds_forward")
        if fds_enabled:
            features = self._fds_forward(features, targets, epoch)

        out = self.head(features)
        if self.aux_head is not None:
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]

        return (out, features) if (fds_enabled and self.training) else out

    def forward(self, x, targets=None, epoch=None):

        x = self.forward_features(x)
        x = self.forward_head(x, targets=targets, epoch=epoch)
        return x

def create_mivolo_d1_224(num_classes=3, in_chans=6):
    return MiVOLOModel(
        layers=(4, 4, 8, 2),
        img_size=224,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=(192, 384, 384, 384),
        num_heads=(6, 12, 12, 12),
    )
