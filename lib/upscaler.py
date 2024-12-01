# BSD 3-Clause License
#
# Copyright (c) 2021, Sberbank AI
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import einops
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# https://huggingface.co/ai-forever/Real-ESRGAN
HF_MODELS = {
    2: {
        "repo_id": "ai-forever/Real-ESRGAN",
        "filename": "RealESRGAN_x2.pth",
    },
    4: {
        "repo_id": "ai-forever/Real-ESRGAN",
        "filename": "RealESRGAN_x4.pth",
    },
    # 8: {
    #     "repo_id": "ai-forever/Real-ESRGAN",
    #     "filename": "RealESRGAN_x8.pth",
    # },
}


def pad_reflect(image, pad_size):
    image_size = image.shape
    height, width = image_size[:2]
    new_image = np.zeros([height + pad_size * 2, width + pad_size * 2, image_size[2]]).astype(np.uint8)
    new_image[pad_size:-pad_size, pad_size:-pad_size, :] = image
    new_image[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0)  # # top
    new_image[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0)  # # bottom
    new_image[:, 0:pad_size, :] = np.flip(new_image[:, pad_size : pad_size * 2, :], axis=1)  # # left
    new_image[:, -pad_size:, :] = np.flip(new_image[:, -pad_size * 2 : -pad_size, :], axis=1)  # right
    return new_image


def unpad_image(image, pad_size):
    return image[pad_size:-pad_size, pad_size:-pad_size, :]


def pad_patch(image_patch, padding_size, channel_last=True):
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            "edge",
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            "edge",
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), "edge")

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    patches = []
    xmax, ymax, _ = padded_image.shape
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    return np.array(patches), padded_image.shape


def stitch_together(patches, padded_image_shape, target_shape, padding_size=4):
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
            row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0 : target_shape[0], 0 : target_shape[1], :]


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    _, _, h, w = x.shape
    assert h % scale == 0 and w % scale == 0, "Height and width must be divisible by scale"
    return einops.rearrange(
        x,
        "b c (h s1) (w s2) -> b (c s1 s2) h w",
        s1=scale,
        s2=scale,
    )


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # scale the residual by a factor of 0.2


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # scale the residual by a factor of 0.2


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RealESRGAN:
    def __init__(self, scale=2, device=None):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

    def to(self, device):
        self.device = device
        self.model.to(device=device)

    def load_weights(self):
        assert self.scale in [2, 4], "You can download models only with scales: 2, 4"
        config = HF_MODELS[self.scale]
        cache_path = hf_hub_download(config["repo_id"], filename=config["filename"])
        loadnet = torch.load(cache_path, weights_only=True)
        if "params" in loadnet:
            self.model.load_state_dict(loadnet["params"], strict=True)
        elif "params_ema" in loadnet:
            self.model.load_state_dict(loadnet["params_ema"], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval().to(device=self.device)

    @torch.autocast("cuda")
    def predict(self, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15):
        if not isinstance(lr_image, np.ndarray):
            lr_image = np.array(lr_image)
        if lr_image.min() < 0.0:
            lr_image = (lr_image + 1.0) / 2.0
        if lr_image.max() <= 1.0:
            lr_image = lr_image * 255.0
        lr_image = pad_reflect(lr_image, pad_size)
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image,
            patch_size=patches_size,
            padding_size=padding,
        )
        patches = torch.Tensor(patches / 255.0)
        image = einops.rearrange(patches, "b h w c -> b c h w").to(device=self.device)

        with torch.inference_mode():
            res = self.model(image[0:batch_size])
            for i in range(batch_size, image.shape[0], batch_size):
                res = torch.cat((res, self.model(image[i : i + batch_size])), 0)

        scale = self.scale
        sr_image = einops.rearrange(res.clamp(0, 1), "b c h w -> b h w c").cpu().numpy()
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        sr_image = stitch_together(
            sr_image,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * scale,
        )
        sr_image = (sr_image * 255).astype(np.uint8)
        sr_image = unpad_image(sr_image, pad_size * scale)
        sr_image = Image.fromarray(sr_image)
        return sr_image
