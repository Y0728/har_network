# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block   # , PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed

import numpy as np
from pointnet_utils_transformer_pretraining import PointNetEncoder, PatchEmbed,feature_transform_reguliarzer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    # def __init__(self, img_size=224, patch_size=16, in_chans=3,
    #              embed_dim=1024, depth=24, num_heads=16,
    #              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #              mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
    def __init__(self, img_size=16*3, patch_size=48, in_chans=5,
                 embed_dim=128, depth=2, num_heads=8,qkv_bias=True,representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,act_layer=None, weight_init='',
                 decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=None, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, 20, in_chans, embed_dim = embed_dim//2)  # 如果既有point又有target 则用embed_dim/2s
        num_patches = 20    # self.patch_embed.num_patches
        self.embed_target = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            # nn.ReLU(),
            # nn.Linear(128, 256),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # ori
        # self.blocks = nn.ModuleList([
        #     # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qkv_scale = None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)

            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()




    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def patchify_ori(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]   # 16
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    #
    #     h = w = imgs.shape[2] // p   # 每个patch的宽和高
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))  # batch_size, 每个patch的pixel数, seq_len*3
    #     return x
    def patchify(self, imgs):
        """
        imgs: (N, seq_len, channel, pointNum) # (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        p = self.patch_embed.patch_size  # 48

        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p   # 宽path数，高patch数
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))  # batch_size, 高patch数*宽patch数（即patch_num）, 每个patch里边包含的像素*通道（即需要decoder预测的值有多少个）
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1],-1))  # batch_size, 高patch数*宽patch数（即patch_num）, 每个patch里边包含的像素*通道（即需要decoder预测的值有多少个）

        return x   # batch_size, seq_len, channel*pointNum

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # def forward_encoder_onlypoint(self, point_data, mask_ratio):
    #     # embed patches
    #     x, trans_fea_array = self.patch_embed(point_data)
    #
    #     # add pos embed w/o cls token
    #     x = x + self.pos_embed[:, 1:, :]
    #
    #     # masking: length -> length * mask_ratio
    #     x, mask, ids_restore = self.random_masking(x, mask_ratio)
    #
    #     # append cls token
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #
    #     # apply Transformer blocks
    #     for blk in self.blocks:
    #         x = blk(x)
    #     x = self.norm(x)
    #
    #     return x, mask, ids_restore,trans_fea_array

    def forward_encoder(self, point_data, target_data, mask_ratio):
        # embed patches
        sample_fea, trans_fea_array = self.patch_embed(point_data)
        embeded_target_data = self.embed_target(target_data)

        x = torch.cat((embeded_target_data, sample_fea), 2).to(device)  # input = torch.cat((target_data, sample_fea), 2).to(device)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        # ori
        # for blk in self.blocks:
        #     x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore,trans_fea_array

    def forward_decoder(self, x, ids_restore):   #input_x.shape=[batch_size, 6, embed_dim]
        # embed tokens
        x = self.decoder_embed(x)   #output_x=[batch_size, 6, decoder_embed_dim]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)  #input_x.shape=[batch_size, seq_len+1, decoder_embed_dim]  output_x.shape=[batch_size, seq_len+1(20+1), pointNum*channel(48*5)]

        # remove cls token
        x = x[:, 1:, :]

        return x   #x.shape=[batch_size, seq_len, pointNum*channel(48*5)]

    def forward_loss(self, imgs, pred, mask,trans_feat_array):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        mat_diff_loss_scale = 0.001
        mat_diff_loss = 0
        for i, trans_feat in enumerate(trans_feat_array):
            # print(trans_feat_array)
            # print(f"the size of trans_feat_array {trans_feat_array.shape}")  #10, 512, 64, 64]

            mat_diff_loss += feature_transform_reguliarzer(trans_feat)  # trans_feat.shape = (batch_size,64,64)
        point_loss = mat_diff_loss * mat_diff_loss_scale
        target = self.patchify(imgs)   # batch_size, 每个patch的pixel数, seq_len*3
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # 加
        loss += point_loss
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, target,mask_ratio=0.75):
        latent, mask, ids_restore,trans_fea_array = self.forward_encoder(imgs, target,mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask,trans_fea_array)
        return loss, pred, mask, trans_fea_array


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss





def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
