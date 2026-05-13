# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import copy
import warnings
import ttnn
from models.experimental.uniad.tt.ttnn_temporal_self_attention import TtTemporalSelfAttention
from models.experimental.uniad.tt.ttnn_spatial_cross_attention import TtSpatialCrossAttention
from models.experimental.uniad.tt.ttnn_ffn import TtFFN


class TtBEVFormerEncoder:
    def __init__(
        self,
        params,
        device,
        num_layers=6,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=256,
        num_heads=8,
        dilation=1,
        kernel_size=(3, 5),
        im2col_step=192,
        feedforward_channels=512,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ):
        super(TtBEVFormerEncoder, self).__init__()
        self.device = device
        self.params = params
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.num_layers = num_layers
        self.fp16_enabled = False

        transformer_layers = dict(
            attn_cfgs=[
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    attention=dict(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        num_levels=1,
                        im2col_step=im2col_step,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=feedforward_channels,
            operation_order=operation_order,
        )

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(TtBEVFormerLayer(self.device, params.layers[f"layer{i}"], **transformer_layers))

    @staticmethod
    def get_reference_points_ttnn(H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device=None, dtype=ttnn.bfloat16):
        if dim == "3d":

            def _linspace_ttnn(start, end, steps):
                idx = ttnn.arange(0, steps, dtype=ttnn.bfloat16, device=device)
                step_size = (end - start) / (steps - 1)
                return ttnn.add(ttnn.multiply(idx, step_size), start)

            # Generate z-values
            z_vals = _linspace_ttnn(0.5, Z - 0.5, num_points_in_pillar)
            z_vals = ttnn.reshape(z_vals, (num_points_in_pillar, 1, 1))
            z_vals = ttnn.expand(z_vals, (num_points_in_pillar, H, W))
            z_vals = ttnn.divide(z_vals, Z)

            # Generate x-values
            x_vals = _linspace_ttnn(0.5, W - 0.5, W)
            x_vals = ttnn.reshape(x_vals, (1, 1, W))
            x_vals = ttnn.expand(x_vals, (num_points_in_pillar, H, W))
            x_vals = ttnn.divide(x_vals, W)

            # Generate y-values
            y_vals = _linspace_ttnn(0.5, H - 0.5, H)
            y_vals = ttnn.reshape(y_vals, (1, H, 1))
            y_vals = ttnn.expand(y_vals, (num_points_in_pillar, H, W))
            y_vals = ttnn.divide(y_vals, H)

            ref = ttnn.stack((x_vals, y_vals, z_vals), dim=-1)  # [P, H, W, 3]
            ref = ttnn.permute(ref, (0, 3, 1, 2))  # [P, 3, H, W]
            ref = ttnn.reshape(ref, (num_points_in_pillar, 3, H * W))
            ref = ttnn.permute(ref, (0, 2, 1))  # [P, H*W, 3]
            ref = ttnn.reshape(ref, (1, num_points_in_pillar, H * W, 3))

            ref = ttnn.repeat(ref, (bs, 1, 1, 1))  # [B, P, HW, 3]
            return ref

        elif dim == "2d":
            y_vals = ttnn.arange(0, H, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
            x_vals = ttnn.arange(0, W, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

            y_vals = ttnn.add(y_vals, 0.5)
            x_vals = ttnn.add(x_vals, 0.5)

            y_vals = ttnn.divide(y_vals, H)
            x_vals = ttnn.divide(x_vals, W)

            y_vals = ttnn.reshape(y_vals, (H, 1))
            y_vals = ttnn.repeat(y_vals, (1, W))  # [H, W]

            x_vals = ttnn.reshape(x_vals, (1, W))
            x_vals = ttnn.repeat(x_vals, (H, 1))  # [H, W]

            y_vals = ttnn.reshape(y_vals, (-1,))
            y_vals = ttnn.unsqueeze(y_vals, 0)  # [1, H*W]
            x_vals = ttnn.reshape(x_vals, (-1,))
            x_vals = ttnn.unsqueeze(x_vals, 0)  # [1, H*W]

            ref = ttnn.stack((x_vals, y_vals), dim=-1)  # [1, H*W, 2]

            ref = ttnn.repeat(ref, (bs, 1, 1))  # [bs, H*W, 2]
            ref = ttnn.reshape(ref, (bs, H * W, 1, 2))  # [bs, H*W, 1, 2]

            return ref

    def point_sampling_ttnn(self, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        reference_points = ttnn.to_torch(reference_points)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = ttnn.from_torch(
            reference_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        lidar2img = ttnn.from_torch(lidar2img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        ref = ttnn.clone(reference_points)

        x, y, z = ttnn.split(ref, (1, 1, 1), dim=3)

        x = x * (pc_range[3] - pc_range[0]) + pc_range[0]
        y = y * (pc_range[4] - pc_range[1]) + pc_range[1]
        z = z * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = ttnn.concat((x, y, z), dim=-1)
        ones = ttnn.ones_like(reference_points[..., :1])
        reference_points = ttnn.concat((reference_points, ones), dim=-1)

        reference_points = ttnn.permute(reference_points, (1, 0, 2, 3))  # [D, B, Q, 4]
        D = reference_points.shape[0]
        B = reference_points.shape[1]
        num_query = reference_points.shape[2]
        num_cam = lidar2img.shape[1]

        reference_points = ttnn.unsqueeze(reference_points, 2)
        reference_points = ttnn.repeat(reference_points, (1, 1, num_cam, 1, 1))
        reference_points = ttnn.unsqueeze(reference_points, -1)

        lidar2img = ttnn.unsqueeze(lidar2img, 0)
        lidar2img = ttnn.unsqueeze(lidar2img, 3)
        lidar2img = ttnn.repeat(lidar2img, (D, 1, 1, num_query, 1, 1))

        reference_points_cam = ttnn.matmul(lidar2img, reference_points)
        reference_points_cam = ttnn.squeeze(reference_points_cam, -1)

        eps = 1e-5
        z = reference_points_cam[..., 2:3]
        bev_mask = z > eps

        reference_points_cam = ttnn.divide(
            reference_points_cam[..., 0:2],
            ttnn.maximum(reference_points_cam[..., 2:3], ttnn.ones_like(reference_points_cam[..., 2:3]) * eps),
        )
        x = reference_points_cam[..., 0]
        y = reference_points_cam[..., 1]

        x = ttnn.divide(x, img_metas[0]["img_shape"][0][1])
        y = ttnn.divide(y, img_metas[0]["img_shape"][0][0])
        x = ttnn.unsqueeze(x, dim=-1)
        y = ttnn.unsqueeze(y, dim=-1)
        reference_points_cam = ttnn.concat([x, y], dim=-1)

        a = reference_points_cam[..., 1:2]
        b = reference_points_cam[..., 0:1]
        y_gt_0 = ttnn.gt(a, 0.0)
        y_lt_1 = ttnn.lt(a, 1.0)
        x_gt_0 = ttnn.gt(b, 0.0)
        x_lt_1 = ttnn.lt(b, 1.0)
        bev_mask = ttnn.logical_and(bev_mask, y_gt_0)
        bev_mask = ttnn.logical_and(bev_mask, y_lt_1)
        bev_mask = ttnn.logical_and(bev_mask, x_gt_0)
        bev_mask = ttnn.logical_and(bev_mask, x_lt_1)

        bev_mask = ttnn.where(ttnn.isnan(bev_mask), ttnn.zeros_like(bev_mask), bev_mask)
        reference_points_cam = ttnn.permute(reference_points_cam, [2, 1, 3, 0, 4])
        bev_mask = ttnn.permute(bev_mask, [2, 1, 3, 0, 4])
        bev_mask = ttnn.squeeze(bev_mask, dim=-1)

        return reference_points_cam, bev_mask

    # TODO Handle fp16
    def __call__(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points_ttnn(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.shape[1],
            device=self.device,
        )

        ref_2d = self.get_reference_points_ttnn(bev_h, bev_w, dim="2d", bs=bev_query.shape[1], device=self.device)

        reference_points_cam, bev_mask = self.point_sampling_ttnn(ref_3d, self.pc_range, kwargs["img_metas"])

        shift_ref_2d = ttnn.clone(ref_2d, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shift = ttnn.reshape(shift, (shift.shape[0], 1, 1, shift.shape[1]))
        shift_ref_2d = shift_ref_2d + shift

        bev_query = ttnn.permute(bev_query, (1, 0, 2))
        bev_pos = ttnn.permute(bev_pos, (1, 0, 2))
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            prev_bev = ttnn.stack([prev_bev, bev_query], 1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, len_bev, -1))
            hybird_ref_2d = ttnn.stack([shift_ref_2d, ref_2d], 1)
            hybird_ref_2d = ttnn.reshape(hybird_ref_2d, (bs * 2, len_bev, num_bev_level, 2))
        else:
            hybird_ref_2d = ttnn.stack([ref_2d, ref_2d], 1)
            hybird_ref_2d = ttnn.reshape(hybird_ref_2d, (bs * 2, len_bev, num_bev_level, 2))
        reference_points_cam = ttnn.to_torch(reference_points_cam)

        # Pre-compute per-camera valid-query indexes and the count tensor
        # once, then reuse across all 6 encoder layers. The original code
        # recomputed both inside every spatial_cross_attention call (5
        # redundant copies per inference). bev_mask is identical across
        # layers because it's a function of (lidar2img, ref_3d, image
        # shape) only — all constants within one forward.
        _precomputed_indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            idx = ttnn.sum(mask_per_img[0], -1)
            idx = ttnn.to_torch(idx).nonzero().squeeze(-1).to(torch.long)
            _precomputed_indexes.append(idx)

        # Also build per-camera scatter-add index tensors on the device, so
        # spatial_cross_attention can run the result-accumulation step
        # (slots[j, idx] += queries[j, i, :len(idx)]) as ttnn.scatter_add
        # instead of pulling the full queries tensor to host every layer.
        # Each entry is shape [bs, len_i, embed_dims] with the same index
        # value broadcast along embed_dims.
        _precomputed_scatter_indexes = []
        _bs_for_scatter = bev_query.shape[0]
        _embed_dims_for_scatter = bev_query.shape[2]
        for idx in _precomputed_indexes:
            len_i = idx.shape[0]
            expanded = idx.view(1, len_i, 1).expand(_bs_for_scatter, len_i, _embed_dims_for_scatter).contiguous()
            expanded_ttnn = ttnn.from_torch(
                expanded.to(torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            _precomputed_scatter_indexes.append(expanded_ttnn)

        # Replace the per-layer host rebatching loop in spatial_cross_attention
        # with a single device-side ttnn.gather. We pad each per-camera index
        # list to max_len with idx[0] (padded outputs are produced but ignored
        # downstream by scatter_add, which only writes the first len_i
        # positions), then concatenate across cameras so a single gather call
        # replaces the per-layer host loop. One gather instead of six avoids
        # the per-camera dispatch overhead.
        _precomputed_max_len = max(idx.shape[0] for idx in _precomputed_indexes)
        _num_cams_for_gather = len(_precomputed_indexes)
        _padded_idx_per_cam = []
        for idx in _precomputed_indexes:
            len_i = idx.shape[0]
            if len_i == 0:
                padded_idx = torch.zeros(_precomputed_max_len, dtype=torch.long)
            elif len_i < _precomputed_max_len:
                pad = idx[0].expand(_precomputed_max_len - len_i)
                padded_idx = torch.cat([idx, pad], dim=0)
            else:
                padded_idx = idx
            _padded_idx_per_cam.append(padded_idx)
        # (num_cams * max_len,) → (bs, num_cams * max_len, embed_dims)
        _concat_idx = torch.cat(_padded_idx_per_cam, dim=0)
        _gather_idx_concat = (
            _concat_idx.view(1, -1, 1)
            .expand(_bs_for_scatter, _num_cams_for_gather * _precomputed_max_len, _embed_dims_for_scatter)
            .contiguous()
        )
        _precomputed_gather_index_concat = ttnn.from_torch(
            _gather_idx_concat.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # reference_points_cam is constant across all 6 encoder layers (it's a
        # function of img_metas only), so build the rebatched tensor once on
        # host using fast torch advanced indexing, then upload once. Saves a
        # per-layer host roundtrip.
        _num_cams_for_rebatch = reference_points_cam.shape[0]
        _D_for_rebatch = reference_points_cam.shape[3]
        _precomputed_ref_rebatch = torch.zeros(
            _bs_for_scatter,
            _num_cams_for_rebatch,
            _precomputed_max_len,
            _D_for_rebatch,
            2,
            dtype=reference_points_cam.dtype,
        )
        for j in range(_bs_for_scatter):
            for i in range(_num_cams_for_rebatch):
                idx_i = _precomputed_indexes[i]
                if idx_i.shape[0] > 0:
                    _precomputed_ref_rebatch[j, i, : idx_i.shape[0]] = reference_points_cam[i, j, idx_i]
        _precomputed_ref_rebatch_ttnn = ttnn.from_torch(
            _precomputed_ref_rebatch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        _precomputed_count = ttnn.sum(bev_mask, -1) > 0
        _precomputed_count = ttnn.permute(_precomputed_count, (1, 2, 0))
        _precomputed_count = ttnn.sum(_precomputed_count, -1)
        _precomputed_count = ttnn.clamp(_precomputed_count, min=1.0)
        _precomputed_count = ttnn.unsqueeze(_precomputed_count, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                _precomputed_indexes=_precomputed_indexes,
                _precomputed_scatter_indexes=_precomputed_scatter_indexes,
                _precomputed_gather_index_concat=_precomputed_gather_index_concat,
                _precomputed_ref_rebatch=_precomputed_ref_rebatch_ttnn,
                _precomputed_max_len=_precomputed_max_len,
                _precomputed_count=_precomputed_count,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return ttnn.stack(intermediate)

        return output


class TtBEVFormerLayer:
    def __init__(
        self,
        device,
        params,
        attn_cfgs,
        feedforward_channels,
        operation_order=None,
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(TtBEVFormerLayer, self).__init__()
        self.params = params
        self.device = device
        self.attn_cfgs = attn_cfgs
        self.feedforward_channels = feedforward_channels
        self.operation_order = operation_order
        self.ffn_num_fcs = ffn_num_fcs
        self.fp16_enabled = False
        self.batch_first = True
        self.attentions = []
        index = 0
        for operation_name in self.operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                if attn_cfgs[index]["type"] == "TemporalSelfAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtTemporalSelfAttention(device, params.attentions[f"attn0"], **attn_cfgs[index])
                    attn_cfgs[index]["type"] = "TemporalSelfAttention"
                elif attn_cfgs[index]["type"] == "SpatialCrossAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = TtSpatialCrossAttention(device, params.attentions[f"attn1"], **attn_cfgs[index])
                    attn_cfgs[index]["type"] = "SpatialCrossAttention"

                self.attentions.append(attention)
                index += 1

        self.pre_norm = operation_order[0] == "norm"

        self.embed_dims = self.attentions[0].embed_dims

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.ffns = []
        num_ffns = operation_order.count("ffn")

        for i in range(num_ffns):
            self.ffns.append(TtFFN(params.ffn[f"ffn{i}"], self.device))

        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def __call__(
        self,
        query,
        key,
        value,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_mask = kwargs.get("bev_mask", None)
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, ttnn.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        # Hoist self_attn spatial_shapes constant out of the per-layer loop
        # — same (bev_h, bev_w) value every iteration; rebuilding it via
        # from_torch on every self_attn was 1 host->device transition per
        # layer × 6 layers.
        if not hasattr(self, "_self_attn_spatial_shapes_cache"):
            self._self_attn_spatial_shapes_cache = None
        if self._self_attn_spatial_shapes_cache is None:
            _sh = torch.tensor([[bev_h, bev_w]])
            self._self_attn_spatial_shapes_cache = ttnn.from_torch(
                _sh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        spatial_shapes_1 = self._self_attn_spatial_shapes_cache

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=spatial_shapes_1,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1

                identity = query

            elif layer == "norm":
                query = ttnn.layer_norm(
                    query,
                    weight=self.params.norms[f"norm{norm_index}"].weight,
                    bias=self.params.norms[f"norm{norm_index}"].bias,
                )
                norm_index += 1

            # spatial cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)

                ffn_index += 1

        return query
