# Supports VPTQ compression, see https://arxiv.org/abs/2409.17066

import math
from typing import Any, Callable, Dict, List, Optional, Union

from numpy import rec
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs
import vptq.libvptq as vptq_ops

_is_cuda = torch.cuda.is_available() and torch.version.cuda
if _is_cuda:
    import sgl_kernel

class MetaData:
    def __init__(self):
        self.num_codebooks = 1
        self.num_centroids = 0
        self.num_res_centroids = 0
        self.vector_len = 0
        self.group_size = 0
        self.output_size = 0

class VPTQConfig(QuantizationConfig):
    """Config class for VPTQ.

    Reference: https://github.com/microsoft/VPTQ
    """

    def __init__(
        self,
        config_for_layers: Dict[str, Dict[str, Any]],
        shared_layer_config: Dict[str, Dict[str, Any]],
    ) -> None:
        self.config_for_layers = config_for_layers
        self.shared_layer_config = shared_layer_config
        self.use_block_quant = False
        self.use_fp8_w8a8 = False
        self.activation_scheme = "static"

    def __repr__(self) -> str:
        return (
            f"VPTQConfig(config_for_layers={self.config_for_layers}, "
            f"shared_layer_config={self.shared_layer_config})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "vptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VPTQConfig":
        config_for_layers: Dict[str, Any] = {}
        shared_layer_config: Dict[str, Any] = {}
        if "config_for_layers" in config:
            config_for_layers = cls.get_from_keys(config, ["config_for_layers"])
        if "shared_layer_config" in config:
            shared_layer_config = cls.get_from_keys(config, ["shared_layer_config"])
        assert len(config_for_layers) > 0 or len(shared_layer_config) > 0, (
            "VPTQConfig must have at least one of 'config_for_layers'\
             or 'shared_layer_config'"
        )

        return cls(config_for_layers, shared_layer_config)

    def get_config_for_key(self, prefix, key):
        merged_name = ".".join([prefix, key])
        if merged_name in self.config_for_layers:
            return self.config_for_layers[merged_name]
        elif key in self.shared_layer_config:
            return self.shared_layer_config[key]
        else:
            raise ValueError(f"Cannot find config for ({prefix}, {key})")

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["VPTQLinearMethod"]:
        from sglang.srt.layers.moe.ep_moe.layer import EPMoE
        if isinstance(layer, LinearBase):
            linear_name = prefix.split(".")[-1]
            base_name = prefix[: prefix.rfind(".")]
            if linear_name == "qkv_proj":
                quant_config = {
                    "q_proj": self.get_config_for_key(base_name, "q_proj"),
                    "k_proj": self.get_config_for_key(base_name, "k_proj"),
                    "v_proj": self.get_config_for_key(base_name, "v_proj"),
                }
            elif linear_name == "gate_up_proj":
                quant_config = {
                    "gate_proj": self.get_config_for_key(base_name, "gate_proj"),
                    "up_proj": self.get_config_for_key(base_name, "up_proj"),
                }
            else:
                quant_config = self.get_config_for_key(base_name, linear_name)
            return VPTQLinearMethod(quant_config)
        elif isinstance(layer, EPMoE):
            return VPTQMoEMethod(quant_config=quant_config, 
                                 layer_idx=layer.layer_idx,
                                 start_expert_id=layer.start_expert_id, 
                                 end_expert_id=layer.end_expert_id,
                                 hidden_size=layer.hidden_size,
                                 intermediate_size=layer.intermediate_size)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class VPTQLinearMethod(LinearMethodBase):
    """Linear method for VPTQ.

    Args:
        quant_config: The VPTQ quantization config.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        self.quant_config = quant_config

    @staticmethod
    def quantized_weight_loader(
        indice_sizes, narrow_dim=1
    ):  # specific for layer.indices/weight_scale&bias
        def wrap_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: Optional[Union[str, int]] = None,
        ):
            if isinstance(loaded_shard_id, str):
                _loaded_shard_id = ["q", "k", "v"].index(loaded_shard_id)
            else:
                _loaded_shard_id = loaded_shard_id or 0

            shard_sizes = [i[1] - i[0] for i in indice_sizes]
            offset, end = indice_sizes[_loaded_shard_id]
            param_data = param.data
            if loaded_shard_id is not None:
                param_data = param_data.narrow(
                    narrow_dim,
                    sum(shard_sizes[:_loaded_shard_id]),
                    shard_sizes[_loaded_shard_id],
                )

            # split for TP
            loaded_weight = loaded_weight.narrow(narrow_dim, offset, end - offset)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)

        return wrap_weight_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        row_parallel_tp_size = input_size // input_size_per_partition
        col_parallel_tp_size = output_size // sum(output_partition_sizes)

        if params_dtype != torch.half and params_dtype != torch.bfloat16:
            raise ValueError("Only half and bfloat16 are currently supported by vptq")
        quant_config = self.quant_config.get("q_proj", self.quant_config)
        quant_config = quant_config.get("gate_proj", quant_config)

        num_codebooks = quant_config["group_num"]
        num_centroids = quant_config["num_centroids"][1]
        group_size = quant_config["group_size"]
        vector_len = quant_config["vector_lens"][1]
        num_res_centroids = quant_config["num_res_centroids"][1]
        enable_residual = num_res_centroids > 0
        enable_norm = quant_config["enable_norm"]
        enable_perm = quant_config["enable_perm"]
        assert not enable_perm, (
            "perm is not absorbed in this model, please process it \
by `pip install vptq && python -m vptq.tools.pre_process \
--input_path xx --output_path xx`"
        )
        assert input_size == group_size
        group_size = input_size_per_partition
        metadata = MetaData()
        metadata.num_centroids = num_centroids
        metadata.num_res_centroids = num_res_centroids
        metadata.vector_len = vector_len
        metadata.group_size = group_size
        layer.metadata = metadata

        num_linears = len(output_partition_sizes)
        orig_weight_loader = extra_weight_attrs["weight_loader"]

        if enable_norm:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                [
                    [
                        (
                            input_size_per_partition * tp_ind,
                            input_size_per_partition * (tp_ind + 1),
                        )
                        for num in output_partition_sizes
                    ]
                    for tp_ind in range(row_parallel_tp_size)
                ][get_tensor_model_parallel_rank() % row_parallel_tp_size],
                0,
            )
            extra_weight_attrs["weight_loader"] = wrapped_weight_loader

            extra_weight_attrs["output_dim"] = 0
            weight_scale = Parameter(
                torch.empty(input_size_per_partition * num_linears, dtype=params_dtype),
                requires_grad=False,
            )
            weight_bias = Parameter(
                torch.empty(input_size_per_partition * num_linears, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(weight_scale, extra_weight_attrs)
            set_weight_attrs(weight_bias, extra_weight_attrs)
            layer.register_parameter("weight_scale", weight_scale)
            layer.register_parameter("weight_bias", weight_bias)
            extra_weight_attrs["weight_loader"] = orig_weight_loader

        index_bits = int(math.log2(num_centroids))
        res_index_bits = int(math.log2(num_res_centroids)) if enable_residual else 0
        total_index_bits = index_bits + res_index_bits
        packed_groupsize = math.ceil(group_size * total_index_bits / 32)

        indice_sizes = [
            [
                (
                    math.floor(num * tp_ind / vector_len),
                    math.ceil(num * (tp_ind + 1) / vector_len),
                )
                for num in output_partition_sizes
            ]
            for tp_ind in range(col_parallel_tp_size)
        ]
        tp_output_offset = [
            [(num * tp_ind) % vector_len for num in output_partition_sizes]
            for tp_ind in range(col_parallel_tp_size)
        ]
        if col_parallel_tp_size > 1:
            this_rank_indice_sizes = indice_sizes[get_tensor_model_parallel_rank()]
        else:
            this_rank_indice_sizes = indice_sizes[0]
        shard_sizes = [i[1] - i[0] for i in this_rank_indice_sizes]
        num_indices = sum(shard_sizes)
        indices = Parameter(
            torch.empty(
                (num_codebooks, num_indices, packed_groupsize), dtype=torch.int32
            ),
            requires_grad=False,
        )
        if row_parallel_tp_size == 1:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                this_rank_indice_sizes
            )
            extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        set_weight_attrs(
            indices,
            {
                # metadata indicates fixed size concatenated along dim 0
                "output_partition_sizes": output_partition_sizes,
                "output_offset": tp_output_offset,
                "shard_sizes": shard_sizes,
                "input_dim": -1,
            },
        )

        extra_weight_attrs["output_dim"] = 1
        set_weight_attrs(indices, extra_weight_attrs)
        layer.register_parameter("indices", indices)
        extra_weight_attrs["weight_loader"] = orig_weight_loader

        extra_weight_attrs.pop("output_dim")
        extra_weight_attrs["is_metadata"] = True
        centroids = torch.nn.Embedding(
            num_codebooks * num_linears, num_centroids * vector_len, dtype=params_dtype
        )
        set_weight_attrs(centroids.weight, extra_weight_attrs)
        set_weight_attrs(
            centroids.weight,
            {
                # metadata indicates fixed size concatenated along dim 0
                "codebook_sizes": [
                    num_centroids * vector_len for _ in output_partition_sizes
                ],
            },
        )
        layer.centroids = centroids
        # layer.register_parameter("centroids", centroids)
        if enable_residual:
            res_centroids = torch.nn.Embedding(
                num_codebooks * num_linears,
                num_res_centroids * vector_len,
                dtype=params_dtype,
            )
            set_weight_attrs(res_centroids.weight, extra_weight_attrs)
            # layer.register_parameter("res_centroids", res_centroids)
            layer.res_centroids = res_centroids
            set_weight_attrs(
                res_centroids.weight,
                {
                    # metadata indicates fixed size concatenated along dim 1
                    "codebook_sizes": [
                        num_res_centroids * vector_len for _ in output_partition_sizes
                    ],
                },
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale if hasattr(layer, "weight_scale") else None
        weight_bias = layer.weight_bias if hasattr(layer, "weight_bias") else None
        perm = layer.perm if hasattr(layer, "perm") else None
        indices = layer.indices
        output_partition_sizes = getattr(indices, "output_partition_sizes", [])
        centroids = layer.centroids.weight
        res_centroids = (
            layer.res_centroids.weight if hasattr(layer, "res_centroids") else None
        )

        # fall back all unoptimized formats
        return merged_dequantize_gemm(
            x,
            indices,
            centroids,
            res_centroids,
            weight_scale,
            weight_bias,
            perm,
            output_partition_sizes,
            bias,
            layer.metadata,
        )

# Handle QKV projection and gate-up projection
# we will do Q K V separately
def merged_dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    indices: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    res_codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: List[int],
    bias: Optional[torch.Tensor],
    metadata: MetaData,
) -> torch.Tensor:
    output_shape = input.shape[:-1] + (sum(output_partition_sizes),)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    indice_sizes = getattr(indices, "shard_sizes", [])
    output_extra_offsets = getattr(indices, "output_offset", [])
    num_codebooks = indices.shape[0]

    tp_rank = get_tensor_model_parallel_rank()
    input_size = input.shape[-1]
    input_offset = 0
    indice_offset = 0
    output_offset = 0
    codebooks_offset = 0

    num_linears = len(output_partition_sizes)
    for linear_idx, output_size, indice_size in zip(
        range(num_linears), output_partition_sizes, indice_sizes
    ):
        metadata.output_size = output_size
        if len(output_extra_offsets) > 1:
            metadata.output_size = (
                output_size + output_extra_offsets[tp_rank][linear_idx]
            )
        shard_output = optimized_dequantize_gemm(
            input,
            indices.narrow(1, indice_offset, indice_size),
            codebooks.narrow(0, codebooks_offset, num_codebooks),
            res_codebooks.narrow(0, codebooks_offset, num_codebooks) if res_codebooks is not None else None,
            weight_scale.narrow(0, input_offset, input_size),
            weight_bias.narrow(0, input_offset, input_size),
            perm.narrow(0, input_offset, input_size) if perm is not None else None,
            bias if bias is None else bias.narrow(0, output_offset, output_size),
            metadata,
        )

        output_slice = output.narrow(-1, output_offset, output_size)
        if tp_rank > 0 and len(output_extra_offsets) > tp_rank:
            shard_output = shard_output.narrow(
                -1, output_extra_offsets[tp_rank][linear_idx], output_size
            )
        assert output_slice.shape == shard_output.shape
        output_slice.copy_(shard_output)
        output_offset += output_size
        indice_offset += indice_size
        codebooks_offset += num_codebooks
        input_offset += input_size
    return output

# call the optimized version of the dequantized matmul
def optimized_dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    indices: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    res_codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
    metadata: MetaData,
) -> torch.Tensor:
    
    codebooks = codebooks.view(
        metadata.num_codebooks, metadata.num_centroids, metadata.vector_len
    )
     
    enable_residual = False
    res_codebooks_ = None
    if res_codebooks is not None:
        enable_residual = True
        shape = (metadata.num_codebooks, metadata.num_res_centroids, metadata.vector_len)
        res_codebooks_ = res_codebooks.view(shape)
    
    residual_indices = None
    outlier_indices = None
    outlier_centroids_ = None
    enable_outlier = False
    
    enable_perm = perm is not None
    enable_norm = weight_scale is not None and weight_bias is not None

    invert_perm = None
    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        invert_perm = invert_perm.to(torch.uint16).view(torch.int16)
    
    in_features = input.shape[-1]
    out_features = metadata.output_size
    
    if (input.numel() // input.shape[-1] < 3):
        out = vptq_ops.quant_gemv(
            input,
            indices,
            codebooks,
            residual_indices,
            res_codebooks_,
            outlier_indices,
            outlier_centroids_,
            perm,
            weight_scale,
            weight_bias,
            bias,
            in_features,
            out_features,
        )
        return out
    else: 
        weight = vptq_ops.dequant(
            indices,
            codebooks,
            residual_indices,
            res_codebooks_,
            outlier_indices,
            outlier_centroids_,
            invert_perm,
            weight_scale,
            weight_bias,
            metadata.vector_len,
            in_features,
            out_features) 
    
        return F.linear(input, weight, bias)


def dequant_experts(
    indices: torch.IntTensor,
    codebooks: torch.Tensor,
    res_codebooks: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    perm: torch.Tensor,
    local_expert_id: int,
    start_expert_id: int,
    end_expert_id: int,
    bias: Optional[torch.Tensor],
):
    # vptq.dequant
    local_expert_id =  - start_expert_id
     

class VPTQMoEMethod:
    """MoE method for VPTQ.
    Supports loading VPTQ checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support VPTQ quantization

    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config, layer_id,
                 start_expert_id, end_expert_id, 
                 hidden_size, intermediate_size, ):
        self.quant_config = quant_config
        self.start_expert_id = start_expert_id
        self.end_expert_id = end_expert_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_id = layer_id

    def weight_loader(self):
        def _weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            print(f'load {weight_name} for expert {expert_id}, {self.start_expert_id}, {self.end_expert_id}')
            # local_expert_id = expert_id - self.start_expert_id
            # if local_expert_id < 0 or local_expert_id >= param.shape[0]:
            #     pass
            # else:
            #     param.data[local_expert_id] = loaded_weight
        return _weight_loader

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        orig_weight_loader = extra_weight_attrs["weight_loader"]
        proj_prefix = ['up_proj', 'down_proj']
        proj_prefix_mapping = {
            'up_proj': 'w13',
            'down_proj': 'w2'
        }
        
        for proj_prefix in proj_prefix:
            # up_proj and gate_proj merged in one operator
            fused_size = 2 if proj_prefix == 'up_proj' else 1
            
            op_name = f'model.layers.{self.layer_id}.mlp.experts.{self.start_expert_id}.{proj_prefix}'
            layer_config = self.quant_config.config_for_layers[op_name]
            vector_len = layer_config['vector_lens'][1]
            num_centroids = layer_config['num_centroids'][1]
            num_res_centroids = layer_config['num_res_centroids'][1]
            num_codebooks = group_num = layer_config['group_num']
            group_size = layer_config['group_size']
            in_features = layer_config['in_features']
            out_features = layer_config['out_features']
            padding = (-out_features) % vector_len
            num_indices = (out_features + padding) // vector_len
            index_bits = math.ceil(math.log2(num_centroids))
            res_index_bits = math.ceil(math.log2(num_res_centroids)) if num_res_centroids > 0 else 0
            packed_group_size = (group_size * (index_bits + res_index_bits) + 31) // 32 
            is_indice_packed = layer_config['is_indice_packed']
            num_outlier_centroids = layer_config['num_centroids'][0]
            assert is_indice_packed == True and num_outlier_centroids <= 0
            extra_weight_attrs["weight_loader"] = self.weight_loader()
            # create indices
            indices = torch.nn.Parameter(
                torch.empty(
                    num_experts_per_partition * fused_size, num_codebooks, num_indices, packed_group_size, dtype=torch.int32
                ),
                requires_grad=False,
            )
            layer.register_parameter(f'{proj_prefix_mapping[proj_prefix]}_indices', indices)
            set_weight_attrs(indices, extra_weight_attrs)
            
            # create centroids and res_centroids
            centroids = torch.nn.Embedding(
                num_experts_per_partition * fused_size, (num_codebooks * num_centroids * vector_len), dtype=torch.bfloat16
            )
            centroids.weight.requires_grad = False
            setattr(layer, f'{proj_prefix_mapping[proj_prefix]}_centroids', centroids)
            set_weight_attrs(centroids.weight, extra_weight_attrs)
            
            if num_res_centroids > 0:
                res_centroids = torch.nn.Embedding(
                    num_experts_per_partition * fused_size, (num_codebooks * num_res_centroids * vector_len), dtype=torch.bfloat16
                )
                res_centroids.weight.requires_grad = False
                setattr(layer, f'{proj_prefix_mapping[proj_prefix]}_res_centroids', res_centroids)
                set_weight_attrs(res_centroids.weight, extra_weight_attrs)
            
            # create weight scale
            weight_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts_per_partition * fused_size, in_features, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter(f'{proj_prefix_mapping[proj_prefix]}_weight_scale', weight_scale)
            set_weight_attrs(weight_scale, extra_weight_attrs)
            
            # create weight bias
            weight_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts_per_partition * fused_size, in_features, dtype=params_dtype
                ),
                requires_grad=False,
            )
            layer.register_parameter(f'{proj_prefix_mapping[proj_prefix]}_weight_bias', weight_bias)
            set_weight_attrs(weight_bias, extra_weight_attrs)
        
        extra_weight_attrs["weight_loader"] = orig_weight_loader

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor, 
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import select_experts
        # topk
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=layer.top_k,
            renormalize=layer.renormalize,
            use_grouped_topk=layer.use_grouped_topk,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            correction_bias=layer.correction_bias,
            custom_routing_function=layer.custom_routing_function,
        )

        # preprocess
        from sglang.srt.layers.moe.ep_moe.kernels import run_moe_ep_preproess
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, layer.num_experts
        )

        # gateup input
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * layer.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        from sglang.srt.layers.moe.ep_moe.kernels import pre_reorder_triton_kernel
        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            layer.start_expert_id,
            layer.end_expert_id,
            layer.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        seg_indptr_cur_rank = seg_indptr[layer.start_expert_id : layer.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            layer.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        # dequantize the weight from rank
        print(f'gateup_input shape: {gateup_input.shape}')
        print(f'reorder_topk_ids: {reorder_topk_ids}')
        print(f'gateup_weight_dim,')
        
        gateup_weight = torch.empty(
            self.num_experts_per_partition,
            self.hidden_size,
            self.intermediate_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # gateup_weight_dim = 1024
        gateup_weight = dequant_experts(
            gateup_weight,
            reorder_topk_ids,
            layer.w13_centroids,
            None,
            layer.w13_weight_scale,
            layer.w13_weight_bias,
            None,
            layer.start_expert_id,
            layer.end_expert_id,
            None)
         
        # only fill selected experts
        gateup_output = torch.empty(
            gateup_input.shape[0],
            gateup_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        gateup_output = layer.grouped_gemm_runner(
            a=gateup_input,
            b=gateup_weight,
            c=gateup_output,
            batch_size=layer.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=layer.use_fp8_w8a8,
            scale_a=None,
            scale_b=None,
            block_shape=None,
        )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        
        layer.w2_input_scale = torch.ones(
            layer.num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_triton_kernel
        from sglang.srt.layers.moe.ep_moe.kernels import gelu_and_mul_triton_kernel
        if layer.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                None,
                layer.start_expert_id,
                layer.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif layer.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                None,
                layer.start_expert_id,
                layer.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {layer.activation=}")

        # GroupGemm-1
        # dequantize from rank
        down_weight = vptq_ops.dequant(
            layer.w2_indices,
            layer.w2_centroids,
            None,
            None,
            None,
            None,
            None,
        )
        
        down_output = torch.empty(
            down_input.shape[0],
            down_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=down_weight,
            c=down_output,
            batch_size=layer.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=layer.use_fp8_w8a8,
            scale_a=None,
            scale_b=None,
            block_shape=layer.block_shape,
        )

        # PostReorder
        from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel
        output = torch.empty_like(hidden_states)
        post_reorder_triton_kernel[(hidden_states.size(0),)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            layer.start_expert_id,
            layer.end_expert_id,
            layer.top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )
        return output
        