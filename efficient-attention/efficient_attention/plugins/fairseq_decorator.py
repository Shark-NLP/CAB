from typing import Dict, Optional
import logging
import torch
from torch import Tensor


def fsmha(cls):

    @with_incremental_state
    class FSCls(cls):

        def __init__(self, *args, encoder_decoder_attention=False, **kwargs):
            super(FSCls, self).__init__(*args, **kwargs)
            self.encoder_decoder_attention = encoder_decoder_attention
            logging.info(f'Using efficient attention {cls.__name__}')

        @torch.jit.export
        def reorder_incremental_state(
                self,
                incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                new_order: Tensor,
        ):
            """Reorder buffered internal state (for incremental generation)."""
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is not None:
                for k in input_buffer.keys():
                    input_buffer_k = input_buffer[k]
                    if input_buffer_k is not None:
                        if self.encoder_decoder_attention and input_buffer_k.size(
                                0
                        ) == new_order.size(0):
                            break
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
                incremental_state = self._set_input_buffer(incremental_state, input_buffer)
            return incremental_state

        def upgrade_state_dict_named(self, state_dict, name):
            prefix = name + "." if name != "" else ""
            items_to_add = {}
            keys_to_remove = []
            for k in state_dict.keys():
                if k.endswith(prefix + "in_proj_weight"):
                    # in_proj_weight used to be q + k + v with same dimensions
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                    items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                    items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                    keys_to_remove.append(k)

                    k_bias = prefix + "in_proj_bias"
                    if k_bias in state_dict.keys():
                        dim = int(state_dict[k].shape[0] / 3)
                        items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                        items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                               dim: 2 * dim
                                                               ]
                        items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                        keys_to_remove.append(prefix + "in_proj_bias")

            for k in keys_to_remove:
                del state_dict[k]

            for key, value in items_to_add.items():
                state_dict[key] = value

    return FSCls