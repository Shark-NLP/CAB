from typing import Optional, Tuple, Dict

import torch.nn as nn
from torch import Tensor


class AbstractAttention(nn.Module):

    def __init__(self, cross=False, causal=False, **kwargs) -> None:
        super(AbstractAttention, self).__init__()
        self.name = f'{self.__class__.__name__}.{hash(self)}'
        self.causal=causal  
        self.cross=cross


    def _reset_parameters(self):
        raise NotImplementedError

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        return incremental_state[self.name] if incremental_state and self.name in incremental_state is not None else {}

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        incremental_state[self.name] = buffer
        return incremental_state

    def _apply_attention(self, *args, **kwargs):
        raise NotImplementedError

    def _get_saved_states(self, *args, **kwargs):
        raise NotImplementedError

    def _update_saved_states(self, *args, **kwargs):
        raise NotImplementedError
