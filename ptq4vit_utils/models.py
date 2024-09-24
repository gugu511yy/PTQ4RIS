from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from lib.backbone import WindowAttention,SpatialImageLanguageAttention
from bert.modeling_bert import * 
from bert.modeling_bert import BertSelfAttention



def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def bert_self_attention_forward(self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False):
    mixed_query_layer = self.query(hidden_states)

    if encoder_hidden_states is not None:
        mixed_key_layer = self.key(encoder_hidden_states)
        mixed_value_layer = self.value(encoder_hidden_states)
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    attention_scores = self.matmul1(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = self.softmax(attention_scores)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = self.matmul2(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs

def image_lang_attention_forward(self, x, l, l_mask):
    B, HW = x.size(0), x.size(1)
    x = x.permute(0, 2, 1)
    l_mask = l_mask.permute(0, 2, 1)
    query = self.f_query(x)
    query = query.permute(0, 2, 1)
    key = self.f_key(l)
    value = self.f_value(l)
    key = key * l_mask
    value = value * l_mask
    n_l = value.size(-1)
    query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
    # (b, num_heads, H*W, self.key_channels//self.num_heads)
    key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
    # (b, num_heads, self.key_channels//self.num_heads, n_l)
    value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
    # # (b, num_heads, self.value_channels//self.num_heads, n_l)
    l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

    sim_map = self.matmul1(query, key)  # (B, self.num_heads, H*W, N_l)
    sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

    sim_map = sim_map + (1e4*l_mask - 1e4)
    sim_map = self.softmax(sim_map)  # (B, num_heads, h*w, N_l)
    out = self.matmul2(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
    out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
    out = out.permute(0, 2, 1)  # (B, value_channels, HW)
    out = self.W(out)  # (B, value_channels, HW)
    out = out.permute(0, 2, 1)  # (B, HW, value_channels)

    return out

def get_backbone_net(model):
    for name, module in model.named_modules():
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)

    model.cuda()
    model.eval()
    return model

def get_fusion_net(model):
    for name,module in model.named_modules():
        if  isinstance(module,SpatialImageLanguageAttention):
            setattr(module,"matmul1",MatMul())
            setattr(module,"matmul2",MatMul())

            module.forward=MethodType(image_lang_attention_forward,module)

    model.cuda()
    model.eval()
    return model


def get_bert_net(model):
    for name,module in model.named_modules():
        if "text_encoder" in name and isinstance(module,BertSelfAttention):
            setattr(module,"matmul1",MatMul())
            setattr(module,"matmul2",MatMul())

            module.forward=MethodType(bert_self_attention_forward,module)

    model.cuda()
    model.eval()
    return model


# def get_net(model):
#     module_config = {
#         WindowAttention: {
#             "attributes": ["matmul1", "matmul2"],
#             "attribute_values": [MatMul(), MatMul()],
#             "forward_method": window_attention_forward
#         },
#         BertSelfAttention: {
#             "attributes": ["matmul1", "matmul2"],
#             "attribute_values": [MatMul(), MatMul()],
#             "forward_method": bert_self_attention_forward
#         },
#         SpatialImageLanguageAttention: {
#             "attributes": ["matmul1", "matmul2"],
#             "attribute_values": [MatMul(), MatMul()],
#             "forward_method": image_lang_attention_forward
#         }
#     }
#
#     for name, module in model.named_modules():
#         for module_type, config in module_config.items():
#             if isinstance(module, module_type):
#                 for attr, value in zip(config["attributes"], config["attribute_values"]):
#                     setattr(module, attr, value)
#                 module.forward = MethodType(config["forward_method"], module)
#
#     model.cuda()
#     model.eval()
#     return model



