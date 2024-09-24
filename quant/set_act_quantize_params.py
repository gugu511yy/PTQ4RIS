import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union

from ptq4vit_utils.quantfusionmodel import  QuantLinear,QuantMatMul,QuantConv1d
from ptq4vit_utils.quant_bert import *
# from ptq4vit_utils.quant_bert_normal import QuantMatMul1,QuantMatMul2,QuantLinear_bert
from ptq4vit_utils.repqvit_model import *
from ptq4vit_utils.pdquant_matmul_model import *




def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cali_data,cali_sentence, cali_embedding,cali_l_mask, batch_size: int = 256):
    
    module.set_quant_state(True, True)
    # module.set_quant_state(False, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)


    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))

    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(), cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())

    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)

def set_act_fusionquantize_params(module:Union[QuantConv1d, QuantLinear, QuantMatMul,QuantModel,QuantModule],
                            cali_data,cali_sentence, cali_embedding,cali_l_mask, batch_size: int = 256):
    import pdb
    pdb.set_trace()


    for t in module.modules():
        if isinstance(t, (QuantConv1d,QuantLinear)):
            t.input_quantizer.set_inited(False)
            t.set_quant_state(True, True)
            
        elif isinstance(t,(QuantMatMul)):
            t.quantizer_A.set_inited(False)
            t.quantizer_B.set_inited(False)
            t.set_quant_state(True, True)


    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))
    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            # module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(),cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_embedding[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())
            if not hasattr(module,"text_encoder"):
                module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(),cali_embedding[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())
            else:
                module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(), cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())

    torch.cuda.empty_cache()

    # for t in module.modules():
    #     if isinstance(t, (QuantConv1d, QuantLinear )):
    #         t.set_quant_state(True,True)
    #     elif isinstance(t,QuantMatMul):
    #         t.set_quant_state(True,True)

    for t in module.modules():
        if isinstance(t, ( QuantConv1d,QuantLinear )):
            t.set_quant_state(True,True)
            # set_quant_state(t,True,False)
            t.input_quantizer.set_inited(True)
        elif isinstance(t,(QuantMatMul)):
            t.set_quant_state(True,True)
            # set_quant_state(t,True,False)
            t.quantizer_A.set_inited(True)
            t.quantizer_B.set_inited(True)


def set_act_all_quantize_params(module:Union[QuantConv1d, QuantLinear, QuantLinear_bert,QuantMatMul,QuantMatMul1,QuantMatMul2,QuantModel,QuantModule],
                            cali_data,cali_sentence,cali_l_mask, batch_size: int = 256):

    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)
            t.set_quant_state(True, True)
        elif isinstance(t,(QuantConv1d,QuantLinear,QuantLinear_bert)):
            t.input_quantizer.set_inited(False)
            t.set_quant_state(True,True)
            
        elif isinstance(t,(QuantMatMul,QuantMatMul1,QuantMatMul2)):
            t.quantizer_A.set_inited(False)
            t.quantizer_B.set_inited(False)
            t.set_quant_state(True,True)

    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))

    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            # module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(),cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_embedding[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(), cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())

    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock )):
            t.set_quant_state(True,True)
            t.act_quantizer.set_inited(True)
        elif isinstance(t,(QuantConv1d,QuantLinear,QuantLinear_bert )):
            t.set_quant_state(True,True)
            t.input_quantizer.set_inited(True)
        elif isinstance(t,(QuantMatMul,QuantMatMul1,QuantMatMul2)):
            t.set_quant_state(True,True)
            t.quantizer_A.set_inited(True)
            t.quantizer_B.set_inited(True)


def set_act_all_repq_quantize_params(module:Union[QuantConv2d,QuantConv1d, QuantLinear,QuantMatMul,QuantMatMul1,QuantMatMul2,QuantModel,QuantModule],
                            cali_data,cali_sentence,cali_l_mask, batch_size: int = 256):

    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)
            t.set_quant_state(True, True)
        elif isinstance(t,(QuantConv1d,QuantLinear)):
            t.input_quantizer.set_inited(False)
            t.set_quant_state(True,True)
            
        elif isinstance(t,(QuantMatMul1,QuantMatMul2,QuantMatMul)):
            t.quantizer_A.set_inited(False)
            t.quantizer_B.set_inited(False)
            t.set_quant_state(True,True)

    batch_size = min(batch_size, cali_data.size(0))

    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            # module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(),cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_embedding[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(), cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())

    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock )):
            t.set_quant_state(True,True)
            t.act_quantizer.set_inited(True)
        elif isinstance(t,(QuantConv1d,QuantLinear,QuantLinear_bert )):
            t.set_quant_state(True,True)
            t.input_quantizer.set_inited(True)
        elif isinstance(t,(QuantMatMul,QuantMatMul1,QuantMatMul2)):
            t.set_quant_state(True,True)
            t.quantizer_A.set_inited(True)
            t.quantizer_B.set_inited(True)

def set_act_all_pdquant_quantize_params(module:Union[QuantConv1d, QuantMatMul,QuantModel,QuantModule],
                            cali_data,cali_sentence,cali_l_mask, batch_size: int = 256):


    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)
            t.set_quant_state(True, True)
        elif isinstance(t,QuantConv1d):
            t.input_quantizer.set_inited(False)
            t.set_quant_state(True,True)
            
        elif isinstance(t,QuantMatMul):
            t.quantizer_A.set_inited(False)
            t.quantizer_B.set_inited(False)
            t.set_quant_state(True,True)

    batch_size = min(batch_size, cali_data.size(0))

    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            # module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(),cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_embedding[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())
            module(cali_data[i * batch_size:(i + 1) * batch_size].cuda(), cali_sentence[i * batch_size:(i + 1) * batch_size].cuda(),cali_l_mask[i * batch_size:(i + 1) * batch_size].cuda())

    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, ( QuantModule, BaseQuantBlock )):
            t.set_quant_state(True,True)
            t.act_quantizer.set_inited(True)
        elif isinstance(t,QuantConv1d):
            t.set_quant_state(True,True)
            t.input_quantizer.set_inited(True)
        elif isinstance(t,QuantMatMul):
            t.set_quant_state(True,True)
            t.quantizer_A.set_inited(True)
            t.quantizer_B.set_inited(True)
