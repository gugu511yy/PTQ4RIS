import datetime
import os
import time
import psutil

import torch
import torch.utils.data
import torch.nn as nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

import argparse 
import random
import copy
from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)

from torch.nn.modules import module
from test_vit import *
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
from ptq4vit_utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import pickle as pkl
from itertools import product
import types
from ptq4vit_utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from ptq4vit_utils.models import get_backbone_net,get_bert_net,get_fusion_net,get_net
from ptq4vit_utils.models import get_fusion_net
from ptq4vit_utils.quantfusionmodel import quant_fusion_model,set_quant_state
from ptq4vit_utils.quant_bert import quant_bert_model,set_quant_state
from quant.set_act_quantize_params import set_act_fusionquantize_params,set_act_all_quantize_params
from quant.set_weight_quantize_params import set_weight_fusionquantize_params,set_weight_all_quantize_params
from ptq4vit_utils.net_wrap import *


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()

            start_inference_time=time.time()

            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    # import pdb
                    # pdb.set_trace()
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            end_inference_time=time.time()
            inference_seconds=end_inference_time-start_inference_time
            metric_logger.update(inference_time=inference_seconds)

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def get_calibration_dataset(image_set,transform,args,num_samples,bert_model=None, model=None, device='cuda'):
    from data.dataset_refer_bert import ReferDataset

    ds=ReferDataset(args,
                    split=image_set,
                    image_transforms=transform,
                    target_transforms=None,
                    eval_mode=True
                    )
    
    calibration_sampler=torch.utils.data.SequentialSampler(ds)
    calibration_loader=torch.utils.data.DataLoader(ds,
                                                   batch_size=1,
                                                   sampler=calibration_sampler,
                                                   num_workers=args.workers)
    calibration_data=[]
    calibration_target=[]
    calibration_sentence=[]
    calibration_embedding=[]
    calibration_l_mask=[]

    idx = 0
    for data in calibration_loader:
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.to(device), target.to(device), \
                                               sentences.to(device), attentions.to(device)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        target = target.cpu().data.numpy()

        embedding_list = []
        l_mask_list = []

        for j in range(sentences.size(-1)):
            l_mask_list.append(attentions[:, :, j].unsqueeze(-1))
            calibration_l_mask.append(torch.cat(l_mask_list, dim=0))
            calibration_sentence.append(sentences[:, :, j])
            if bert_model is not None:
                last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                embedding_list.append(embedding)

                calibration_embedding.append(torch.cat(embedding_list, dim=0))

        calibration_data.append(image)
        calibration_target.append(target)

        idx += 1
        if idx == num_samples:
            break


    calibration_data = torch.cat(calibration_data, dim=0)[:num_samples]
    calibration_sentence=torch.cat(calibration_sentence,dim=0)[:num_samples]
    calibration_l_mask = torch.cat(calibration_l_mask, dim=0)[:num_samples]

    if bert_model is not None:
        calibration_embedding = torch.cat(calibration_embedding, dim=0)[:num_samples]

        return calibration_data, calibration_sentence,calibration_embedding, calibration_l_mask
    else:
        # calibration_embedding = torch.zeros((calibration_data.size(0), sentences.size(1), sentences.size(2)), device=device)
        calibration_embedding = torch.zeros((calibration_data.size(0), 768, sentences.size(1)), device=device)
        return calibration_data, calibration_sentence, calibration_embedding,calibration_l_mask



class cfg_modifier():
    def __init__(self, **kwargs):

        for name, value in kwargs.items():
            setattr(self,name,value)
    def __call__(self, cfg):

        cfg.bit = self.bit_setting

        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg

def main(args):
    seed_all(args.seed)

    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)


    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)

    else:
        bert_model = None

    quant_cfg=init_config(config_name='PTQ4ViT')
    modifier = cfg_modifier(linear_ptq_setting = (1,1,1), metric="hessian", bit_setting = (args.n_bits_w,args.n_bits_a))
    quant_cfg = modifier(quant_cfg)

    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise':args.channel_wise, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}

    w_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise}
    a_params = {'n_bits': args.n_bits_a, 'channel_wise': False}

    all_quantmodel=copy.deepcopy(model)
    all_quantmodel=get_bert_net(all_quantmodel)
    all_quantmodel=get_backbone_net(all_quantmodel)
    all_quantmodel=get_fusion_net(all_quantmodel)
    all_quantmodel=quant_fusion_model(model= all_quantmodel, input_quant_params=a_params,weight_quant_params=w_params)
    all_quantmodel=quant_bert_model(model= all_quantmodel, input_quant_params=a_params,weight_quant_params=w_params)
    all_quantmodel=QuantModel(model=all_quantmodel, weight_quant_params=wq_params, act_quant_params=aq_params)

    wrapped_module=net_wrap.wrap_modules_in_net(all_quantmodel,quant_cfg)

    cali_data, cali_sentence,cali_embedding, cali_l_mask = get_calibration_dataset(args.calibration_dataset, get_transform(args=args), args, num_samples=args.num_samples, bert_model=bert_model, model=model, device=device)

    device=next(all_quantmodel.parameters()).device

    quant_calibrator= HessianQuantCalibrator(all_quantmodel,wrapped_module,cali_data,cali_sentence,cali_embedding, cali_l_mask,sequential=False,batch_size=1)
    quant_calibrator.batching_quant_calib()

    set_weight_all_quantize_params(all_quantmodel)
    set_act_all_quantize_params(all_quantmodel, cali_data=cali_data,cali_sentence=cali_sentence,cali_l_mask=cali_l_mask,batch_size=50) 

    evaluate(all_quantmodel, data_loader_test, bert_model, device=device)
    end=time.time()
    inference_time=end-start


    print(f"calibration size: {args.num_samples} \n")
    print(f"bit settings: {quant_cfg.bit} \n")

    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")

    print(f"Inference Time:{inference_time} seconds") 


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)