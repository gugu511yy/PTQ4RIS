import torch
import time
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import parameter
from copy import deepcopy
from ptq4vit_utils.models import MatMul
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ptq4vit_utils.quantfusionmodel import  *


def quant_bert_model(model, input_quant_params={}, weight_quant_params={}):

    input_quant_params_matmul = deepcopy(input_quant_params)
    input_quant_params_matmul['channel_wise'] = False

    module_dict={}
    
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx] 
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
       
        if  "text_encoder" in name :
            if isinstance(m, nn.Linear):
                # Linear Layer
                idx = idx + 1 if idx != 0 else idx

                new_m = QuantLinear_bert(m.in_features, m.out_features, input_quant_params, weight_quant_params)
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias
                setattr(father_module, name[idx:], new_m)

            elif isinstance(m, MatMul):
                # Matmul Layer
                idx = idx + 1 if idx != 0 else idx
                if 'matmul2' in name:
                    new_m = QuantMatMul2(input_quant_params_matmul)
                else:
                    new_m = QuantMatMul1(input_quant_params_matmul)
                setattr(father_module, name[idx:], new_m)

    return model

def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantLinear_bert,QuantMatMul1,QuantMatMul2)):
            m.set_quant_state(input_quant, weight_quant)

class QuantLinear_bert(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={}):
        super(QuantLinear_bert, self).__init__(in_features, out_features)

        input_quant_params_bert_input = deepcopy(input_quant_params)
        input_quant_params_bert_input['channel_wise'] = False
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.input_quantizer = BERTVARActQuantizer(**input_quant_params_bert_input)

        


        self.use_input_quant = False
        self.use_weight_quant = False


    def __repr__(self):
        s = super(QuantLinear_bert, self).__repr__()
        s = "(" + s + "weight_quant={},input_quant={})".format(self.use_weight_quant, self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def calculate_mse(self, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return torch.mean((original - quantized) ** 2)

    def forward(self, x):
        if self.use_input_quant:
            x = self.input_quantizer(x)
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight
        out = F.linear(x, weight=w, bias=self.bias)
        return out

class QuantMatMul1(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul1, self).__init__()
        input_quant_params_matmul = deepcopy(input_quant_params)
        if 'channel-wise' in input_quant_params_matmul:
            input_quant_params_matmul['channel_wise'] = False
        self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)
        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul1, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    # 设置量化状态
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
    # 前向传播
    def forward(self, A, B):

        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        
        out = A @ B
        return out

class QuantMatMul2(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul2, self).__init__()
        input_quant_params_matmul2 = deepcopy(input_quant_params)
        if 'channel_wise' in input_quant_params_matmul2:
            input_quant_params_matmul2['channel_wise'] = False

        self.quantizer_A = UniformQuantizer(**input_quant_params_matmul2)
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul2)
        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul2, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, A, B):
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        
        out = A @ B
        return out
    
from scipy import stats
class BERTCONFActQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, max_iters: int = 10, confidence_level: float = 0.99):
        super(BERTCONFActQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.deltas = []
        self.zero_points = []
        self.means = []
        self.thresholds = []
        self.confidence_level = confidence_level
        self.outlier_values = []
        self.outlier_counts = []
        self.inited = False
        self.channel_wise = channel_wise
        self.max_iters = max_iters

    def __repr__(self):
        s = super(BERTCONFActQuantizer, self).__repr__()
        s = "(" + s + " n_bits={},inited={}, channel_wise={},max_iters={},confidence_level={})".format(
            self.n_bits, self.inited, self.channel_wise, self.max_iters, self.confidence_level)
        return s

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_dequant = torch.zeros_like(x)
        x_abs = x.abs()

        group_indices = torch.zeros_like(x, dtype=torch.long)
        for i, threshold in enumerate(self.thresholds):
            group_indices += (x_abs <= threshold).long()

        for group in range(len(self.thresholds)):
            mask = (group_indices == group)
            x_group = x[mask]

            if x_group.numel() > 0:
                delta = self.deltas[group]
                zero_point = self.zero_points[group]

                x_int = torch.round(x_group / delta) + zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant[mask] = (x_quant - zero_point) * delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            raise NotImplementedError("Channel-wise quantization not implemented for this method.")
        else:
            x_clone = x.clone().detach()

            for i in range(self.max_iters):
                non_zero_elements = x_clone[x_clone != 0]
                if non_zero_elements.numel() == 0:
                    break
                mean_val = non_zero_elements.abs().mean()
                std_dev = non_zero_elements.abs().std()
                ci_lower, ci_upper = stats.norm.interval(self.confidence_level, loc=mean_val.item(), scale=std_dev.item())
                threshold = ci_upper
                self.thresholds.append(threshold)

                outliers = (x_clone.abs() > threshold)
                self.outlier_values.append(x_clone[outliers].tolist())
                self.outlier_counts.append(outliers.sum().item())
                inliers = x_clone[~outliers]
                if inliers.numel() >= x_clone.numel():
                    break

                delta, zero_point = self.compute_scale_zero_point(inliers)
                self.deltas.append(delta)
                self.zero_points.append(zero_point)

                x_clone[~outliers] = 0

            while len(self.deltas) < len(self.thresholds):
                self.deltas.append(self.deltas[-1])
                self.zero_points.append(self.zero_points[-1])

    def compute_scale_zero_point(self, x: torch.Tensor):
        x_min = x.min()
        x_max = x.max()
        delta = (x_max - x_min) / (self.n_levels - 1)
        zero_point = (-x_min / delta).round()
        return delta, zero_point


class BERTVARActQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, max_iters: int = 10):
        super(BERTVARActQuantizer, self).__init__()
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.deltas = []
        self.zero_points = []
        self.means = []
        self.thresholds = []  # 存储每次迭代的阈值
        self.outlier_values = []
        self.outlier_counts = []
        self.inited = False
        self.channel_wise = channel_wise
        self.max_iters = max_iters

    def __repr__(self):
        s = super(BERTVARActQuantizer, self).__repr__()
        s = "(" + s + " n_bits={},inited={}, channel_wise={},max_iters={})".format(self.n_bits, self.inited, self.channel_wise,self.max_iters)
        return s

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_dequant = torch.zeros_like(x)
        x_abs = x.abs()

        group_indices = torch.zeros_like(x, dtype=torch.long)
        for i, threshold in enumerate(self.thresholds):
            if i == 0:
                group_indices += (x_abs <= threshold).long()*i
            else:
                prev_threshold = self.thresholds[i - 1]
                group_indices += ((x_abs > prev_threshold) & (x_abs <= threshold)).long()*i

        group_indices = torch.clamp(group_indices, 0, len(self.deltas) - 1)

        for group in range(len(self.thresholds)):
            mask = (group_indices == group)
            x_group = x[mask]

            if x_group.numel() > 0:
                delta = self.deltas[group]
                zero_point = self.zero_points[group]

                x_int = torch.round(x_group / delta) + zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant[mask] = (x_quant - zero_point) * delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            raise NotImplementedError("Channel-wise quantization not implemented for this method.")
        else:
            x_clone = x.clone().detach()
            for i in range(self.max_iters):
                non_zero_elements = x_clone[x_clone != 0]

                if non_zero_elements.numel() == 0:
                    break

                mean_val = non_zero_elements.abs().mean()
                variance_val = non_zero_elements.abs().var()
                self.means.append(mean_val.item())
                # self.variances.append(variance_val.item())

                threshold = mean_val + 3 * variance_val.sqrt()
                self.thresholds.append(threshold.item())

                if threshold > x_clone.abs().max():
                    inliers = non_zero_elements
                    outliers = (x_clone.abs() > threshold)
                else:
                    outliers = (x_clone.abs() > threshold)
                    self.outlier_values.append(x_clone[outliers].tolist())
                    self.outlier_counts.append(outliers.sum().item())

                    inliers = x_clone[~outliers]

                if outliers.numel() == 0:
                    break

                delta, zero_point = self.compute_scale_zero_point(inliers)
                self.deltas.append(delta)
                self.zero_points.append(zero_point)

                x_clone[~outliers] = 0

            self.save_outlier_data()

    def compute_scale_zero_point(self, x: torch.Tensor):
        x_min = x.min()
        x_max = x.max()
        delta = (x_max - x_min) / (self.n_levels - 1)
        zero_point = (-x_min / delta).round()
        return delta, zero_point


class BERTMADActQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, max_iters: int = 10):
        super(BERTMADActQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.deltas = []
        self.zero_points = []
        self.medians = []
        self.thresholds=[]
        self.outlier_values = []
        self.outlier_counts = []
        self.inited = False
        self.channel_wise = channel_wise
        self.max_iters = max_iters

    def __repr__(self):
        s = super(BERTMADActQuantizer, self).__repr__()
        s = "(" + s + " n_bits={}, inited={}, channel_wise={},max_iters={})".format(self.n_bits, self.inited, self.channel_wise,self.max_iters)
        return s

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_dequant = torch.zeros_like(x)
        x_abs = x.abs()

        group_indices = torch.zeros_like(x, dtype=torch.long)

        for i, median in enumerate(self.medians):
            group_indices += (x_abs >= median).long()
            
        group_indices = torch.clamp(group_indices, 0, len(self.deltas) - 1)

        for group in range(len(self.medians)):
            mask = (group_indices == group)
            x_group = x[mask]

            if x_group.numel() > 0:
                delta = self.deltas[group]
                zero_point = self.zero_points[group]

                x_int = torch.round(x_group / delta) + zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant[mask] = (x_quant - zero_point) * delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            raise NotImplementedError("Channel-wise quantization not implemented for this method.")
        else:
            x_clone = x.clone().detach()
            for i in range(self.max_iters):
                non_zero_elements = x_clone[x_clone != 0]
                if non_zero_elements.numel() == 0:
                    break

                median_val = non_zero_elements.abs().median()
                mad = (non_zero_elements.abs() - median_val).abs().median()
                threshold = median_val + 3 * mad

                self.medians.append(median_val.item())

                outliers = (x_clone.abs() > threshold)
                self.outlier_values.append(x_clone[outliers].tolist())
                self.outlier_counts.append(outliers.sum().item())

                inliers = x_clone[~outliers]
                if inliers.numel() >= x_clone.numel():
                    break

                delta, zero_point = self.compute_scale_zero_point(inliers)
                self.deltas.append(delta)
                self.zero_points.append(zero_point)

                x_clone[~outliers] = 0

            self.save_outlier_data()

    def compute_scale_zero_point(self, x: torch.Tensor):
        x_min = x.min()
        x_max = x.max()
        delta = (x_max - x_min) / (self.n_levels - 1)
        zero_point = (-x_min / delta).round()
        return delta, zero_point

    def save_outlier_data(self):
        with open("outliers.txt", "w") as f:
            for i, (values, count) in enumerate(zip(self.outlier_values, self.outlier_counts)):
                f.write(f"Iteration {i + 1}:\n")
                f.write(f"Outlier count: {count}\n")
                f.write(f"Outlier values: {values}\n")
                f.write("\n")


class BERTActQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, max_iters: int = 10):
        super(BERTActQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.deltas = []
        self.zero_points = []
        self.means = []
        self.outlier_values = []
        self.outlier_counts = []
        self.inited = False
        self.channel_wise = channel_wise
        self.max_iters = max_iters

    def __repr__(self):
        s = super(BERTActQuantizer, self).__repr__()
        s = "(" + s + " n_bits={},inited={}, channel_wise={},max_iters={})".format(self.n_bits, self.inited, self.channel_wise,self.max_iters)
        return s

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.init_quantization_scale(x, self.channel_wise)
            self.inited = True
        x_dequant = torch.zeros_like(x)
        x_abs = x.abs()
        group_indices = torch.zeros_like(x, dtype=torch.long)

        for i, mean in enumerate(self.means):
            if i == 0:
                group_indices += (x_abs <= mean).long()*i
            else:
                prev_mean = self.means[i - 1]
                group_indices += ((x_abs > prev_mean) & (x_abs <= mean)).long()*i

        group_indices = torch.clamp(group_indices, 0, len(self.deltas) - 1)

        for group in range(len(self.means)):
            mask = (group_indices == group)
            x_group = x[mask]

            if x_group.numel() > 0:
                delta = self.deltas[group]
                zero_point = self.zero_points[group]

                x_int = torch.round(x_group / delta) + zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant[mask] = (x_quant - zero_point) * delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            raise NotImplementedError("Channel-wise quantization not implemented for this method.")
        else:
            x_clone = x.clone().detach()

            for i in range(self.max_iters):
                non_zero_elements = x_clone[x_clone != 0]
                mean_val = non_zero_elements.abs().mean()
                self.means.append(mean_val.item())

                outliers = (x_clone.abs() > mean_val)
                self.outlier_values.append(x_clone[outliers].tolist())

                inliers = x_clone[~outliers]
                if inliers.numel() >= x_clone.numel():
                    break

                delta, zero_point = self.compute_scale_zero_point(inliers)
                self.deltas.append(delta)
                self.zero_points.append(zero_point)

                x_clone[~outliers] = 0


    def compute_scale_zero_point(self, x: torch.Tensor):
        x_min = x.min()
        x_max = x.max()
        delta = (x_max - x_min) / (self.n_levels - 1)
        zero_point = (-x_min / delta).round()
        return delta, zero_point

    # def compute_scale_zero_point(self, x: torch.Tensor):
    #     best_score = 1e+10
    #     best_new_max, best_new_min = None, None
    #     for pct in [0.999, 0.9999, 0.99999]:
    #         try:
    #             new_max = torch.quantile(x.reshape(-1), pct)
    #             new_min = torch.quantile(x.reshape(-1), 1.0 - pct)
    #         except:
    #             new_max = torch.tensor(np.percentile(
    #                 x.reshape(-1).cpu(), pct * 100),
    #                 device=x.device,
    #                 dtype=torch.float32)
    #             new_min = torch.tensor(np.percentile(
    #                 x.reshape(-1).cpu(), (1 - pct) * 100),
    #                 device=x.device,
    #                 dtype=torch.float32)

    #         x_q = self.quantize(x, new_max, new_min)
    #         score = self.lp_loss(x, x_q, p=2, reduction='all')
    #         if score < best_score:
    #             best_score = score
    #             best_new_max = new_max
    #             best_new_min = new_min

    #     if best_new_max is None or best_new_min is None:
    #         # 使用默认值
    #         best_new_max = x.max()
    #         best_new_min = x.min()

    #     delta = (best_new_max - best_new_min) / (self.n_levels - 1)
    #     zero_point = (-best_new_min / delta).round()
    #     return delta, zero_point


    # def quantize(self, x: torch.Tensor, new_max, new_min):
    #     delta = (new_max - new_min) / (self.n_levels - 1)
    #     zero_point = (-new_min / delta).round()
    #     x_int = torch.round(x / delta) + zero_point
    #     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
    #     x_float_q=(x_quant - zero_point) * delta
    #     return x_float_q
    

    def lp_loss(self, original: torch.Tensor, quantized: torch.Tensor, p=2, reduction='all'):
        if reduction == 'all':
            return torch.norm(original - quantized, p=p)
        else:
            return torch.norm(original - quantized, p=p, dim=reduction)

    def save_outlier_data(self):
        with open("outliers.txt", "w") as f:
            for i, (values, count) in enumerate(zip(self.outlier_values, self.outlier_counts)):
                f.write(f"Iteration {i + 1}:\n")
                f.write(f"Outlier count: {count}\n")
                f.write(f"Outlier values: {values}\n")
                f.write("\n")

    def get_group(self, value):
        differences = [abs(value - mean) for mean in self.means]
        return differences.index(min(differences))

class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " n_bits={},inited={}, channel_wise={})".format(self.n_bits,self.inited, self.channel_wise)
        return s

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None,None

        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError

        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)

                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        if self.sym:
            raise NotImplementedError
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans


    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)


    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1

        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):  # x_min[512]
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
 
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)

            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:

            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()

        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max


    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.sym:
            best_min, best_max = self.perform_1D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:

            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'w_bit={}, wq_is_training={}, w_inited={},w_channel_wise={}'.format(
            self.n_bits, self.is_training, self.inited,self.channel_wise
        )
    

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()