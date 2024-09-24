from numpy import isin
import torch
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import torch.nn.functional as F
from tqdm import tqdm


class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """

    def __init__(self, net, wrapped_modules,cali_data, cali_sentence,cali_embedding, cali_l_mask, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.cali_data = cali_data
        self.cali_embedding = cali_embedding
        self.cali_l_mask = cali_l_mask
        self.cali_sentence=cali_sentence
        self.sequential = sequential
        self.calibrated = False


    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        n_calibration_steps=2

        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'

            with torch.no_grad():
                image=self.cali_data.cuda()
                l_mask=self.cali_l_mask.cuda()
                if hasattr(self.net, "text_encoder"):
                    sentences = self.cali_sentence.cuda() 
                    self.net(image, sentences, l_mask=l_mask)
                else:
                    embedding=self.cali_embedding.cuda()
                    self.net(image, embedding, l_mask=l_mask)

        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")

    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")

        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'

        with torch.no_grad():
            image=self.cali_data.cuda()
            l_mask=self.cali_l_mask.cuda()
            if hasattr(self.net, "text_encoder"):
                sentences = self.cali_sentence.cuda() 
                self.net(image, sentences, l_mask=l_mask)
            else:
                embedding=self.cali_embedding.cuda()
                self.net(image, embedding, l_mask=l_mask)

        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"

                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()

        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")

    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        q = tqdm(self.wrapped_modules.items(), desc="Brecq")

        for name, module in q:
            q.set_postfix_str(name)

            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))

            image=self.cali_data.cuda()
            embedding=self.cali_embedding.cuda()
            sentences = self.cali_sentence.cuda() 
            l_mask=self.cali_l_mask.cuda()
            for batch_st in range(0, self.cali_data.size(0), self.batch_size):
                self.net.zero_grad()
                if not hasattr(self.net, "text_encoder"):
                    image_batch = image[batch_st:batch_st+self.batch_size].cuda()
                    embedding_batch = embedding[batch_st:batch_st+self.batch_size].cuda()
                    l_mask_batch = l_mask[batch_st:batch_st+self.batch_size].cuda()
                    self.net(image_batch, embedding_batch, l_mask=l_mask_batch)
                else:
                    image_batch = image[batch_st:batch_st+self.batch_size].cuda()
                    sentence_batch=sentences[batch_st:batch_st+self.batch_size].cuda()
                    l_mask_batch = l_mask[batch_st:batch_st+self.batch_size].cuda()
                    self.net(image_batch, sentence_batch, l_mask=l_mask_batch)
                del image,embedding,sentences,l_mask
                torch.cuda.empty_cache()

            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)

            for hook in hooks:
                hook.remove()

            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()

            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("calibration finished")

def grad_hook(module, grad_input, grad_output):

    if module.raw_grad is None:
        module.raw_grad = []

    module.raw_grad.append(grad_output[0].cpu().detach())


def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []

    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())

class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, cali_data, cali_sentence,cali_embedding, cali_l_mask, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, cali_data, cali_sentence,cali_embedding, cali_l_mask, sequential=sequential)
        self.batch_size = batch_size

    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        with torch.no_grad():
            if not hasattr(self.net, "text_encoder"):
                raw_pred = self.net(self.cali_data.cuda(),self.cali_embedding.cuda(),self.cali_l_mask.cuda())
            else:
                raw_pred = self.net(self.cali_data.cuda(),self.cali_sentence.cuda(),self.cali_l_mask.cuda())
            raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()

            torch.cuda.empty_cache()

        q = tqdm(self.wrapped_modules.items(), desc="Brecq")

        for name, module in q:
            q.set_postfix_str(name)
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))

            image=self.cali_data.cuda()
            embedding=self.cali_embedding.cuda()
            sentences = self.cali_sentence.cuda() 
            l_mask=self.cali_l_mask.cuda()

            for batch_st in range(0,self.cali_data.size(0),self.batch_size):

                self.net.zero_grad()
                if not hasattr(self.net, "text_encoder"):

                    image_batch = image[batch_st:batch_st+self.batch_size].cuda()
                    embedding_batch = embedding[batch_st:batch_st+self.batch_size].cuda()
                    l_mask_batch = l_mask[batch_st:batch_st+self.batch_size].cuda()

                    pred = self.net(image_batch, embedding_batch, l_mask=l_mask_batch)
                else:
                    image_batch = image[batch_st:batch_st+self.batch_size].cuda()
                    sentence_batch = sentences[batch_st:batch_st+self.batch_size].cuda()
                    l_mask_batch = l_mask[batch_st:batch_st+self.batch_size].cuda()

                    pred = self.net(image_batch, sentence_batch, l_mask=l_mask_batch)

                loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")

                loss.backward()
                del image,embedding,sentences,l_mask, pred, loss
                torch.cuda.empty_cache()

            if isinstance(module, MinMaxQuantLinear):

                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)

            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = torch.cat(module.raw_grad, dim=0)

            for hook in hooks:
                hook.remove()

            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                torch.cuda.empty_cache()

            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")

    def batching_quant_calib(self):

        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        with torch.no_grad():
            image=self.cali_data.cuda()
            embedding=self.cali_embedding.cuda()
            l_mask=self.cali_l_mask.cuda()
            sentences = self.cali_sentence.cuda()
            raw_pred = self.net(image,sentences,l_mask)
            raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()

            torch.cuda.empty_cache()
        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            q.set_postfix_str(name)

            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))

            image=self.cali_data.cuda()
            embedding=self.cali_embedding.cuda()
            l_mask=self.cali_l_mask.cuda()
            sentences = self.cali_sentence.cuda()
            for batch_st in range(0,self.cali_data.size(0),self.batch_size):
                self.net.zero_grad()
                image_batch = image[batch_st:batch_st+self.batch_size].cuda()
                sentence_batch = sentences[batch_st:batch_st+self.batch_size].cuda()
                l_mask_batch = l_mask[batch_st:batch_st+self.batch_size].cuda()
                pred = self.net(image_batch, sentence_batch, l_mask=l_mask_batch)
                loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                loss.backward(retain_graph=True)

            del image,embedding,sentences,l_mask, pred,loss  
            torch.cuda.empty_cache()

            if isinstance(module, MinMaxQuantLinear):

                module.raw_input = torch.cat(module.raw_input, dim=0)   
                module.raw_out = torch.cat(module.raw_out, dim=0)       # 
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0) 

            for hook in hooks:
                hook.remove()

            with torch.no_grad():

                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()

            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")


