import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer, wage_quantizer
import numpy as np
from torch import autograd


class LinearFAFunction(autograd.Function):

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class DFALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, logger=None, clip_weight=False, wage_init=False,
                 quantize_weight=False, clip_output=False, quantize_output=False,
                 wl_input=8, wl_activate=8, wl_error=8, wl_weight=8, inference=0, onoffratio=10, cellBit=1,
                 subArray=128, ADCprecision=5, vari=0, t=0, v=0, detect=0, target=0, debug=0, name='Qlinear', rule='bp',
                 activation=1):
        super(DFALinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.scale = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)
        self.activation = activation

        if rule == 'dfa':
            B = torch.empty(out_features, 10, requires_grad=False)
            nn.init.xavier_uniform_(B, gain=nn.init.calculate_gain('relu'))
            self.weight_fa = B


    def forward(self, input):

        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1:
            # retention
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1 / onoffratio
            output = torch.zeros_like(outputOrignal)
            cellRange = 2 ** self.cellBit  # cell precision is 4
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2
            # need to divide to different subArray
            numSubArray = int(weight.shape[1] / self.subArray)

            if numSubArray == 0:
                mask = torch.zeros_like(weight)
                mask[:, :] = 1
                # quantize input into binary sequence
                inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ - inputB) / 2)
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                    outputP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
                    for k in range(int(bitWeight / self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange) * mask
                        X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                        # Now also consider weight has on/off ratio effects
                        # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                        remainderQ = (upper - lower) * (remainder - 0) + (
                                    cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                        remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari,
                                                                                           device='cuda').float())
                        outputPartial = F.linear(inputB, remainderQ * mask, self.bias)
                        outputDummyPartial = F.linear(inputB, dummyP * mask, self.bias)
                        # Add ADC quanization effects here !!!
                        outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        scaler = cellRange ** k
                        outputP = outputP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                        outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                    scalerIN = 2 ** z
                    outputIN = outputIN + (outputP - outputD) * scalerIN
                output = output + outputIN / (2 ** bitActivation)
            else:
                inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ - inputB) / 2)
                    outputP = torch.zeros_like(outputOrignal)
                    for s in range(numSubArray):
                        mask = torch.zeros_like(weight)
                        mask[:, (s * self.subArray):(s + 1) * self.subArray] = 1
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                        outputSP = torch.zeros_like(outputOrignal)
                        outputD = torch.zeros_like(outputOrignal)
                        for k in range(int(bitWeight / self.cellBit)):
                            remainder = torch.fmod(X_decimal, cellRange) * mask
                            X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                            # Now also consider weight has on/off ratio effects
                            # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                            # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                            remainderQ = (upper - lower) * (remainder - 0) + (
                                        cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                            remainderQ = remainderQ + remainderQ * torch.normal(0.,
                                                                                torch.full(remainderQ.size(), self.vari,
                                                                                           device='cuda').float())
                            outputPartial = F.linear(inputB, remainderQ * mask, self.bias)
                            outputDummyPartial = F.linear(inputB, dummyP * mask, self.bias)
                            # Add ADC quanization effects here !!!
                            outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                            outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
                                                                                   self.ADCprecision)
                            scaler = cellRange ** k
                            outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                            outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2 ** z
                    outputIN = outputIN + outputP * scalerIN
                output = output + outputIN / (2 ** bitActivation)
            output = output / (2 ** bitWeight)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            output = LinearFAFunction.apply(input, weight, self.weight_fa, self.bias)

        output = output / self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)

        return output

