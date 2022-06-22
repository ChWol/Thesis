import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer, wage_quantizer
import numpy as np


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, logger=None, clip_weight=False, wage_init=False,
                 quantize_weight=False, clip_output=False, quantize_output=False,
                 wl_input=8, wl_activate=8, wl_error=8, wl_weight=8, inference=0, onoffratio=10, cellBit=1,
                 subArray=128, ADCprecision=5, vari=0, t=0, v=0, detect=0, target=0, debug=0, name='Qconv'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
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


    def forward(self, input):
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        outputOrignal = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            input = wage_quantizer.Q(input, self.wl_input)
            output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output / self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)

        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, logger=None, clip_weight=False, wage_init=False,
                 quantize_weight=False, clip_output=False, quantize_output=False,
                 wl_input=8, wl_activate=8, wl_error=8, wl_weight=8, inference=0, onoffratio=10, cellBit=1,
                 subArray=128, ADCprecision=5, vari=0, t=0, v=0, detect=0, target=0, debug=0, name='Qlinear'):
        super(QLinear, self).__init__(in_features, out_features, bias)
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


    def forward(self, input):

        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            input = wage_quantizer.Q(input, self.wl_input)
            output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            output = F.linear(input, weight, self.bias)

        output = output / self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)

        return output
