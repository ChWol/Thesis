import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from modules.quantization_cpu_np_infer import QConv2d, QLinear

# Implement FA first?
# Activation functions?
# Quantization routine as before? Connect to hooks

# Constructing net as before but with added matrices?
# Instead of x = ... use add_module? Reshaping for linear? How for activations?
# How to initialize weights
# How to Convolutional Layer?
# Error passed by multiplication?
# How to make use of this backward instead of wage?
# Parallelizable?

# Automatically create list of random feedback weights upon initialization
# Create dfanet.backward(logit_error) using initialized random feedback weights to project error,
# use 'with torch.no_grad()' in backward operation


class DFANet(torch.nn.Module):
    def __init__(self, args, logger):
        super(DFANet, self).__init__()

        self.linear1 = QLinear(784, 512, logger=logger,
                              wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                              wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                              cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                              vari=args.vari,
                              t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '1' + '_', rule='dfa')
        self.linear2 = QLinear(512, 1024, logger=logger,
                              wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                              wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                              cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                              vari=args.vari,
                              t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '2' + '_', rule='dfa')
        self.linear3 =QLinear(1024, 10, logger=logger,
                              wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                              wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                              cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                              vari=args.vari,
                              t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '3' + '_', rule='dfa')
        self.layers = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.linear1.input = x
        x = self.linear1(x)
        self.linear1.output = x
        self.linear2.input = x
        x = self.linear2(x)
        self.linear2.output = x
        self.linear3.input = x
        x = self.linear3(x)
        self.linear3.output = x
        return x

    def dfa(self, error):
        for layer in self.layers:
            B = layer.dfa_matrix
            e = torch.transpose(error, 0, 1)
            a = torch.transpose(layer.output, 0, 1)
            i = layer.input

            B = B.cuda()
            e = e.cuda()
            a = a.cuda()
            i = i.cuda()

            print(layer)
            print(self.named_parameters())

            print("Size dfa matrix: {}".format(B.size()))
            print("Size error: {}".format(e.size()))
            print("Size output: {}".format(a.size()))
            print("Size input: {}".format(i.size()))
            layer.grad = torch.matmul(torch.matmul(B, e) * a, i)
            print("Size of grad: {}".format(layer.grad.size()))


class LinearFANetwork(nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list):
        super(LinearFANetwork, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list

        self.linear = [LinearFAModule(self.in_features, self.num_hidden_list[0])]
        for idx in range(self.num_layers - 1):
            self.linear.append(LinearFAModule(self.num_hidden_list[idx], self.num_hidden_list[idx + 1]))

        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        linear1 = self.linear[0](inputs)
        linear2 = self.linear[1](linear1)
        return linear2


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
    def backward(context, *grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFAModule(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(LinearFAModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight_fa = Variable(torch.FloatTensor(output_features, input_features), requires_grad=False)
        torch.nn.init.kaiming_uniform(self.weight)
        torch.nn.init.kaiming_uniform(self.weight_fa)
        torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
