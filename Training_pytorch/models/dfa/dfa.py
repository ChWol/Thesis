import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

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
    def __init__(self):
        super(DFANet, self).__init__()
        layers = []

    def add_module(self, name, module):
        super().add_module(name, module)  # maintain the same base functionality
        if type(module) is torch.nn.Linear:
            ### code for implementing random error projection matrices
            self.dfa_matrices.append(torch.nn.Linear(self.logit_dim, module.out_features))

    def backward(self, error):
        # Iterates through the dfa_matrices list
        for layer in layers:
            # Use the randomly initialized weights to propagate error in parallel through the network
            layer.grad =
        # Make compatible with existing PyTorch optimizers: Store the result in the .grad attribute of all the layers

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(name)
            add feedback_layers
            self.feedback_layers.append(torch.nn.Linear(self.output_dim, value.out_features))

    # Todo: Activation functions
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#########################

# 2 different models: Forward = trained module, take the loss as input for backward
# Backward: Single layer of parallel matrices, calculate output
# Overwrite gradients of forward with output of backward


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
