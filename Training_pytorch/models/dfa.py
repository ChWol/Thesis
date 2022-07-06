import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from modules.quantization_cpu_np_infer import QConv2d, QLinear


class DFANet(torch.nn.Module):
    def __init__(self, args, logger):
        super(DFANet, self).__init__()

        activation = 'tanh'
        activation_function = nn.Tanh()

        self.linear1 = QLinear(784, 512, logger=logger,
                               wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                               wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                               cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                               vari=args.vari,
                               t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '1' + '_',
                               rule='dfa', activation=activation)
        self.relu1 = activation_function
        self.linear2 = QLinear(512, 1024, logger=logger,
                               wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                               wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                               cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                               vari=args.vari,
                               t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '2' + '_',
                               rule='dfa', activation=activation)
        self.relu2 = activation_function
        self.linear3 = QLinear(1024, 10, logger=logger,
                               wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                               wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                               cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision,
                               vari=args.vari,
                               t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + '3' + '_',
                               rule='dfa', activation='none')
        self.layers = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.linear1.input = x
        x = self.linear1(x)
        self.linear1.output = x
        x = self.relu1(x)
        self.linear2.input = x
        x = self.linear2(x)
        self.linear2.output = x
        x = self.relu2(x)
        self.linear3.input = x
        x = self.linear3(x)
        self.linear3.output = x
        return x

    def dfa(self, error):
        for layer in self.layers:
            B = layer.dfa_matrix.cuda()
            a = torch.transpose(layer.output, 0, 1).cuda()
            e = torch.transpose(error, 0, 1).cuda()
            y = layer.input.cuda()
            if layer.activation == 'relu':
                a = torch.where(a > 0, 1, 0)
            if layer.activation == 'tanh':
                tanh = nn.Tanh()
                a = torch.ones_like(a) - torch.square(tanh(a))
            if layer.activation == 'simgoid':
                sigmoid = nn.Sigmoid()
                a = torch.matmul(sigmoid(a), torch.ones_like(a) - sigmoid(a))
            else:
                a = torch.ones_like(a)

            print(layer.name)
            print(layer.weight)
            print(layer.weight.grad)
            layer.weight.grad += torch.matmul(torch.matmul(B, e) * a, y)
