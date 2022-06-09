from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
import torch
import csv

class CIFAR(nn.Module):
    def __init__(self, args, dimensions, features, num_classes,logger):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            QLinear(dimensions, 1024, logger=logger,
                    wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC1_'),
            nn.ReLU(inplace=True),
            QLinear(1024, num_classes, logger=logger,
                    wl_input = args.wl_activate,wl_activate=-1, wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,name='FC2_'))

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def build_csv(layers, linear_dimension, input_dimension=32, input_depth=3):
    print('################ TESTING ################')
    ifm_dimension = input_dimension
    once = False
    with open('test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        ifm_depth = input_depth
        for i in range(len(layers)):
            pooling = 0
            if layers[i][0] == 'M':
                continue
            if layers[i][0] == 'C':
                if layers[i+1][0] == 'M':
                    pooling = 1
                row = [ifm_dimension, ifm_dimension, ifm_depth, layers[i][2], layers[i][2], layers[i][1], pooling, 1]
                ifm_dimension = ifm_dimension/2
            if layers[i][0] == 'L':
                if not once:
                    ifm_depth = linear_dimension
                    once = True
                row = [1, 1, ifm_depth, 1, 1, layers[i][1], 0, 1]
            print(row)
            ifm_depth = layers[i][1]
            writer.writerow(row)

# Todo: Same for linear layers
def make_layers(cfg, args, logger, in_dimension):
    layers = []
    in_channels = in_dimension
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                             logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                             wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name = 'Conv'+str(i)+'_')
            non_linearity_activation = nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
    return nn.Sequential(*layers)


# Todo: Use more semantic notation
cfg_list = {
    'cifar10': [('C', 128, 3, 'same'),
                ('M', 2, 2),
                ('C', 256, 3, 'same'),
                ('M', 2, 2),
                ('C', 512, 3, 'same'),
                ('M', 2, 2),
                ('L', 1024, 1, 'same'),
                ('L', 10, 1, 'same')],
    'alexnet':  [('C', 96, 11, 'same'),
                ('M', 3, 2),
                ('C', 256, 5, 'same'),
                ('M', 3, 2),
                ('C', 384, 3, 'same'),
                ('C', 384, 3, 'same'),
                ('C', 256, 3, 'same'),
                ('M', 3, 2),
                ('L', 1024, 1, 'same'),
                ('L', 10, 1, 'same')],
    'vgg8':     [('C', 128, 3, 'same'),
                ('C', 128, 3, 'same'),
                ('M', 2, 2),
                ('C', 256, 3, 'same'),
                ('C', 256, 3, 'same'),
                ('M', 2, 2),
                ('C', 512, 3, 'same'),
                ('C', 512, 3, 'same'),
                ('M', 2, 2),
                ('L', 1024, 1, 'same'),
                ('L', 10, 1, 'same')],
    'vgg16':    [('C', 64, 3, 'same'),
                ('C', 64, 3, 'same'),
                ('M', 2, 2),
                ('C', 128, 3, 'same'),
                ('C', 128, 3, 'same'),
                ('M', 2, 2),
                ('C', 256, 3, 'same'),
                ('C', 256, 3, 'same'),
                ('C', 256, 3, 'same'),
                ('M', 2, 2),
                ('C', 512, 3, 'same'),
                ('C', 512, 3, 'same'),
                ('C', 512, 3, 'same'),
                ('M', 2, 2),
                ('C', 512, 3, 'same'),
                ('C', 512, 3, 'same'),
                ('C', 512, 3, 'same'),
                ('M', 2, 2),
                ('L', 1024, 1, 'same'),
                ('L', 10, 1, 'same')]
}

# Todo: Merge to one method
def cifar10( args, logger, pretrained=None):
    cfg = cfg_list['cifar10']
    build_csv(cfg, 32, 3, 8192)
    layers = make_layers(cfg, args,logger, 3)
    model = CIFAR(args,8192,layers, num_classes=10,logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

def cifar100( args, logger, pretrained=None):
    cfg = cfg_list['cifar10']
    build_csv(cfg, 32, 3)
    layers = make_layers(cfg, args,logger, 3)
    model = CIFAR(args,8192,layers, num_classes=100,logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

def mnist( args, logger, pretrained=None):
    cfg = cfg_list['cifar10']
    build_csv(cfg, 32, 1, 4608)
    layers = make_layers(cfg, args,logger, 1)
    model = CIFAR(args,4608,layers, num_classes=10,logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


