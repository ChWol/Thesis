import os.path

import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d, QLinear
import torch
import csv
from utee import misc

print = misc.logger.info


class MODEL(nn.Module):
    def __init__(self, features, classifier):
        super(MODEL, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)

        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, QLinear):
                layer.input = x
                x = layer(x)
                layer.output = x
            else:
                x = layer(x)
        return x

    def direct_feedback_alignment(self, error):
        for i, layer in enumerate(self.classifier):
            if not isinstance(layer, QLinear):
                continue

            B = layer.dfa_matrix.cuda()
            a = torch.transpose(layer.output, 0, 1).cuda()
            e = torch.transpose(error, 0, 1).cuda()
            y = layer.input.cuda()

            if layer.activation == 'relu':
                a = torch.where(a > 0, 1, 0)
            elif layer.activation == 'tanh':
                tanh = nn.Tanh()
                a = torch.ones_like(a) - torch.square(tanh(a))
            elif layer.activation == 'sigmoid':
                sigmoid = nn.Sigmoid()
                a = torch.matmul(sigmoid(a), torch.ones_like(a) - sigmoid(a))
            else:
                a = torch.ones_like(a)

            if i == len(self.classifier)-1:
                layer.weight.grad = torch.matmul(e, y)
            else:
                layer.weight.grad = torch.matmul(torch.matmul(B, e) * a, y)


def build_csv(features, classifiers, linear_dimension, input_depth=3):
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, '../NeuroSIM/NetWork.csv')

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        ifm_depth = input_depth

        for i in range(len(features)):
            pooling = 0
            if features[i][0] == 'M':
                continue
            if features[i][0] == 'C':
                if features[i + 1][0] == 'M':
                    pooling = 1
                row = [features[i][4], features[i][4], ifm_depth, features[i][2], features[i][2], features[i][1],
                       pooling, 1]
            ifm_depth = features[i][1]
            writer.writerow(row)

        ifm_depth = linear_dimension

        for classifier in classifiers:
            row = [1, 1, ifm_depth, 1, 1, classifier[1], 0, 1]
            ifm_depth = classifier[1]
            writer.writerow(row)


def make_features(features, args, logger, in_dimension):
    layers = []
    in_channels = in_dimension
    for i, v in enumerate(features):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2] // 2
            else:
                padding = 0
            conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                             detect=args.detect, target=args.target,
                             name='Conv' + str(i) + '_')
            if args.relu:
                non_linearity_activation = nn.ReLU()
            else:
                non_linearity_activation = nn.Sigmoid()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
    return nn.Sequential(*layers)


def make_classifiers(classifiers, args, logger, in_dimension, num_classes):
    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'tanh':
        activation = nn.Tanh()
    elif args.activation == 'sigmoid':
        activation = nn.Sigmoid()

    layers = []
    in_size = in_dimension

    for i, classifier in enumerate(classifiers):
        if i == len(classifiers) - 1:
            wl_activate = -1
        else:
            wl_activate = args.wl_activate

        linear = QLinear(in_size, classifier[1], logger=logger,
                         wl_input=args.wl_activate, wl_activate=wl_activate, wl_error=args.wl_error,
                         wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                         cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari,
                         t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + str(i) + '_',
                         activation=args.activation, num_classes=num_classes, rule=args.rule)

        if i == len(classifiers) - 1:
            layers += [linear]
        else:
            layers += [linear, activation]

        in_size = classifier[1]
    return nn.Sequential(*layers)


def get_model(num_classes, network):
    networks = {
        'single': {
            'features': [],
            'classifier': [('L', num_classes, 1, 'same', 1)]
        },
        'double': {
            'features': [],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'triple': {
            'features': [],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', 1024, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'depth': {
            'features': [],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', 256, 1, 'same', 1),
                           ('L', 128, 1, 'same', 1),
                           ('L', 64, 1, 'same', 1),
                           ('L', 32, 1, 'same', 1),
                           ('L', 16, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'vgg8': {
            'features': [('C', 128, 3, 'same', 32),
                         ('C', 128, 3, 'same', 32),
                         ('M', 2, 2),
                         ('C', 256, 3, 'same', 16),
                         ('C', 256, 3, 'same', 16),
                         ('M', 2, 2),
                         ('C', 512, 3, 'same', 8),
                         ('C', 512, 3, 'same', 8),
                         ('M', 2, 2)],
            'classifier': [('L', 1024, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        }
    }
    return networks[network]


def cifar(args, logger, num_classes, pretrained=None):
    model = get_model(num_classes, args.network)
    features = model["features"]
    classifiers = model["classifier"]

    if len(features) == 0:
        input = 1024
    else:
        input = 8192

    build_csv(features, classifiers, input, 3)

    features = make_features(features, args, logger, 3)
    classifiers = make_classifiers(classifiers, args, logger, input, num_classes)

    model = MODEL(features, classifiers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def mnist(args, logger, num_classes, pretrained=None):
    model = get_model(num_classes, args.network)
    features = model["features"]
    classifiers = model["classifier"]

    print("features sollten leer sein")
    print(features)
    if len(features) == 0:
        input = 784
    else:
        input = 4096

    build_csv(features, classifiers, input, 1)

    features = make_features(features, args, logger, 1)
    classifiers = make_classifiers(classifiers, args, logger, input, num_classes)

    model = MODEL(features, classifiers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model