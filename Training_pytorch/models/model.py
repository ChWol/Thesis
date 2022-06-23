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

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


def make_classifiers(classifiers, args, logger, in_dimension):
    if args.relu == 1:
        activation = nn.ReLU(inplace=True)
    else:
        activation = nn.Sigmoid()

    layers = []
    in_size = in_dimension

    for i in range(len(classifiers)):
        if i == len(classifiers) - 1:
            wl_activate = -1
        else:
            wl_activate = args.wl_activate

        linear = QLinear(in_size, classifiers[i][1], logger=logger,
                         wl_input=args.wl_activate, wl_activate=wl_activate, wl_error=args.wl_error,
                         wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                         cellBit=args.cellBit, subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari,
                         t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + str(i) + '_')

        if i == len(classifiers) - 1:
            layers += [linear]
        else:
            layers += [linear, activation]

        in_size = classifiers[i][1]
    print(layers)
    return nn.Sequential(*layers)


def get_model(num_classes, network):
    networks = {
        'short': {
            'features': [('C', 16, 3, 'same', 32),
                         ('M', 2, 2),
                         ('C', 32, 3, 'same', 16),
                         ('M', 2, 2)],
            'classifier': [('L', 256, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'speed': {
            'features': [('C', 16, 3, 'same', 32),
                         ('M', 2, 2),
                         ('C', 32, 3, 'same', 16),
                         ('M', 2, 2),
                         ('C', 64, 3, 'same', 8),
                         ('M', 2, 2)],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'vgg8': {
            'features': [('C', 16, 3, 'same', 32),
                         ('C', 16, 3, 'same', 32),
                         ('M', 2, 2),
                         ('C', 32, 3, 'same', 16),
                         ('C', 32, 3, 'same', 16),
                         ('M', 2, 2),
                         ('C', 64, 3, 'same', 8),
                         ('C', 64, 3, 'same', 8),
                         ('M', 2, 2)],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        },
        'old': {
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
        },
        'vgg11': {
            'features': [('C', 64, 3, 'same', 32),
                         ('M', 2, 2),
                         ('C', 128, 3, 'same', 16),
                         ('C', 128, 3, 'same', 16),
                         ('M', 2, 2),
                         ('C', 256, 3, 'same', 8),
                         ('C', 256, 3, 'same', 8),
                         ('M', 2, 2),
                         ('C', 512, 3, 'same', 4),
                         ('C', 512, 3, 'same', 4),
                         ('M', 2, 2),
                         ('C', 512, 3, 'same', 2),
                         ('C', 512, 3, 'same', 2),
                         ('M', 2, 2)],
            'classifier': [('L', 512, 1, 'same', 1),
                           ('L', 512, 1, 'same', 1),
                           ('L', num_classes, 1, 'same', 1)]
        }
    }
    return networks[network]


def cifar(args, logger, num_classes, pretrained=None):
    model = get_model(num_classes, args.network)
    features = model["features"]
    classifiers = model["classifier"]

    build_csv(features, classifiers, 2048, 3)

    features = make_features(features, args, logger, 3)
    classifiers = make_classifiers(classifiers, args, logger, 2048)

    model = MODEL(features, classifiers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def mnist(args, logger, pretrained=None):
    model = get_model(10, args.network)
    features = model["features"]
    classifiers = model["classifier"]

    build_csv(features, classifiers, 576, 1)

    features = make_features(features, args, logger, 1)
    classifiers = make_classifiers(classifiers, args, logger, 576)

    model = MODEL(features, classifiers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model
