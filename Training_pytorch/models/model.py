import os.path

import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d, QLinear
import torch
import csv
from utee import misc

print = misc.logger.info


class MODEL(nn.Module):
    def __init__(self, args, dimensions, features, classifier, num_classes, logger):
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


# Todo: Develop same for linear layers, rename in_dimension
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
                         t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC' + str(i) + '_'),

        if i == len(classifiers) - 1:
            layers += [linear]
        else:
            layers += [linear, activation]

        in_size = classifiers[i][1]

    return nn.Sequential(*layers)


# Todo: Use more semantic notation, make linear layers work, add Resnet
cfg_list = {
    'speed': {
        'features': [('C', 128, 3, 'same', 32),
                     ('M', 2, 2),
                     ('C', 256, 3, 'same', 16),
                     ('M', 2, 2),
                     ('C', 512, 3, 'same', 8),
                     ('M', 2, 2)],
        'classifier': [('L', 1024, 1, 'same', 1),
                       ('L', 10, 1, 'same', 1)]
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
                       ('L', 10, 1, 'same', 1)]
    },
    'vgg16': {
        'features': [('C', 64, 3, 'same', 32),
                     ('C', 64, 3, 'same', 32),
                     ('M', 2, 2),
                     ('C', 128, 3, 'same', 16),
                     ('C', 128, 3, 'same', 16),
                     ('M', 2, 2),
                     ('C', 256, 3, 'same', 16),
                     ('C', 256, 3, 'same', 16),
                     ('C', 256, 3, 'same', 16),
                     ('M', 2, 2),
                     ('C', 512, 3, 'same', 8),
                     ('C', 512, 3, 'same', 8),
                     ('C', 512, 3, 'same', 8),
                     ('M', 2, 2),
                     ('C', 512, 3, 'same', 4),
                     ('C', 512, 3, 'same', 4),
                     ('C', 512, 3, 'same', 4),
                     ('M', 2, 2)],
        'classifier': [('L', 1024, 1, 'same', 1),
                       ('L', 10, 1, 'same', 1)]
    }
}


def cifar10(args, logger, pretrained=None):
    features = cfg_list[args.network]["features"]
    classifiers = cfg_list[args.network]["classifier"]
    build_csv(features, classifiers, 8192, 3)
    features = make_features(features, args, logger, 1)
    classifiers = make_classifiers(classifiers, args, logger, 8192)
    model = MODEL(args, 8192, features, classifiers, num_classes=10, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def cifar100(args, logger, pretrained=None):
    features = cfg_list[args.network]["features"]
    classifiers = cfg_list[args.network]["classifier"]
    build_csv(features, classifiers, 8192, 3)
    features = make_features(features, args, logger, 1)
    classifiers = make_classifiers(classifiers, args, logger, 8192)
    model = MODEL(args, 8192, features, classifiers, num_classes=100, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def mnist(args, logger, pretrained=None):
    features = cfg_list[args.network]["features"]
    classifiers = cfg_list[args.network]["classifier"]
    build_csv(features, classifiers, 4608, 1)
    features = make_features(features, args, logger, 1)
    classifiers = make_classifiers(classifiers, args, logger, 4608)
    model = MODEL(args, 4608, features, classifiers, num_classes=10, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model
