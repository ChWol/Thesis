import argparse
import os
import time

from utee import wage_util
from utee import misc
import torch
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from utee import make_path
from utee import hook
from models import dataset
from models import models
from datetime import datetime
from subprocess import call
import wandb


parser = argparse.ArgumentParser(description='Evaluation of Biologically-Plausible Learning Rules on Neuromorphic '
                                             'Hardware Architectures')
parser.add_argument('--dataset', default='fashion', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--grad_scale', type=float, default=1, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed')
parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training '
                                                                  'status')
parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
parser.add_argument('--onoffratio', type=int, default=10)
parser.add_argument('--cellBit', type=int, default=6, help='cell precision (cellBit==wl_weight==wl_grad)')
parser.add_argument('--wl_weight', type=int, default=6)
parser.add_argument('--wl_grad', type=int, default=6)
parser.add_argument('--inference', type=int, default=0)
parser.add_argument('--subArray', type=int, default=128)
parser.add_argument('--ADCprecision', type=int, default=5)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)
parser.add_argument('--nonlinearityLTP', type=float, default=1.75, help='nonlinearity in LTP')
parser.add_argument('--nonlinearityLTD', type=float, default=1.46, help='nonlinearity in LTD (negative if LTP and LTD '
                                                                        'are asymmetric)')
parser.add_argument('--d2dVari', type=float, default=0, help='device-to-device variation')
parser.add_argument('--c2cVari', type=float, default=0.003, help='cycle-to-cycle variation')
parser.add_argument('--network', type=int, default=5)
parser.add_argument('--hidden', type=int, default=1024)
parser.add_argument('--technode', type=int, default=22)
parser.add_argument('--memcelltype', type=int, default=3)
parser.add_argument('--activation', default='relu')
parser.add_argument('--rule', default='bp')
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--neurosim', type=int, default=1)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--scheduler', type=int, default=0)
parser.add_argument('--initial', default='xavier')
parser.add_argument('--gradient_analysis', type=int, default=0)
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.wl_weight = args.wl_grad = args.cellBit
args.max_level = 2 ** args.cellBit
technode_to_width = {7: 14, 10: 14, 14: 22, 22: 32, 32: 40, 45: 50, 65: 100, 90: 200, 130: 200}
args.wireWidth = technode_to_width[args.technode]

if args.memcelltype == 1:
    args.cellBit = 1

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args, ['log_interval', 'test_interval', 'logdir', 'epochs', 'onoffratio', 'subArray', 'ADCprecision', 'neurosim', 'memcelltype'])

wandb.init(project=args.dataset.upper() + "-Inference", config=args, entity='duke-tum')
wandb.run.name = "{} ({}): {}".format(args.network, args.rule, wandb.run.id)

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('====================')
print('Path', args.logdir)
model_path = (args.logdir + '/latest.pth')

args.inference = 1

# models loader and model
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=10, pretrained=model_path)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=100, pretrained=model_path)
elif args.dataset == 'mnist':
    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
    model = models.mnist(args=args, logger=logger, num_classes=10, pretrained=model_path)
elif args.dataset == 'fashion':
    train_loader, test_loader = dataset.get_fashion(batch_size=args.batch_size, num_workers=1)
    model = models.mnist(args=args, logger=logger, num_classes=10, pretrained=model_path)
else:
    raise ValueError("Unknown dataset type")

if args.cuda:
    model.cuda()

best_acc, old_file = 0, None
t_begin = time.time()
model.eval()
test_loss = 0
correct = 0
trained_with_quantization = True

# for models, target in test_loader:
for i, (data, target) in enumerate(test_loader):
    if i == 0:
        if args.activation == 'sigmoid':
            relu = 0
        else:
            relu = 1
        if args.memcelltype == 1:
            cellBit, wl_weight = 1, args.wl_weight
        else:
            cellBit, wl_weight = args.cellBit, args.wl_weight
        hook_handle_list = hook.hardware_evaluation(model, wl_weight, args.wl_activate,
                                                    0, args.batch_size, args.cellBit, args.technode,
                                                    args.wireWidth, relu, args.memcelltype,
                                                    2 ** args.ADCprecision,
                                                    args.onoffratio, args.rule, args.inference)
    indx_target = target.clone()
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss_i = (0.5 * (wage_util.SSE(output, target) ** 2)).sum()
        test_loss += test_loss_i.data
        pred = output.data.max(1)[1]
        correct += pred.cpu().eq(indx_target).sum()
    if i == 0:
        hook.remove_hook_list(hook_handle_list)

test_loss = test_loss / len(test_loader)
acc = 100. * correct / len(test_loader.dataset)
wandb.log({'Test Accuracy': acc/100, 'Test Loss': test_loss})

print(" --- Hardware Properties --- ")
print("subArray size: ")
print(args.subArray)
print("ADC precision: ")
print(args.ADCprecision)
print("cell precision: ")
print(args.cellBit)
print("on/off ratio: ")
print(args.onoffratio)
print("variation: ")
print(args.vari)

logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset), acc))

accuracy = acc.cpu().data.numpy()

# Running C++ files for layer estimation
call(["/bin/bash", "./layer_record/trace_command.sh"])
log_input = {}
layer_out = pd.read_csv("Layer.csv").to_dict()
for key, value in layer_out.items():
    for layer, result in value.items():
        log_input["Layer {}: {}".format(layer + 1, key)] = result
wandb.log(log_input)
summary_out = pd.read_csv("Summary.csv").to_dict()
log_input = {}
for key, value in summary_out.items():
    log_input[key] = value[0]
wandb.log(log_input)

finish_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
