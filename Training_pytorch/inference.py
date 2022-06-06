import argparse
import os
import time
# Todo: epxlain
from utee import misc
# Import Pytorch & Numpy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# Todo: explain
from utee import make_path
from utee import hook
# Import Cifar dataset
# Todo: Import other datasets
from cifar import dataset
from cifar import model
#from IPython import embed
from datetime import datetime
from subprocess import call

# Parsing training & architecture parameters
# Todo: Explain each parameter
# Todo: Remove unneccessary ones
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', default=2)
parser.add_argument('--wl_grad', default=8)
parser.add_argument('--wl_activate', default=8)
parser.add_argument('--wl_error', default=8)
parser.add_argument('--inference', default=1)
parser.add_argument('--onoffratio', default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
parser.add_argument('--cellBit', default=4, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--subArray', default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--ADCprecision', default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--vari', default=0, help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', default=0, help='retention time')
parser.add_argument('--v', default=0, help='drift coefficient')
parser.add_argument('--detect', default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', default=0, help='drift target for fixed-direction drift')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
# Set to run inference simulation
args.inference = 1            
# Hardware Properties
# If you do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
args.vari = 0                 
args.t = 0                     
args.v = 0                   
args.detect = 1               
args.target = 0.5             

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

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

# Todo: Check if model saving works
model_path = './log/default/batch_size=200/decreasing_lr=200,250/grad_scale=8/seed=117/type=cifar10/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/latest.pth'

# data loader and model
# Todo: Add option to choose different datasets
assert args.type in ['cifar10', 'cifar100'], args.type
train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
modelCF = model.cifar10(args = args, logger=logger, pretrained = model_path)
print(args.cuda)
if args.cuda:
	modelCF.cuda()
best_acc, old_file = 0, None
t_begin = time.time()
# ready to go
modelCF.eval()
test_loss = 0
correct = 0
trained_with_quantization = True

# for data, target in test_loader:
for i, (data, target) in enumerate(test_loader):
	if i==0:
		hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,0)
	indx_target = target.clone()
	if args.cuda:
		data, target = data.cuda(), target.cuda()
	with torch.no_grad():
		data, target = Variable(data), Variable(target)
		output = modelCF(data)
		test_loss += F.cross_entropy(output, target).data
		pred = output.data.max(1)[1]  # get the index of the max log-probability
		correct += pred.cpu().eq(indx_target).sum()
	if i==0:
		hook.remove_hook_list(hook_handle_list)

test_loss = test_loss / len(test_loader)  # average over number of mini-batch
acc = 100. * correct / len(test_loader.dataset)

accuracy = acc.cpu().data.numpy()

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

# Running C++ files for layer estimation
# Todo: Extract parameters
call(["/bin/bash", "./layer_record/trace_command.sh"])

finish_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
