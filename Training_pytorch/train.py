import argparse
import os
import time
from utee import misc
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from utee import make_path
from utee import wage_util
from utee import wage_quantizer
from utee import hook
from models import dataset
from models import models
from datetime import datetime
from subprocess import call
from models import dfa
import wandb
from decimal import Decimal

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='simple', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train')
parser.add_argument('--grad_scale', type=float, default=1, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed')
parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=6, help='weight precision')
parser.add_argument('--wl_grad', type=int, default=6, help='gradient precision')
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
parser.add_argument('--onoffratio', type=int, default=10)
parser.add_argument('--cellBit', type=int, default=6, help='cell precision (cellBit==wl_weight==wl_grad)')
parser.add_argument('--inference', type=int, default=0)
parser.add_argument('--subArray', type=int, default=128)
parser.add_argument('--ADCprecision', type=int, default=5)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)
parser.add_argument('--nonlinearityLTP', type=float, default=1.75, help='nonlinearity in LTP')
parser.add_argument('--nonlinearityLTD', type=float, default=1.46, help='nonlinearity in LTD (negative if LTP and LTD are asymmetric)')
parser.add_argument('--max_level', type=int, default=32, help='Maximum number of conductance states during weight update (floor(log2(max_level))=cellBit)')
parser.add_argument('--d2dVari', type=float, default=0, help='device-to-device variation')
parser.add_argument('--c2cVari', type=float, default=0.003, help='cycle-to-cycle variation')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--network', default='simple')
parser.add_argument('--technode', type=int, default='32')
parser.add_argument('--memcelltype', type=int, default=3)
parser.add_argument('--relu', type=int, default=1)
parser.add_argument('--rule', default='dfa')

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
args = parser.parse_args()
args.max_level = 2 ** args.cellBit
if args.memcelltype == 1:
    args.cellBit = 1
args.wl_weight = args.cellBit
args.wl_grad = args.cellBit
technode_to_width = {7: 14, 10: 14, 14: 22, 22: 32, 32: 40, 45: 50, 65: 100, 90: 200, 130: 200}
args.wireWidth = technode_to_width[args.technode]
gamma = args.momentum
alpha = 1 - args.momentum

wandb.init(project=args.type.upper(), config=args, entity='duke-tum')
wandb.run.name = args.network + ": " + wandb.run.id

delta_distribution = open("delta_dist.csv", 'ab')
delta_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean", "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(delta_distribution, delta_firstline, delimiter=",", fmt='%s')
weight_distribution = open("weight_dist.csv", 'ab')
weight_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean", "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(weight_distribution, weight_firstline, delimiter=",", fmt='%s')
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args, ['log_interval', 'test_interval', 'logdir', 'epochs'])
misc.logger.init(args.logdir, 'train_log_' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if args.type == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=10)
if args.type == 'cifar100':
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=100)
if args.type == 'mnist':
    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
    model = models.mnist(args=args, logger=logger, input=4608)
if args.type == 'simple':
    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
    if args.rule == 'dfa':
        model = dfa.DFANet(args, logger)
    else:
        model = models.mnist(args=args, logger=logger, input=784)


#model.load_state_dict(torch.load(os.path.abspath(os.path.expanduser(os.path.join(args.logdir, 'best-6.pth')))))
# /home/chwolters/Thesis/Training_pytorch/log/default/ADCprecision=5/batch_size=200/c2cVari=0.003/cellBit=6/d2dVari=0/decreasing_lr=200,250/detect=0/grad_scale=1/inference=0/max_level=64/memcelltype=3/momentum=0.9/network=speed/nonlinearityLTD=1.46/nonlinearityLTP=1.75/onoffratio=10/relu=1/seed=117/subArray=32/t=0/target=0/technode=7/type=cifar10/v=0/vari=0/wireWidth=14/wl_activate=8/wl_error=8/wl_grad=6/wl_weight=6/best-4.pth
# /home/chwolters/Thesis/Training_pytorch/log/default/ADCprecision=5/batch_size=200/c2cVari=0.003/cellBit=6/d2dVari=0/decreasing_lr=200,250/detect=0/grad_scale=1/inference=0/max_level=64/memcelltype=3/momentum=0.9/network=speed/nonlinearityLTD=1.46/nonlinearityLTP=1.75/onoffratio=10/relu=1/seed=117/subArray=32/t=0/target=0/technode=7/type=cifar10/v=0/vari=0/wireWidth=14/wl_activate=8/wl_error=8/wl_grad=6/wl_weight=6/best-{4}.pth'
# Todo: From 1.3
'''
assert args.type in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.type == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
elif args.type == 'cifar100':
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
elif args.type == 'imagenet':
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")

assert args.network in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.network == 'VGG8':
    from models import VGG
    model = VGG.vgg8(args = args, logger=logger)
elif args.network == 'DenseNet40':
    from models import DenseNet
    model = DenseNet.densenet40(args = args, logger=logger)
elif args.network == 'ResNet18':
    from models import ResNet
    model = ResNet.resnet18(args = args, logger=logger)
else:
    raise ValueError("Unknown model type")
'''

if args.cuda:
    model.cuda()

#Todo: Add momentum and weight decay
# torch.optim.SGD(model_fa.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
accumulated_time = 0
t_begin = time.time()
grad_scale = args.grad_scale

try:
    # ready to go
    if args.cellBit != args.wl_weight:
        print("Warning: Weight precision should be the same as the cell precison !")
    # add d2dVari
    paramALTP = {}
    paramALTD = {}
    k = 0
    for layer in list(model.parameters())[::-1]:
        d2dVariation = torch.normal(torch.zeros_like(layer), args.d2dVari * torch.ones_like(layer))
        NL_LTP = torch.ones_like(layer) * args.nonlinearityLTP + d2dVariation
        NL_LTD = torch.ones_like(layer) * args.nonlinearityLTD + d2dVariation
        paramALTP[k] = wage_quantizer.GetParamA(NL_LTP.cpu().numpy()) * args.max_level
        paramALTD[k] = wage_quantizer.GetParamA(NL_LTD.cpu().numpy()) * args.max_level
        k = k + 1

    for epoch in range(args.epochs):
        split_time = time.time()
        model.train()

        velocity = {}
        i = 0
        for layer in list(model.parameters())[::-1]:
            velocity[i] = torch.zeros_like(layer)
            i = i + 1

        if epoch in decreasing_lr:
            grad_scale = grad_scale / 8.0

        logger("training phase")
        wandb.watch(model, log="all", log_freq=10)
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            output = model(data)
            error = wage_util.SSE(output, target)
            loss = 0.5 * (error ** 2)
            loss = loss.sum()

            if args.rule == 'dfa':
                model.dfa(error, epoch)
            else:
                loss.backward()

            # introduce non-ideal property
            j = 0
            #for name, param in list(model.named_parameters())[::-1]:
                #velocity[j] = gamma * velocity[j] + alpha * param.grad.data
                #param.grad.data = velocity[j]
                #param.grad.data = wage_quantizer.QG(param.data, args.wl_weight, param.grad.data, args.wl_grad,
                 #                                   grad_scale,
                  #                                  torch.from_numpy(paramALTP[j]).cuda(),
                   #                                 torch.from_numpy(paramALTD[j]).cuda(), args.max_level,
                    #                                args.max_level)
                #j = j + 1

            # Update function
            optimizer.step()
            # scheduler.step()
            #for name, param in list(model.named_parameters())[::-1]:
                #param.data = wage_quantizer.W(param.data, param.grad.data, args.wl_weight, args.c2cVari)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                wandb.log({'Epoch': epoch + 1, 'Train Accuracy': acc, 'Train Loss': loss})
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        misc.model_save(model, os.path.join(args.logdir, 'latest.pth'))

        if not os.path.exists('./layer_record'):
            os.makedirs('./layer_record')
        if os.path.exists('./layer_record/trace_command.sh'):
            os.remove('./layer_record/trace_command.sh')

        delta_std = np.array([])
        delta_mean = np.array([])
        w_std = np.array([])
        w_mean = np.array([])
        oldWeight = {}
        k = 0

        for name, param in list(model.named_parameters()):
            oldWeight[k] = param.data + param.grad.data
            k = k + 1
            delta_std = np.append(delta_std, (torch.std(param.grad.data)).cpu().data.numpy())
            delta_mean = np.append(delta_mean, (torch.mean(param.grad.data)).cpu().data.numpy())
            w_std = np.append(w_std, (torch.std(param.data)).cpu().data.numpy())
            w_mean = np.append(w_mean, (torch.mean(param.data)).cpu().data.numpy())

        delta_mean = np.append(delta_mean, delta_std, axis=0)
        np.savetxt(delta_distribution, [delta_mean], delimiter=",", fmt='%f')
        w_mean = np.append(w_mean, w_std, axis=0)
        np.savetxt(weight_distribution, [w_mean], delimiter=",", fmt='%f')

        print("weight distribution")
        print(w_mean)
        print("delta distribution")
        print(delta_mean)

        h = 0

        #for i, layer in enumerate(model.features.modules()):
         #   if isinstance(layer, QConv2d) or isinstance(layer, QLinear):
          #      weight_file_name = './layer_record/weightOld' + str(layer.name) + '.csv'
           #     hook.write_matrix_weight((oldWeight[h]).cpu().data.numpy(), weight_file_name)
            #    h = h + 1
        #for i, layer in enumerate(model.classifier.modules()):
         #   if isinstance(layer, QLinear):
          #      weight_file_name = './layer_record/weightOld' + str(layer.name) + '.csv'
           #     hook.write_matrix_weight((oldWeight[h]).cpu().data.numpy(), weight_file_name)
            #    h = h + 1

        # Run tests including hardware simulation
        accumulated_time += time.time() - split_time
        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            logger("testing phase")
            for i, (data, target) in enumerate(test_loader):
                if i == 0:
                    hook_handle_list = hook.hardware_evaluation(model, args.wl_weight, args.wl_activate,
                                                                epoch, args.batch_size, args.cellBit, args.technode,
                                                                args.wireWidth, args.relu, args.memcelltype,
                                                                2 ** args.ADCprecision,
                                                                args.onoffratio)
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                    output = model(data)
                    # Todo: Update loss function to match here again
                    test_loss_i = wage_util.SSE(output, target).sum()
                    test_loss += test_loss_i.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()
                if i == 0:
                    hook.remove_hook_list(hook_handle_list)

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            wandb.log({'Epoch': epoch + 1, 'Test Accuracy': acc, 'Test Loss': test_loss})
            logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(test_loader.dataset), acc))
            accuracy = acc.cpu().data.numpy()

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_save(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file
            call(["/bin/bash", "./layer_record/trace_command.sh"])

            log_input = {"Epoch": epoch + 1}
            layer_out = pd.read_csv("Layer.csv").to_dict()
            for key, value in layer_out.items():
                for layer, result in value.items():
                    log_input["Layer {}: {}".format(layer + 1, key)] = result
            wandb.log(log_input)
            summary_out = pd.read_csv("Summary.csv").to_dict()
            log_input = {"Epoch": epoch + 1}
            for key, value in summary_out.items():
                exponential = '%.2E' % Decimal(value[0])
                log_input[key] = exponential
            wandb.log(log_input)


except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    wandb.log({'Training time': accumulated_time})
    logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
