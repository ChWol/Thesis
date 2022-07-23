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
from modules.quantization_cpu_np_infer import QConv2d, QLinear
import wandb
from decimal import Decimal
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluation of Biologically-Plausible Learning Rules on Neuromorphic '
                                             'Hardware Architectures')
parser.add_argument('--dataset', default='mnist', help='dataset for training')
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
parser.add_argument('--network', default='two')
parser.add_argument('--technode', type=int, default='32')
parser.add_argument('--memcelltype', type=int, default=3)
parser.add_argument('--activation', default='relu')
parser.add_argument('--rule', default='bp')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--neurosim', type=int, default=1)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--scheduler', type=int, default=0)
parser.add_argument('--initial', default='xavier')
parser.add_argument('--gradient_analysis', type=int, default=0)

args = parser.parse_args()
args.wl_weight = args.wl_grad = args.cellBit
args.max_level = 2 ** args.cellBit
if args.memcelltype == 1:
    args.cellBit = 1
technode_to_width = {7: 14, 10: 14, 14: 22, 22: 32, 32: 40, 45: 50, 65: 100, 90: 200, 130: 200}
args.wireWidth = technode_to_width[args.technode]

gamma = 0.9
alpha = 0.1

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

wandb.init(project=args.dataset.upper(), config=args, entity='duke-tum')
wandb.run.name = "{} ({}): {}".format(args.network, args.rule, wandb.run.id)

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

assert args.dataset in ['cifar10', 'cifar100', 'mnist', 'fashion'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=10)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
    model = models.cifar(args=args, logger=logger, num_classes=100)
elif args.dataset == 'mnist':
    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
    model = models.mnist(args=args, logger=logger, num_classes=10)
elif args.dataset == 'fashion':
    train_loader, test_loader = dataset.get_fashion(batch_size=args.batch_size, num_workers=1)
    model = models.mnist(args=args, logger=logger, num_classes=10)
else:
    raise ValueError("Unknown dataset type")

if args.cuda:
    model.cuda()

if args.optimizer == 'adam':
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam([{"params": model.classifier[0].parameters(), "lr": 1e-2},
                            {"params": model.classifier[1].parameters(), "lr": 1e-2},
                            {"params": model.classifier[2].parameters(), "lr": 1e-3},
                            {"params": model.classifier[3].parameters(), "lr": 1e-3}],
                           lr=args.learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
if args.scheduler == 1:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

best_acc, old_file = 0, None
accumulated_time = 0
gradient_accumulated = 0
t_begin = time.time()
grad_scale = args.grad_scale

try:
    if args.cellBit != args.wl_weight:
        print("Warning: Weight precision should be the same as the cell precison !")
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

        logger("training phase")
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            gradient_time = time.time()
            if args.rule == 'dfa':
                # ToDo: Optimize this calculation with torch no grad
                with torch.no_grad():
                    output = model(data)
                    error = wage_util.SSE(output, target)
                    loss = (0.5 * (error ** 2)).sum()
                    model.direct_feedback_alignment(error)
            else:
                output = model(data)
                error = wage_util.SSE(output, target)
                loss = (0.5 * (error ** 2)).sum()
                loss.backward()
            gradient_accumulated += time.time() - gradient_time

            j = 0
            for name, param in list(model.named_parameters())[::-1]:
                velocity[j] = gamma * velocity[j] + alpha * param.grad.data
                param.grad.data = velocity[j]
                param.grad.data = wage_quantizer.QG(param.data, args.wl_weight, param.grad.data, args.wl_grad,
                                                    grad_scale,
                                                    torch.from_numpy(paramALTP[j]).cuda(),
                                                    torch.from_numpy(paramALTD[j]).cuda(), args.max_level,
                                                    args.max_level)
                j = j + 1

            optimizer.step()

            for name, param in list(model.named_parameters())[::-1]:
                param.data = wage_quantizer.W(param.data, param.grad.data, args.wl_weight, args.c2cVari)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                wandb.log({'Epoch': epoch + 1, 'Train Accuracy': acc, 'Train Loss': loss})
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))

                for name, param in model.named_parameters():
                    with torch.no_grad():
                        weights_np = torch.clone(param).cpu()
                        gradients_np = torch.clone(param.grad).cpu()
                        weights = torch.reshape(weights_np, (-1,))
                        gradients = torch.reshape(gradients_np, (-1,))
                    if args.gradient_analysis == 1:
                        wandb.log({"Gradient visualization of {}".format(name): [
                            wandb.Image(plt.imshow(gradients_np, cmap='viridis'), caption="Gradient")],
                            "Weight visualization of {}".format(name): [
                                wandb.Image(plt.imshow(weights_np, cmap='viridis'), caption="Weight")],
                            "Epoch": epoch + 1
                        })
                    wandb.log({"Weight avg of {}".format(name): torch.mean(param),
                               "Weight std of {}".format(name): torch.std(param),
                               "Gradient avg of {}".format(name): torch.mean(param.grad),
                               "Gradient std of {}".format(name): torch.std(param.grad),
                               "Gradients of {}".format(name): wandb.Histogram(gradients),
                               "Weights of {}".format(name): wandb.Histogram(weights),
                               'Epoch': epoch + 1})

        if args.scheduler == 1:
            scheduler.step()

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        if epoch == args.epochs - 1:
            wandb.log({"Time/epoch": speed_epoch, "Time/Batch": speed_batch})

        misc.model_save(model, os.path.join(args.logdir, 'latest.pth'))

        if not os.path.exists('./layer_record'):
            os.makedirs('./layer_record')
        if os.path.exists('./layer_record/trace_command.sh'):
            os.remove('./layer_record/trace_command.sh')

        oldWeight = {}
        k = 0

        for name, param in list(model.named_parameters()):
            oldWeight[k] = param.data + param.grad.data
            k = k + 1

        h = 0

        for i, layer in enumerate(model.features.modules()):
            if isinstance(layer, QConv2d) or isinstance(layer, QLinear):
                weight_file_name = './layer_record/weightOld' + str(layer.name) + '.csv'
                hook.write_matrix_weight((oldWeight[h]).cpu().data.numpy(), weight_file_name)
                h = h + 1
        for i, layer in enumerate(model.classifier.modules()):
            if isinstance(layer, QLinear):
                weight_file_name = './layer_record/weightOld' + str(layer.name) + '.csv'
                hook.write_matrix_weight((oldWeight[h]).cpu().data.numpy(), weight_file_name)
                h = h + 1

        accumulated_time += time.time() - split_time
        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            logger("testing phase")
            for i, (data, target) in enumerate(test_loader):
                if i == 0:
                    if args.activation == 'sigmoid':
                        relu = 0
                    else:
                        relu = 1
                    hook_handle_list = hook.hardware_evaluation(model, args.wl_weight, args.wl_activate,
                                                                epoch, args.batch_size, args.cellBit, args.technode,
                                                                args.wireWidth, relu, args.memcelltype,
                                                                2 ** args.ADCprecision,
                                                                args.onoffratio, args.rule)
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
            wandb.log({'Epoch': epoch + 1, 'Test Accuracy': acc, 'Test Loss': test_loss})
            logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(test_loader.dataset), acc))
            accuracy = acc.cpu().data.numpy()

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_save(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file

            if args.neurosim == 1:
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
    wandb.log({'Training time': accumulated_time, 'Gradient time': gradient_accumulated})
    logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
