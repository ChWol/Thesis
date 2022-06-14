import argparse
import os
import time
import csv
# Todo: explain
from utee import misc
# Import Pytorch & Numpy & Pandas
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
# Todo: explain
from utee import make_path
from utee import wage_util
from utee import wage_quantizer
from utee import hook
# Import Cifar dataset
# Todo: Import other datasets
from cifar import dataset
from cifar import model
# Todo: explain
from modules.quantization_cpu_np_infer import QConv2d, QLinear
# from IPython import embed
from datetime import datetime
from subprocess import call
# Import weights and biases
import wandb

# Parsing training & architecture parameters
# Todo: Explain each parameter
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train')
parser.add_argument('--grad_scale', type=float, default=1, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=6, help='weight precision')
parser.add_argument('--wl_grad', type=int, default=6, help='gradient precision')
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
parser.add_argument('--onoffratio', type=int, default=10)
parser.add_argument('--cellBit', type=int, default=6, help='cell precision (cellBit==wl_weight==wl_grad)')
parser.add_argument('--inference', default=0)
parser.add_argument('--subArray', type=int, default=128)
parser.add_argument('--ADCprecision', type=int, default=5)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)
parser.add_argument('--nonlinearityLTP', type=float, default=1.75, help='nonlinearity in LTP')
parser.add_argument('--nonlinearityLTD', type=float, default=-1.46,
                    help='nonlinearity in LTD (negative if LTP and LTD are asymmetric)')
parser.add_argument('--max_level', type=int, default=32,
                    help='Maximum number of conductance states during weight update (floor(log2(max_level))=cellBit)')
parser.add_argument('--d2dVari', type=float, default=0, help='device-to-device variation')
parser.add_argument('--c2cVari', type=float, default=0.003, help='cycle-to-cycle variation')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--network', default='speed')
parser.add_argument('--run', default='')
parser.add_argument('--technode', type=int, default='32')
parser.add_argument('--memcelltype', type=int, default=3)
parser.add_argument('--relu', type=int, default=1)
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
# Manually overwriting arguments to match simulator-conditions
if args.memcelltype == 1:
    args.cellBit = 1
args.wl_weight = args.cellBit
args.wl_grad = args.cellBit

technode_to_width = { 7: 14, 10: 14, 14: 22, 22: 32, 32: 40, 45: 50, 65: 100, 90: 200, 130: 200 }
args.wireWidth = technode_to_width[args.technode]

# Initializing Weights and Biases
wandb.init(project=args.type, config=args)
wandb.run.name = (args.network + ' - ' + args.run + ' ({})').format(wandb.run.id)

# momentum
gamma = args.momentum
alpha = 1 - args.momentum

# Output values for simulation/hardware
NeuroSim_Out = np.array([["L_forward (s)", "L_activation gradient (s)", "L_weight gradient (s)", "L_weight update (s)",
                          "E_forward (J)", "E_activation gradient (J)", "E_weight gradient (J)", "E_weight update (J)",
                          "L_forward_Peak (s)", "L_activation gradient_Peak (s)", "L_weight gradient_Peak (s)",
                          "L_weight update_Peak (s)",
                          "E_forward_Peak (J)", "E_activation gradient_Peak (J)", "E_weight gradient_Peak (J)",
                          "E_weight update_Peak (J)",
                          "TOPS/W", "TOPS", "Peak TOPS/W", "Peak TOPS"]])
np.savetxt("NeuroSim_Output.csv", NeuroSim_Out, delimiter=",", fmt='%s')
if not os.path.exists('./NeuroSim_Results_Each_Epoch'):
    os.makedirs('./NeuroSim_Results_Each_Epoch')

# Output values for network
out = open("PythonWrapper_Output.csv", 'ab')
out_firstline = np.array([["epoch", "average loss", "accuracy"]])
np.savetxt(out, out_firstline, delimiter=",", fmt='%s')

# Todo: explain
delta_distribution = open("delta_dist.csv", 'ab')
delta_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean", "1_std",
                             "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(delta_distribution, delta_firstline, delimiter=",", fmt='%s')

# Todo: explain
weight_distribution = open("weight_dist.csv", 'ab')
weight_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean", "7_mean", "8_mean", "1_std",
                              "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(weight_distribution, weight_firstline, delimiter=",", fmt='%s')

# Todo: explain
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args, ['log_interval', 'test_interval', 'logdir', 'epochs'])
misc.logger.init(args.logdir, 'train_log_' + current_time)
logger = misc.logger.info

# console logger
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

# data loader and model
# Todo: Allow changes, add further datasets
assert args.type in ['cifar10', 'cifar100', 'mnist'], args.type
assert args.network in ['speed', 'vgg8', 'vgg16', 'alexnet'], args.network
if args.type == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
    model = model.cifar10(args=args, logger=logger)
if args.type == 'cifar100':
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
    model = model.cifar100(args=args, logger=logger)
if args.type == 'mnist':
    train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
    model = model.mnist(args=args, logger=logger)
if args.cuda:
    model.cuda()

# Todo: Try different architectures, maybe point to change for different algorithms
optimizer = optim.SGD(model.parameters(), lr=1)

# Todo: explain
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
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
    # Todo: explain
    for layer in list(model.parameters())[::-1]:
        d2dVariation = torch.normal(torch.zeros_like(layer), args.d2dVari * torch.ones_like(layer))
        NL_LTP = torch.ones_like(layer) * args.nonlinearityLTP + d2dVariation
        NL_LTD = torch.ones_like(layer) * args.nonlinearityLTD + d2dVariation
        paramALTP[k] = wage_quantizer.GetParamA(NL_LTP.cpu().numpy()) * args.max_level
        paramALTD[k] = wage_quantizer.GetParamA(NL_LTD.cpu().numpy()) * args.max_level
        k = k + 1

    # Actual training process
    for epoch in range(args.epochs):
        model.train()

        # Todo: explain
        velocity = {}
        i = 0
        for layer in list(model.parameters())[::-1]:
            velocity[i] = torch.zeros_like(layer)
            i = i + 1

        if epoch in decreasing_lr:
            grad_scale = grad_scale / 8.0

        logger("training phase")
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = wage_util.SSE(output, target)

            # Backpropagation, possible point of change for different algorithms
            loss.backward()

            # Todo: explain
            # introduce non-ideal property
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

            # Update function
            optimizer.step()

            # Todo: explain
            for name, param in list(model.named_parameters())[::-1]:
                param.data = wage_quantizer.W(param.data, param.grad.data, args.wl_weight, args.c2cVari)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                wandb.log({'epoch': epoch+1, 'train_accuracy': acc, 'train_loss': loss})
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        # Todo: Check if model saving works
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

        # Todo: explain
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

        # Todo: explain
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

        # Run tests including hardware simulation
        # Todo: explain
        # Todo: extract printed information for WandB
        # Todo: not only log on last
        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            logger("testing phase")
            for i, (data, target) in enumerate(test_loader):
                if i == 0:
                    hook_handle_list = hook.hardware_evaluation(model, args.wl_weight, args.wl_activate,
                                                                epoch, args.batch_size, args.cellBit, args.technode,
                                                                args.wireWidth, args.relu, args.memcelltype, 2 ** args.ADCprecision,
                                                                args.onoffratio)
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                    output = model(data)
                    test_loss_i = wage_util.SSE(output, target)
                    test_loss += test_loss_i.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()
                if i == 0:
                    hook.remove_hook_list(hook_handle_list)

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            wandb.log({'epoch': epoch+1, 'test_accuracy': acc, 'test_loss': test_loss})
            logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(test_loader.dataset), acc))
            accuracy = acc.cpu().data.numpy()
            np.savetxt(out, [[epoch, test_loss.cpu(), accuracy]], delimiter=",", fmt='%f')

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_save(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file
            call(["/bin/bash", "./layer_record/trace_command.sh"])

            log_input = {"Epoch": epoch+1}
            layer_out = pd.read_csv("Layer.csv").to_dict()
            for key, value in layer_out.items():
                for layer, result in value.items():
                    log_input["Layer {}: {}".format(layer+1, key)] = result
            wandb.log(log_input)
            summary_out = pd.read_csv("Summary.csv").to_dict()
            log_input = {"Epoch": epoch + 1}
            for key, value in summary_out.items():
                log_input[key] = value[0]
            wandb.log(log_input)

except Exception as e:
    import traceback

    traceback.print_exc()
finally:
    logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
