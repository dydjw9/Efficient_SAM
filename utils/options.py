#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujiawei@u.nus.edu'
#Descrption:
from utils.MiscTools import *

from model.PyramidNet import PyramidNet as PYRM
from model.wideresnet import WideResNet28
from model.resnet import ResNet18 as resnet18
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
parser.add_argument("--beta", default=1.0, type=float, help="Dropout rate.")
parser.add_argument("--gamma", default=1.0, type=float, help="Dropout rate.")
parser.add_argument("--wide_dropout", default=0., type=float, help="Dropout rate.")
parser.add_argument("--nograd_cutoff", default=0.05, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--temperature", default=300, type=int, help="temperature.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--sam_updating_layers", default="full", type=str)
parser.add_argument("--output_dir", default="../output/", type=str)
parser.add_argument("--name", default="r1", type=str)
parser.add_argument("--resume_name", default = "None", type=str)
parser.add_argument("--arch", default="resnet18", type=str)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--SCE_loss", default="False", type=str)
parser.add_argument('--isSAM', type=str, default='True')
parser.add_argument('--isASAM', type=str, default='False')

#fp16
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")

#distributed learning
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")


args = parser.parse_args()
vars(args)['isSAM'] = str2bool(args.isSAM)
vars(args)['isASAM'] = str2bool(args.isASAM)
vars(args)['SCE_loss'] = str2bool(args.SCE_loss)
vars(args)['opt_dropout'] = 1-args.gamma
vars(args)['weight_dropout'] = 1-args.beta


if args.dataset == "cifar100":
    vars(args)["num_classes"] = 100



def setup_model(args):
    if args.arch=="resnet18":
        model = resnet18(num_classes = args.num_classes)
    elif args.arch=="wideresnet18":
        model = WideResNet28(args.num_classes)
    else:
        model = PYRM(args.dataset, 110, 270, args.num_classes, False) # for ResNet  

    # if resume 
    if args.resume_name != "None":
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.resume_name)
        model.load_state_dict(torch.load(model_checkpoint,map_location='cpu'))
 
    model.to(args.device)
    num_params = count_parameters(model)
    return model



