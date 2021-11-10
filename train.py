import argparse
import torch

from model.smooth_cross_entropy import smooth_crossentropy,trades_loss
from utils.cifar import Cifar,Cifar100
from utils.log import Log
from utils.initialize import initialize
from utils.step_lr import StepLR
from utils.Esam import ESAM
from torch.utils.tensorboard import SummaryWriter
import os 
from utils.mail import send_email

from utils.options import args,setup_model
from utils.MiscTools import count_parameters
from utils.dist_util import get_world_size

import torch.nn.functional as F
import logging
from datetime import timedelta
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)
def train(args,model):

    initialize(args, seed=42)
    device = args.device

    dataset = Cifar(args) if args.dataset =="cifar10" else Cifar100(args)
    log = Log(log_each=10)

    if args.SCE_loss =="True":
        loss_fct = smooth_crossentropy
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    paras = model.parameters()
    base_optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
    optimizer = ESAM(paras, base_optimizer, rho=args.rho, beta=args.beta,gamma=args.gamma,adaptive=args.isASAM,nograd_cutoff=args.nograd_cutoff)
    optimizer0 = optimizer.base_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=args.epochs)

    #half float setting 
    if args.fp16:
        model, [optimizer0,optimizer] = amp.initialize(models=model,
                                      optimizers=[optimizer0,optimizer],
                                      opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        opt_list = [optimizer0,optimizer]

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)



    best_acc = 0.0
    global_step = -1
    sampler = dataset.train.sampler
    for epoch in range(args.epochs):
        if args.local_rank != -1:
            sampler.set_epoch(epoch)
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            global_step += 1
            # inputs, targets = (b.to(args.device) for b in batch)
            inputs, targets = (b.to(args.device) for b in batch)

            def defined_backward(loss):
                if args.fp16:
                    with amp.scale_loss(loss, optimizer0) as scaled_loss:
                    # with amp.scale_loss(loss, [optimizer0,optimizer]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            paras = [inputs,targets,loss_fct,model,defined_backward]
            optimizer.paras = paras
            optimizer.step()
            predictions,loss = optimizer.returnthings


 

            with torch.no_grad():
                if len(inputs)!=len(predictions):
                    continue
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])
                acc = (correct.sum()+0.01) / (len(targets)+0.01) 

            if  args.local_rank in [-1, 0]:
                writer.add_scalar("train/loss", scalar_value=loss.mean(), global_step=global_step)
                writer.add_scalar("train/acc", scalar_value=acc, global_step=global_step)

        scheduler.step()
        if  args.local_rank in [-1, 0]:
            model.eval()
            log.eval(len_dataset=len(dataset.test))

            with torch.no_grad():
                tol_cor = 0
                tol_len = 0
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
                    acc = (correct.sum()+0.01) / (len(targets)+0.01) 
                    tol_len += len(targets)
                    tol_cor += correct.sum()
                acc = tol_cor/(tol_len+0.0)
                if acc > best_acc:
                    best_acc = acc 
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(),"../output/"+"%s_checkpoint.bin" %args.name)
                writer.add_scalar("test/acc", scalar_value=tol_cor/(tol_len+0.0), global_step=global_step)
    if args.local_rank in [-1,0]:
        email_text = "training of {} finished, best acc is {}".format(args.name,best_acc) 
        send_email(email_text)
        log.flush()

def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    train_name = "train" 
    log_path = args.name + "_" + train_name
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',filename = '../output/logs/'+log_path,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed

    # Model & Tokenizer Setup
    model = setup_model(args)


    # Training
    train(args, model)


if __name__ == "__main__":
    main(args)
