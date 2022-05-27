# Efficient Sharpness-aware Minimization for Improved Training of Neural Networks

Code for [“Efficient Sharpness-aware Minimization for Improved Training of Neural Networks”](https://openreview.net/forum?id=n0OeTdNRG0Q), which has been accepte by ICLR 2022. 


## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.8.8
- torch = 1.8.0
- torchvision = 0.9.0

## What is in this repository

Codes for our ESAM on CIFAR10/CIFAR100 datasets. 







## How to use it

```
from utils.layer_dp_sam import ESAM
base_optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
optimizer = ESAM(paras, base_optimizer, rho=args.rho, weight_dropout=args.weight_dropout,adaptive=args.isASAM,nograd_cutoff=args.nograd_cutoff,opt_dropout = args.opt_dropout,temperature=args.temperature)
```

--beta the SWP hyperparameter

--gamma the SDS hyperparameter

During training 
loss_fct should have reduction="none", to return instance-wise losses. 
defined_backward is the function used for DDP and mixed precision backward

```
loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
def defined_backward():
    if args.fp16:
    with amp.scale_loss(loss, optimizer0) as scaled_loss:
        scaled_loss.backward()
    else:
        loss.backward()

paras = [inputs,targets,loss_fct,model,defined_backward]
optimizer.paras = paras
optimizer.step()
predictions_logits,loss = optimizer.returnthings
```

## Example

```bash run.sh```



## Reference Code
[1] [SAM](https://github.com/davda54/sam)
