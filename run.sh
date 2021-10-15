python -m torch.distributed.launch --nproc_per_node=4 train.py --beta 0.6 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --name R18C10_01 --batch_size 64 --dataset cifar10
#python -m torch.distributed.launch --nproc_per_node=4 train.py --beta 0.6 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.05 --name R18C100_01 --batch_size 64 --dataset cifar100

#python -m torch.distributed.launch --nproc_per_node=4 train.py --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 64  --arch wideresnet18 --name wideC10 --dataset cifar10
#python -m torch.distributed.launch --nproc_per_node=4 train.py --beta 0.5 --gamma 0.5  --learning_rate 0.05 --weight_decay 1e-3  --rho 0.1 --batch_size 64  --arch wideresnet18 --name wideC100 --dataset cifar100

#python -m torch.distributed.launch --nproc_per_node=4 train.py --name PmidC10 --beta 0.5 --gamma 0.5  --batch_size 64 --learning_rate 0.05 --weight_decay 5e-4 --rho 0.2 --arch pyramidnet --dataset cifar10 --epoch 300 
#python -m torch.distributed.launch --nproc_per_node=4 train.py --name PmidC100 --beta 0.5 --gamma 0.5  --batch_size 64 --learning_rate 0.05 --weight_decay 5e-4 --rho 0.2 --arch pyramidnet --dataset cifar100 --epoch 300 
