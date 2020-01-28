import os


commands = []

# -------------------------------------------------
# ---------- 0: Reproducing Meta-Weight net ----------
# -------------------------------------------------
# resnet performance over balanced data
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=30000 --imb_factor=1.0 --model_type=resnet32 --optimizer=sgd_momentum --lr=0.1' )

commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=40000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')

# resnet over imbalanced data
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=100 --imb_factor=0.01 --reweighting_strategy=vnet'
#                 ' --vnet_loss_ratio=1.0 --model_type=resnet32 --vnet_exemplars_per_class=10')


# -------------------------------------------------
# ---------- 1: Imbalanced data analysis ----------
# -------------------------------------------------

# # - Baselines
# # balanced JT
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 ')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 ')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 ')
#
# # ----------------------------------------------------------
# # imbalanced0.1
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1')
#
# # imbalanced0.01
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.01')

# # ----------------------------------------------------------
# # imbalanced0.1 weightedCE
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')
#
# # imbalanced0.01 weightedCE
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')

# ----------------------------------------------------------
# vnet experiments

# imbalanced0.1 vnet0.5
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet')
#
# # imbalanced0.01 vnet0.5
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=20000 --imb_factor=0.01 --reweighting_strategy=vnet')

# # imbalanced0.1 vnet0.2
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')
#
# # imbalanced0.01 vnet0.2
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=0.2')

# # imbalanced0.1 vnet1.0
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')
#
# # imbalanced0.01 vnet1.0
# commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')
# commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=1.0')
# commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=vnet --vnet_loss_ratio=1.0 --model_type=resnet32 --')



for i in range(len(commands)):
    print(commands[i])
    os.system(commands[i])
