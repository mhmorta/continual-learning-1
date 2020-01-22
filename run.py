import os


commands = []

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

# ----------------------------------------------------------
# imbalanced0.1 weightedCE
commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')
commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')
commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.1 --reweighting_strategy=weighted_ce')

# imbalanced0.01 weightedCE
commands.append('python ./main.py --experiment=splitMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')
commands.append('python ./main.py --experiment=splitFashionMNIST --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')
commands.append('python ./main.py --experiment=splitCIFAR10 --tasks=1 --iters=10000 --imb_factor=0.01 --reweighting_strategy=weighted_ce')

# ----------------------------------------------------------

for i in range(len(commands)):
    print(commands[i])
    os.system(commands[i])
