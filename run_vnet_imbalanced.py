import os


commands = []

# Joint-training
# commands.append('python ./main.py  --tasks=1 --experiment=splitMNIST --iters=10000')
# commands.append('python ./main.py  --tasks=1 --experiment=splitFashionMNIST --iters=10000')
# commands.append('python ./main.py  --tasks=1 --experiment=splitCIFAR10 --model_type=resnet32 --iters=10000')


# commands.append('python ./main.py  --tasks=1 --experiment=splitMNIST --rs=vnet --iters=10000')
# commands.append('python ./main.py  --tasks=1 --experiment=splitFashionMNIST --rs=vnet --iters=10000')
commands.append('python ./main.py  --tasks=1 --experiment=splitCIFAR10 --rs=vnet --model_type=resnet32 --iters=10000')

# -----------------------------
# --------- iCarl  -------------
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --rs=vnet')

# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=3000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --rs=vnet --iters=5000')

# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32 --rs=vnet')

# # -----------------------------
# --------- DGR  -------------

# commands.append('python ./main.py  --tasks=5 --experiment=splitMNIST --replay=generative ')
# commands.append('python ./main.py  --tasks=5 --experiment=splitMNIST --replay=generative --rs=vnet')

# commands.append('python ./main.py  --tasks=5 --experiment=splitFashionMNIST --replay=generative --iters=3000')
# commands.append('python ./main.py  --tasks=5 --experiment=splitFashionMNIST --replay=generative --rs=vnet --iters=3000')


# seeds = [1, 10, 100, 102]
seeds = [11, 101, 1001, 1021]

result_dir = " --results-dir=./results/vnet_imbalanced"
imb_factors = [0.1, 0.05]
for i in range(len(commands)):
    for seed in seeds:
        for imb_factor in imb_factors:
            command = commands[i] + ' --seed='+ str(seed) + result_dir + ' --imb_factor=' + str(imb_factor)
            print(command)
            os.system(command)
