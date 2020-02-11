
import os


commands = []

# Joint-training
# commands.append('python ./main.py  --tasks=1 --experiment=splitMNIST')
# commands.append('python ./main.py  --tasks=1 --experiment=splitFashionMNIST')
commands.append('python ./main.py  --tasks=1 --experiment=splitCIFAR10 --model_type=resnet32 --iters=10000')

# Fine-tuning
# commands.append('python ./main.py  --tasks=5 --experim ent=splitMNIST')
# commands.append('python ./main.py  --tasks=5 --experiment=splitFashionMNIST')
# commands.append('python ./main.py  --tasks=5 --experiment=splitCIFAR10 --model_type=resnet32')

# -----------------------------
# --------- iCarl  -------------

# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --rs=vnet')

# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --rs=vnet')

commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32 --rs=vnet  ')

# -----------------------------
# --------- DGR  -------------

# commands.append('python ./main.py  --tasks=5 --experiment=splitMNIST --replay=generative ')
# commands.append('python ./main.py  --tasks=5 --experiment=splitMNIST --replay=generative --rs=vnet')

# commands.append('python ./main.py  --tasks=5 --experiment=splitFashionMNIST --replay=generative ')
# commands.append('python ./main.py  --tasks=5 --experiment=splitFashionMNIST --replay=generative --rs=vnet')

# seeds = [1, 10, 100]
seeds = [11, 101, 1001]

result_dir = " --results-dir=./results/vnet_balanced"

for i in range(len(commands)):
    for seed in seeds:
        command = commands[i] + ' --seed='+ str(seed) + result_dir
        print(command)
        os.system(command)
