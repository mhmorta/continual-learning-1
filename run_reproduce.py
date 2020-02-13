
import os


commands = []

commands.append('python ./main.py  --tasks=1 --experiment=splitCIFAR10 --model_type=resnet32 --rs=vnet --imb_factor=0.01 --iters=40000 --vnet_loss_ratio=1.0')

result_dir = " --results-dir=./results/"
seeds = [1]

for i in range(len(commands)):
    for seed in seeds:
        command = commands[i] + ' --seed='+ str(seed) + result_dir
        print(command)
        os.system(command)