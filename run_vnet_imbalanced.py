import os


commands = []


# Baselines
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --budget=2000 --imb_factor=0.1')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --budget=2000 --imb_factor=0.02')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --budget=2000 --imb_factor=0.1')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --budget=2000 --imb_factor=0.02')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --budget=2000 --imb_factor=0.1')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --budget=2000 --imb_factor=0.02')

commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --rs=vnet --imb_factor=0.1 ')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --rs=vnet --imb_factor=0.02 ')

commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --rs=vnet --imb_factor=0.1 ')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --rs=vnet --imb_factor=0.02 ')

commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --rs=vnet --imb_factor=0.1 ')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --rs=vnet --imb_factor=0.02 ')


for i in range(len(commands)):
    print(commands[i])
    os.system(commands[i])
