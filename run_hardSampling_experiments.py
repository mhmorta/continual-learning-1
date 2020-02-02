import os


commands = []

# Baselines
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --budget=2000')

# MNIST
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=2000')

# Fashion-MNIST
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=2000')

# CIFAR-10
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=2000')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=200')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=500')
commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=2000')


for i in range(len(commands)):
    print(commands[i])
    os.system(commands[i])
