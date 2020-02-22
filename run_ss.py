import os


commands = []

# Baselines
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --iters=2000 --budget=2000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --budget=2000')

# MNIST
commands.append('python ./main.py   --icarl')
commands.append('python ./main.py   --icarl  --ss=hard_sampling')


# Fashion-MNIST
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=2000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=2000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=2000')

# CIFAR-10
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=1000 --ss=hard_sampling --hs_samples=64 --budget=2000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=2000 --ss=hard_sampling --hs_samples=64 --budget=2000')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=200')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=500')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --iters=10000 --ss=hard_sampling --hs_samples=64 --budget=2000')


seeds = [11, 101]
budgets = [200, 500, 1000, 2000]
iterss = [500, 1000, 2000]
hs_samples = [40, 64]
result_dir = " --results-dir=./results/sampling_strategy/new/"

for i in range(len(commands)):
    for seed in seeds:
        for budget in budgets:
            for iters in iterss:
                for hs_sample in hs_samples:
                    command = commands[i] + ' --seed='+ str(seed) + result_dir + ' --budget=' + str(budget) + ' --iters=' + str(iters) + ' --hs_samples=' + str(hs_sample)
                    print(command)
                    os.system(command)
