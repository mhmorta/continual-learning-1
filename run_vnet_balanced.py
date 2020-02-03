import os


commands = []

# Fine-tuning
commands.append('python ./main.py  --tasks=1 --experiment=splitMNIST')
commands.append('python ./main.py  --tasks=1 --experiment=splitFashionMNIST --model_type=resnet32')
commands.append('python ./main.py  --tasks=1 --experiment=splitCIFAR10 --model_type=resnet32')

# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitMNIST --rs=vnet')
#
#
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --rs=vnet')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --model_type=resnet32')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitFashionMNIST --model_type=resnet32 --rs=vnet')
#
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --rs=vnet  ')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32')
# commands.append('python ./main.py  --tasks=5 --icarl --experiment=splitCIFAR10 --model_type=resnet32 --rs=vnet  ')


for i in range(len(commands)):
    print(commands[i])
    os.system(commands[i])
