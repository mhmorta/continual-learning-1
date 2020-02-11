import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
import time
import matplotlib.pyplot as plt
import os

from data import SubDataset, ExemplarDataset
from continual_learner import ContinualLearner
from vnet import *
from visual_plt import plot_loss_vs_weight, plot_class_vs_loss, plot_losses


def train_cl(model, train_datasets, meta_datasets, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, eval_cbs_exemplars=list(), reweighting_strategy = 'none', imb_factor = 1.0,
             imb_inverse= False, reset_vnet = False, reset_vnet_optim=False, vnet_enable_from = 2, vnet_exemplars_per_class = 20,
             metadataset_building_strategy = 'none', vnet_loss_ratio=0.5, vnet_opt=None, vnet_dir = "", sampling_strategy=None,
             vnet_plot_count = 4, hs_samples = 40):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress
    [use_vnet] `        <bool> should it use vnet?
    [imb_factor]        <float> imbalanced data factor
    '''


    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()
    print('Running on {}'.format(device))


    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    vnet = None

    # Initializations of the vnet
    if reweighting_strategy=='vnet':
        vnet = VNet(1, 100, 1).to(device)

        vnet_plot_freq = int(iters / (vnet_plot_count-1))

        vnet_weights_dict = {}
        vnet_weights_dict[0] = vnet.loss_weights()

        if vnet_opt=='sgd_momentum':
            optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                          momentum=0.9, nesterov=True,
                                          weight_decay=5e-4)
        elif vnet_opt=='sgd':
            optimizer_c = torch.optim.SGD(vnet.params(), 1e-3)
        elif vnet_opt=='adam':
            optimizer_c = torch.optim.Adam(vnet.params(), betas=(0.9, 0.999))
        else:
            optimizer_c = None


        class_weights = {}
        class_loss = {}
        class_weighted_loss = {}
        for i in range(10):
            class_weights[i] = []
            class_loss[i] = []
            class_weighted_loss[i] = []

        loss_list = []
        vnet_loss_list = []
        loss_original_list = []

        meta_sub_indeces_list = []
        torch.backends.cudnn.benchmark = True



    metadata_list = []
    CE_weights = None
    x_meta = None
    y_meta = None
    x_top = None
    y_top = None
    training_dataset_sampler = None

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        if reset_vnet == True:
            vnet = VNet(1, 200, 1).to(device)
            optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                          momentum=0.9, nesterov=True,
                                          weight_decay=5e-4)

        # --------------------------------------------
        # creating dataloader for meta-data set
        new_classes = list(range(classes_per_task)) if scenario == "domain" else list(
            range(classes_per_task * (task - 1),
                  classes_per_task * task))

        # ---------------- build meta-training set ----------------
        classes = int(task * 10 / len(train_datasets))
        meta_dataset = ConcatDataset(meta_datasets[:task])
        rand_sampler = torch.utils.data.RandomSampler(meta_dataset, replacement=False)
        train_meta_loader = iter(utils.get_data_loader(
            meta_dataset, batch_size=vnet_exemplars_per_class * classes, cuda=cuda, drop_last=True,
            sampler=rand_sampler, shuffle=False
        ))
        x_meta, y_meta = next(train_meta_loader)
        # ---------------- end of meta-training --------------------

        if reweighting_strategy=='vnet' or reweighting_strategy=='meta_update' or reweighting_strategy=='hard_sampling':
            if metadataset_building_strategy == 'trainingset':
                # targets = np.array(train_dataset.sub_indeces).astype(int)
                # tmp = np.array([(i, train_dataset.dataset[i][1]) for i in targets])
                # for cls in new_classes:
                #     sub_targets = np.where(tmp[:,1] == 0)[0]
                #     ind = np.random.choice(sub_targets, vnet_exemplars_per_class)
                #     meta_sub_indeces_list.extend(tmp[ind][:,0])
                #
                # validation_sub_dataset = torch.utils.data.Subset(train_dataset, meta_sub_indeces_list)
                # train_meta_loader = iter(utils.get_data_loader(
                #     validation_sub_dataset, 20, cuda=cuda, drop_last=False, shuffle=True
                # ))
                #
                # # meta_training_sampler = torch.utils.data.SubsetRandomSampler(meta_sub_indeces_list)
                # # train_meta_loader = iter(utils.get_data_loader(
                # #     train_dataset, 20, cuda=cuda, drop_last=False, shuffle=False, sampler=meta_training_sampler
                # # ))
                #
                # x, y = next(train_meta_loader)
                # x_meta = x if x_meta is None else torch.cat((x_meta, x), dim=0)
                #
                # training_dataset_sampler = torch.utils.data.SubsetRandomSampler(np.delete(targets, meta_sub_indeces_list))
                # print('Meta-dataset generated from training-set')
                print('implementation should be updated')
                raise

            # elif metadataset_building_strategy == 'testset':
                # classes = int(task * 10 / len(train_datasets))
                #
                # meta_dataset = ConcatDataset(meta_datasets[:task])
                #
                # rand_sampler = torch.utils.data.RandomSampler(meta_dataset, replacement=False)
                #
                # train_meta_loader = iter(utils.get_data_loader(
                #     meta_dataset, batch_size = vnet_exemplars_per_class * classes, cuda=cuda, drop_last=True, sampler=rand_sampler, shuffle=False
                # ))
                #
                # x_meta, y_meta = next(train_meta_loader)
                # print('implementation should be updated')
                # raise

            elif metadataset_building_strategy == 'exemplar' and len(model.exemplar_sets)>0 and task >= vnet_enable_from:
                # num_classes = len(model.exemplar_sets)
                # meta_ds_list = []
                #
                # for class_id in range(num_classes):
                #     indxs = np.random.choice(len(model.exemplar_sets[class_id]), vnet_exemplars_per_class)
                #     meta_ds_list.append(model.exemplar_sets[class_id][indxs])
                #
                # for class_id in new_classes:
                #     # create new dataset containing only all examples of this class
                #     current_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                #     # based on this dataset, construct new exemplar-set for this class
                #     model.construct_exemplar_set(dataset=current_dataset, n=vnet_exemplars_per_class, meta_data=True)
                #
                # for l in model.meta_data:
                #     meta_ds_list.append(l)
                #
                # model.meta_data = []
                #
                # vnet_meta_dataset = ExemplarDataset(meta_ds_list, target_transform=None)
                #
                # train_meta_loader = iter(cycle(utils.get_data_loader(
                #     vnet_meta_dataset, 100, cuda=cuda, drop_last=False, shuffle=True
                # )))
                # print('Meta-dataset generated from exemplar-set')
                print('implementation should be updated')
                raise
            # x_meta, y_meta = next(train_meta_loader)

        # ----------------------------------------

        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets()

        # ----------------------------------------
        # Creating imabalanced training_dataset
        # ----------------------------------------

        # Creating trainingset data loader for JT case
        if imb_factor < 1.0 and len(train_datasets) == 1:
            classes_sub_datasets = []
            samples_per_class = [int(np.floor(5000*((imb_factor)**(i / (10 - 1.0))))) for i in range(10) ]
            samples_per_class = [samples_per_class[i] for i in range(9,-1,-1)  ] if imb_inverse else samples_per_class
            if reweighting_strategy=="weighted_ce":
                summ = sum(samples_per_class)
                CE_weights = torch.FloatTensor([ (1.0 / class_count) for class_count in samples_per_class]).to(device)
                print("CE class weights: ", CE_weights.data)

            print("samples per classes = ", samples_per_class)

            imb_sub_indeces_list = []
            targets = np.array(train_dataset.dataset.targets)
            for cls in range(10):
                sub_targets = np.where(targets == cls)[0]
                inds = np.random.choice(sub_targets, samples_per_class[cls])
                imb_sub_indeces_list.extend(inds)

            # imb_sub_indeces_list = np.delete(imb_sub_indeces_list, meta_sub_indeces_list)
            training_dataset_sampler = torch.utils.data.SubsetRandomSampler(imb_sub_indeces_list)

            print('imb_sub_indeces_list = ', len(imb_sub_indeces_list))

        elif imb_factor < 1.0 and scenario is "class":
            pow = len(train_datasets) - task if imb_inverse else (task -1)
            ratio = (imb_factor**((pow*len(new_classes)) / (len(train_datasets) - 1.0)))
            num_samples = int(np.floor(ratio * len( train_dataset.sub_indeces)))
            train_dataset.sub_indeces = np.random.choice(train_dataset.sub_indeces, num_samples)
            print(len(train_dataset.sub_indeces))

        # ----------------------------------------
        # ----------------------------------------
        # for imbalanced JT: Populating ExemplarDataset for iCar
        if imb_factor < 1.0 and len(train_datasets) == 1 and add_exemplars:
            exemplars_per_class = 20
            # for each new class trained on, construct examplar-set
            for class_id in new_classes:
                start = time.time()
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            print("Constructed exemplars for iCard in imbalanced-JT")
            print("Should be checked")
            raise

        # ----------------------------------------
        # ----------------------------------------

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task>1:
            # ---------- ADHOC SOLUTION: permMNIST needs transform to tensor, while splitMNIST does not ---------- #
            if len(train_datasets)>6:
                target_transform = (lambda y, x=classes_per_task: torch.tensor(y%x)) if (
                        scenario=="domain"
                ) else (lambda y: torch.tensor(y))
            else:
                target_transform = (lambda y, x=classes_per_task: y%x) if scenario=="domain" else None
            # ---------------------------------------------------------------------------------------------------- #
            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
            exemplar_loader = iter(cycle(utils.get_data_loader(exemplar_dataset, 200, cuda=cuda, drop_last=False)))

            if sampling_strategy != 'hard_sampling':
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset
                ss_data_loader = iter(cycle(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True)))

        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))

        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        for batch_index in range(1, iters_to_use+1):
            use_vnet_for_loss = (reweighting_strategy == 'vnet') and task >= vnet_enable_from
            # ----------------------------------------------
            if reweighting_strategy == 'meta_update':
                meta_model = copy.deepcopy(model)
                meta_model.train()
                # ----- Update meta-model -----

                input_var = to_var(x_meta, requires_grad=False)
                target_var = to_var(y_meta, requires_grad=False)

                meta_model.load_state_dict(model.state_dict())

                y_f_hat = meta_model(input_var)
                l = F.cross_entropy(y_f_hat, target_var)

                meta_model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.params()), 'lr': 0.001}]
                optimizer = optim.Adam(meta_model.optim_list, betas=(0.9, 0.999))

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                y_f_updated = meta_model(input_var)
                cost = F.cross_entropy(y_f_updated, target_var, reduce=False)

                tmp = []
                for cls in range(len(np.unique(y_meta))):
                    tmp.append(cost[target_var == cls].mean())

                CE_weights = torch.FloatTensor(tmp).to(device)

            # ---------------------------------------------
            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                if training_dataset_sampler is None:
                    data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=False))
                else:
                    data_loader = iter(utils.get_data_loader(
                        training_dataset, batch_size, cuda=cuda, drop_last=True, sampler=training_dataset_sampler, shuffle=False
                    ))

                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact:
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)

            # -----------------Collect data------------------#
            if sampling_strategy=='hard_sampling' and task > 1:
                meta_model = copy.deepcopy(model)
                meta_model.train()
                # ----- Update meta-model -----
                input_var, target_var = next(ss_data_loader)
                input_var = to_var(input_var, requires_grad=False)
                target_var = to_var(target_var, requires_grad=False)

                # input_var = to_var(x_meta, requires_grad=False)
                # target_var = to_var(y_meta, requires_grad=False)

                meta_model.load_state_dict(model.state_dict())

                y_f_hat = meta_model(input_var)
                l = F.cross_entropy(y_f_hat, target_var)

                meta_model.optim_list = [{'params': filter(lambda p: p.requires_grad, meta_model.params()), 'lr': 0.001}]
                optimizer = optim.Adam(meta_model.optim_list, betas=(0.9, 0.999))

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                # exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=None)
                # exemplar_loader = iter(utils.get_data_loader(exemplar_dataset, 200, cuda=cuda, drop_last=False))
                # for batch in range(20):
                x_exemplars, y_exemplars = next(exemplar_loader)
                x_exemplars = to_var(x_exemplars, requires_grad=False)
                y_exemplars = to_var(y_exemplars, requires_grad=False)

                y_pred = meta_model(x_exemplars)
                cost = F.cross_entropy(y_pred, y_exemplars, reduce=False)

                x_top = x_exemplars[torch.topk(cost, hs_samples)[1]]
                y_top = y_exemplars[torch.topk(cost, hs_samples)[1]]
                # print(y_top)


                # print(y_top)

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                x, y = next(data_loader)                                    #--> sample training data of current task
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device

                if sampling_strategy=='hard_sampling' and x_top is not None:
                    x = torch.cat((x, x_top), 0)
                    y = torch.cat((y, y_top), 0)
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None


            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)
                        scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)
                        if scenario=="domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None




            #---> Train MAIN MODEL
            if batch_index <= iters:
                # -----------------------------------------
                if use_vnet_for_loss:
                    meta_model = copy.deepcopy(model)
                    meta_model.train()
                    vnet.train()
                    # ----- Update meta-model -----
                    # input_var = to_var(x_meta, requires_grad=False)
                    # target_var = to_var(y_meta, requires_grad=False)

                    input_var = to_var(x, requires_grad=False)
                    target_var = to_var(y, requires_grad=False)

                    # meta_model = build_model()

                    meta_model.load_state_dict(model.state_dict())

                    y_f_hat = meta_model(input_var)
                    cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
                    cost_v = torch.reshape(cost, (len(cost), 1))

                    v_lambda = vnet(cost_v.data)

                    norm_c = torch.sum(v_lambda)

                    if norm_c != 0:
                        v_lambda_norm = v_lambda / norm_c
                    else:
                        v_lambda_norm = v_lambda

                    l_f_meta = torch.sum(cost_v * v_lambda_norm)
                    meta_model.zero_grad()
                    grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                    # meta_lr = args.lr * ((0.1 ** int(iters >= 18000)) * (0.1 ** int(iters >= 19000)))  # For WRN-28-10
                    meta_lr = 0.1
                    # meta_lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
                    meta_model.update_params(lr_inner=meta_lr, source_params=grads)
                    del grads

                    # --- update vnet ---

                    input_validation, target_validation = x_meta, y_meta
                    input_validation_var = to_var(input_validation, requires_grad=False)
                    target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

                    y_g_hat = meta_model(input_validation_var)
                    l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
                    # prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

                    optimizer_c.zero_grad()
                    l_g_meta.backward()
                    optimizer_c.step()

                    # if sampling_strategy == 'vnet':
                    #     # rank examples of the budget and add to x
                    #     exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=None)
                    #     exemplar_loader = iter(utils.get_data_loader(exemplar_dataset, 50, cuda=cuda, drop_last=False))
                    #     num_batches = int(len(exemplar_dataset) / 50)
                    #     for i in range(num_batches):
                    #         input, target = next(iter(train_meta_loader))
                    #         input = to_var(input, requires_grad=False)
                    #         target = to_var(target.type(torch.LongTensor), requires_grad=False)
                    #
                    #         y_g_hat = meta_model(input)
                    #         cost_w = F.cross_entropy(y_g_hat, target_var, reduction='none')
                    #         cost_v = torch.reshape(cost_w, (len(cost_w), 1))
                    #
                    #         with torch.no_grad():
                    #             w_new = vnet(cost_v)
                    #         norm_v = torch.sum(w_new)
                    #
                    #         if norm_v != 0:
                    #             w_v = w_new / norm_v
                    #         else:
                    #             w_v = w_new
                    #
                    #         l_f = torch.sum(cost_v * w_v)
                # -----------------------------------------


                # ----- ------------------------------------
                # Train the main model with this batch


                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_,
                                                scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task, vnet=vnet,
                                                use_vnet_for_loss=use_vnet_for_loss, loss_weights = CE_weights, vnet_loss_ratio=vnet_loss_ratio)
                # -----------------------------------------

                if reweighting_strategy=='vnet':
                    loss_list.append(loss_dict['loss_current'])
                    vnet_loss_list.append(loss_dict['loss_vnet'])
                    loss_original_list.append((loss_dict['loss_original']))

                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c>0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)
                if use_vnet_for_loss and ((batch_index) % vnet_plot_freq == 0) and batch_index >1:
                    vnet_weights_dict[batch_index] = vnet.loss_weights()

            # End of if batch_index <= iters:


            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1./task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)


        ##----------> UPON FINISHING EACH TASK...
        # How is vnet trained?
        if use_vnet_for_loss:
            plot_loss_vs_weight(vnet_dir, vnet_weights_dict, task)

        # plot class weights
        if reweighting_strategy=='vnet':
            # curentdata_data_loader = iter(cycle(utils.get_data_loader(
            #     train_dataset, 128, cuda=cuda, drop_last=False, shuffle=True
            # )))

            x, y = x_meta, y_meta
            x = to_var(x, requires_grad=False)
            y = to_var(y.type(torch.LongTensor), requires_grad=False)

            y_f_hat = model(x)
            cost = F.cross_entropy(y_f_hat, y, reduce=False)

            cost_v = torch.reshape(cost.clone(), (len(cost), 1))

            v_lambda = vnet(cost_v.data)

            norm_c = torch.sum(v_lambda)

            if norm_c != 0:
                v_lambda_norm = v_lambda / norm_c
            else:
                v_lambda_norm = v_lambda

            l_f_meta = cost_v * v_lambda_norm

            for cls in new_classes:
                class_loss[cls].append(cost[y==cls].mean())
                class_weights[cls].append(v_lambda[y==cls].mean())
                class_weighted_loss[cls].append(v_lambda_norm[y==cls].mean())

            if task == len(train_datasets):
                plot_class_vs_loss(vnet_dir, class_loss, task, title='CE loss of classes', file_name='loss')
                plot_class_vs_loss(vnet_dir, class_weights, task,title='Weight of classes', file_name='weight')
                plot_class_vs_loss(vnet_dir, class_weighted_loss, task, title='Normalized weighted of classes', file_name='normalized weights')

        # plot losses
        if reweighting_strategy=='vnet' and task == len(train_datasets):
            plot_losses(vnet_dir, loss_list, vnet_loss_list, loss_original_list)

        # -------------------------------------------
        # Close progress-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario=="task" else (list(range(classes_per_task*task)) if scenario=="class" else None)
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            model.update_omega(W, model.epsilon)

        # EXEMPLARS: update exemplar sets
        if (add_exemplars or use_exemplars) or replay_mode=="exemplars":
            exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task*task)))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(classes_per_task)) if scenario=="domain" else list(range(classes_per_task*(task-1),
                                                                                              classes_per_task*task))
            for class_id in new_classes:
                start = time.time()
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
                print("Constructed exemplar-set for class {}: {} seconds".format(class_id, round(time.time()-start)))
            model.compute_means = True
            # evaluate this way of classifying on test set
            for eval_cb in eval_cbs_exemplars:
                if eval_cb is not None:
                    eval_cb(model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
