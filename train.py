import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
import time
import matplotlib.pyplot as plt

from data import SubDataset, ExemplarDataset, MetaDataset
from continual_learner import ContinualLearner
from vnet import *
from visual_plt import plot_vnet


def train_cl(model, train_datasets, meta_datasets, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, eval_cbs_exemplars=list(), use_vnet=False, imb_factor = 1.0,
             imb_inverse= False, reset_vnet = False, reset_vnet_optim=False, vnet_enable_from = 2, vnet_exemplars_per_class = 20,
             metadataset_building_strategy = 'none'):
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
    if use_vnet:
        vnet = VNet(1, 100, 1).to(device)

        weight_dict = {}
        weight_dict[0] = vnet.loss_weights()

        plot_vnet(weight_dict, name="vnet_start")

        optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                      momentum=0.9, nesterov=True,
                                      weight_decay=5e-4)

        vnet_weights_dict = {}

    metadata_list = []


    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        # ----------- weight-net relate -----------
        if use_vnet and reset_vnet:
            vnet = VNet(1, 100, 1).to(device)
            optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                          momentum=0.9, nesterov=True,
                                          weight_decay=5e-4)

        if use_vnet and reset_vnet_optim:
            optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                          momentum=0.9, nesterov=True,
                                          weight_decay=5e-4)
            # optimizer_c = torch.optim.Adam(vnet.params(), betas=(0.9, 0.999))
        # --------------------------------------------
        new_classes = list(range(classes_per_task)) if scenario == "domain" else list(
            range(classes_per_task * (task - 1),
                  classes_per_task * task))

        if use_vnet and task >= vnet_enable_from:
            if metadataset_building_strategy == 'trainingset':
                # Adding to the meta_model dataset from the current data
                for class_id in new_classes:
                    # create new dataset containing only all examples of this class
                    current_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                    # based on this dataset, construct new exemplar-set for this class
                    model.construct_exemplar_set(dataset=current_dataset, n=vnet_exemplars_per_class, meta_data=True)

                vnet_meta_dataset = ExemplarDataset(model.meta_data, target_transform=None)

                train_meta_loader = iter(cycle(utils.get_data_loader(
                    vnet_meta_dataset, 256, cuda=cuda, drop_last=False, shuffle=True
                )))
            elif metadataset_building_strategy == 'testset':
                rand_sampler = torch.utils.data.RandomSampler(meta_datasets[task - 1], num_samples=64, replacement=True)
                train_meta_loader = iter(cycle(utils.get_data_loader(
                    meta_datasets[task-1], 64, cuda=cuda, drop_last=True, sampler=rand_sampler, shuffle=False
                )))
            elif metadataset_building_strategy == 'exemplar' and len(model.exemplar_sets)>0:
                num_classes = len(model.exemplar_sets)
                meta_ds_list = []

                for class_id in range(num_classes):
                    indxs = np.random.choice(len(model.exemplar_sets[class_id]), vnet_exemplars_per_class)
                    meta_ds_list.append(model.exemplar_sets[class_id][indxs])

                for class_id in new_classes:
                    # create new dataset containing only all examples of this class
                    current_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                    # based on this dataset, construct new exemplar-set for this class
                    model.construct_exemplar_set(dataset=current_dataset, n=vnet_exemplars_per_class, meta_data=True)

                for l in model.meta_data:
                    meta_ds_list.append(l)

                model.meta_data = []

                vnet_meta_dataset = ExemplarDataset(meta_ds_list, target_transform=None)

                train_meta_loader = iter(cycle(utils.get_data_loader(
                    vnet_meta_dataset, 256, cuda=cuda, drop_last=False, shuffle=True
                )))

        # ----------------------------------------



        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets()

        # --------------------
        # Enforce imbalanced data factor to the train_dataset


        if imb_factor < 1.0 and len(train_datasets) == 1:
            classes_sub_datasets = []
            samples_per_class = [int(np.floor(5000*((0.01)**(i / (10 - 1.0))))) for i in range(10) ]
            print("samples per classes = ", samples_per_class)
            final_dataset = None
            for cls in range(10):
                tmp = SubDataset(train_dataset,  sub_labels=[cls])
                id = 9 - cls if imb_inverse else cls
                tmp.sub_indeces = np.random.choice(tmp.sub_indeces, samples_per_class[id])
                final_dataset = tmp if final_dataset is None else ConcatDataset([final_dataset, tmp])
            print('final_dataset = ', len(final_dataset))
            train_dataset = final_dataset

        elif imb_factor < 1.0 and scenario is "class":
            pow = len(train_datasets) - task if imb_inverse else (task -1)
            ratio = (imb_factor**(pow / (len(train_datasets) - 1.0)))
            num_samples = int(np.floor(ratio * len( train_dataset.sub_indeces)))
            train_dataset.sub_indeces = np.random.choice(train_dataset.sub_indeces, num_samples)
            print(len(train_dataset.sub_indeces))

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
            training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
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

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
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

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                x, y = next(data_loader)                                    #--> sample training data of current task
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
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
                if use_vnet and task >= vnet_enable_from:
                    meta_model = copy.deepcopy(model)
                    meta_model.train()
                    vnet.train()
                    # ----- Update meta-model -----
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

                    input_validation, target_validation = next(iter(train_meta_loader))
                    input_validation_var = to_var(input_validation, requires_grad=False)
                    target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

                    y_g_hat = meta_model(input_validation_var)
                    l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
                    # prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

                    prev = vnet.linear1.weight.clone().data
                    optimizer_c.zero_grad()
                    l_g_meta.backward()
                    optimizer_c.step()
                    curr = vnet.linear1.weight.clone()
                    eqq = torch.equal(prev, curr)
                    diff = prev - curr
                    diff = torch.abs(diff).sum()
                    grad_sum = torch.abs(vnet.linear1.weight.grad).sum()
                    diff = diff


                # -----------------------------------------
                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task, vnet=vnet)

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
        if use_vnet and task >= vnet_enable_from:
            vnet_weights_dict[task]= vnet.loss_weights()

            # if it's the last task
            if task == len(train_datasets):
                plot_vnet(vnet_weights_dict)


        # -------------------------------------------




        # Close progres-bar(s)
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
