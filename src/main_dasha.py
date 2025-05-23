import os
import json
import argparse
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from timeit import default_timer
from functools import partial
import copy

from relax.nas import MixedOptimizer
from dash import MixtureSupernet
from task_configs import get_data, get_config, get_model, get_metric, get_hp_configs, get_optimizer
from task_utils import count_params, print_grad, calculate_stats
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='DEEPSEA', help='dataset name')
    parser.add_argument('--root_dir', type=str, default='./data/', help='root directory for the dataset')
    parser.add_argument('--save_dir', type=str, default='', help='save directory for the dataset (absolute path)')
    parser.add_argument('--arch', type=str, default=None, help='backbone architecture')
    parser.add_argument('--experiment_id', type=str, default='0', help='directory name to save the experiment results')
    parser.add_argument('--baseline', type=int, default=0, help='evaluate backbone without architecture search')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kernel_choices', type=int, default=[None], nargs='+', help='specify the set of kernel sizes (K)' )
    parser.add_argument('--dilation_choices', type=int, default=[None], nargs='+', help='specify the set of dilation rates (D)')
    parser.add_argument('--verbose', type=int, default=0, help='print gradients of arch params for debugging')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--valid_split', type=int, default=0, help='use train-validation-test (3-way) split')
    parser.add_argument('--reproducibility', type=int, default=0, help='exact reproducibility')
    parser.add_argument('--separable', type=int, default=1, help='use separable conv')
    parser.add_argument('--stream', type=int, default=1, help='use streaming for implementing aggconv')
    parser.add_argument('--grad_scale', type=int, default=None, help='scale gradient for arch params')
    parser.add_argument('--pool_k', type=int, default=None, help='pooling kernel size')
    parser.add_argument('--retrain_only', action='store_true', help='only retrain with given kernel pattern')
    parser.add_argument('--hp_dir', type=str, default='', help='directory containing hp grid search results')
    parser.add_argument('--split_state', action='store_true', help='split state space')

    args = parser.parse_args()

    if args.split_state:
        args.split_state = np.random.randint(1, 50)
        print("split state:", args.split_state)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_id = 'baseline' if args.baseline else args.experiment_id
    args.save_dir = os.path.join(args.save_dir, 'results_acc/'  + args.dataset + '/' + "search_init" +'/' + exp_id + "/" + str(args.seed))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.retrain_only and args.hp_dir == '':
        raise ValueError("Argument hp_dir must be specified for retraining")
    
    if args.retrain_only:
        args.arch, args.grad_scale = np.load(os.path.join(args.hp_dir, 'search_hps.npy'))

    with open(args.save_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print("------- Experiment Summary --------")
    print(args.__dict__)

    torch.cuda.empty_cache()
    set_random_seed(args.seed)

    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, weight_sched_hpo, accum, clip, retrain_clip, validation_freq, retrain_freq, \
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(args.dataset, args)  
    
    arch = args.arch if not(args.arch is None) else arch_default
    if config_kwargs['arch_retrain_default'] is not None:
        arch_retrain = config_kwargs['arch_retrain_default']
    else:
        arch_retrain = arch

    if arch_retrain == "unet2d" or arch_retrain == "unet2d_small":
        optimizer_type = "AdamW"
    else:
        optimizer_type = "SGD"
    
    if args.baseline:
        arch = arch if len(args.arch) > 0 else arch_retrain

    kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
    dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.root_dir, args.dataset, batch_size, arch, args.valid_split)
    model = get_model(arch, sample_shape, num_classes, config_kwargs)
    metric, compare_metrics = get_metric(args.dataset)
    
    train_score, train_time, retrain_score, retrain_time, param_values_list, prev_param_values = [], [], [], [], [], None

    model = MixtureSupernet.create(model.cpu(), in_place=True)

    if not args.baseline:
        model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable, 
            stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)

        if dims == 1:
            model.remove_module("chomp")
        opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay), arch_opt(model.arch_params(), lr=arch_lr, weight_decay=weight_decay)]

    else:
        opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay)]
        epochs = retrain_epochs
        weight_sched_search = weight_sched_train
        clip = retrain_clip
    
    optimizer = MixedOptimizer(opts)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
    lr_sched_iter = arch == 'convnext'
    
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    n_train_temp = int(quick_search * n_train) + 1 if quick_search < 1 else n_train

    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()
    
    if not args.retrain_only:
        print("search arch:", arch, "\tretrain arch:", arch_retrain)
        print("batch size:", batch_size, "\tlr:", lr, "\tarch lr:", arch_lr)
        print("arch configs:", config_kwargs)
        print("kernel choices:", kernel_choices_default, "\tdilation choices:", dilation_choices_default)
        print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)

        print("\n------- Start Arch Search --------")
        print("param count:", count_params(model))
    else:
        epochs = 1
    for ep in range(epochs):
        if not args.retrain_only:
            time_start = default_timer()

            train_loss = train_one_epoch(model, optimizer, scheduler, args.device, train_loader, loss, clip, 1, n_train_temp, decoder, transform, lr_sched_iter, scale_grad=not args.baseline)

            if args.verbose and not args.baseline and (ep + 1) % args.print_freq == 0:
                print_grad(model, kernel_choices, dilation_choices)

            if ep % validation_freq == 0 or ep == epochs - 1: 
                if args.baseline:
                    val_loss, val_score = evaluate(model, args.device, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)
                    train_score.append(val_score)

                    if (ep + 1) % args.print_freq == 0  or ep == epochs - 1: 
                        print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (default_timer() - time_start), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

                    np.save(os.path.join(args.save_dir, 'train_score.npy'), train_score)
                
                elif (ep + 1) % args.print_freq == 0 or ep == epochs - 1:
                    print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (default_timer() - time_start), "\ttrain loss:", "%.4f" % train_loss)

                time_end = default_timer()
                train_time.append(time_end - time_start)
                np.save(os.path.join(args.save_dir, 'train_time.npy'), train_time)
        if not args.baseline and ((ep + 1) % retrain_freq == 0 or ep == epochs - 1):
            param_values, ks, ds = [], [], []

            if args.retrain_only:
                ks, ds = np.load(os.path.join(args.hp_dir, 'searched_kernel_pattern.npy'))
                ks = ks.tolist()
                ds = ds.tolist()
            else:
                for name, param in model.named_arch_params():
                    param_values.append(param.data.argmax(0))
                    if args.verbose:
                        print(name, param.data)
                    
                    ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
                    ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

                param_values = torch.stack(param_values, dim = 0)

            print("[searched kernel pattern] ks:", ks, "\tds:", ds)
            del model
            del train_loader, val_loader, test_loader
            if prev_param_values is not None and torch.equal(param_values, prev_param_values):
                print("\n------- Arch Search Converge --------")
            else:
                print("\n------- Start Hyperparameter Search --------")

                if os.path.isfile(os.path.join(args.save_dir, 'network_hps.npy')) or args.retrain_only:
                    hp_dir = args.hp_dir if args.retrain_only else args.save_dir
                    lr, drop_rate, weight_decay, momentum = np.load(os.path.join(hp_dir, 'network_hps.npy'))
                    print("[load hp] bs = ", batch_size, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
                else:
                    best_results = initialize_asha_search(arch_retrain, sample_shape, num_classes, config_kwargs, ks, ds, decoder, transform, args, optimizer_type, weight_sched_hpo, lr_sched_iter, loss, metric, retrain_clip, compare_metrics, batch_size, n_train)

                    random_seed = best_results.config['random_seed']
                    lr = best_results.config['lr']
                    drop_rate = best_results.config['drop_rate']
                    weight_decay = best_results.config['weight_decay']
                    momentum = best_results.config['momentum']

                    np.save(os.path.join(args.save_dir, 'network_hps.npy'), (random_seed, lr, drop_rate, weight_decay, momentum))
                    print("[selected hp] random_seed: ", random_seed, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
                    print("[best val score]", best_results.metrics['val_score'])
                    for file in os.listdir(args.save_dir):
                        if file.endswith(".pt") and not file.startswith(f'init_{random_seed}_{lr}_{drop_rate}_{weight_decay}_{momentum}'):
                            os.remove(os.path.join(args.save_dir, file))

                print("\n------- Start Retrain --------")
                retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
                if args.split_state:
                    retrain_train_loader, retrain_val_loader, retrain_test_loader, retrain_n_train, retrain_n_val, retrain_n_test, data_kwargs = get_data(args.root_dir, args.dataset, accum * batch_size, arch_retrain, True, args.split_state)
                else:
                    retrain_train_loader, retrain_val_loader, retrain_test_loader, retrain_n_train, retrain_n_val, retrain_n_test, data_kwargs = get_data(args.root_dir, args.dataset, accum*batch_size, arch_retrain, False)
                retrain_n_temp = int(quick_retrain * retrain_n_train) + 1 if quick_retrain < 1 else retrain_n_train

                retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
                retrain_model = retrain_model.to(args.device)

                init_dir = args.hp_dir if args.retrain_only else args.save_dir
                retrain_model.load_state_dict(torch.load(os.path.join(init_dir, f'init_{random_seed}_{lr}_{drop_rate}_{weight_decay}_{momentum}.pt')))
                retrain_optimizer = get_optimizer(momentum = momentum, weight_decay = weight_decay, type=optimizer_type)(retrain_model.parameters(), lr = lr)
                retrain_model.set_arch_requires_grad(False)

                retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda = weight_sched_train)
            
                time_retrain = 0
                score = []
                print("param count:", count_params(retrain_model))
                for retrain_ep in range(retrain_epochs):
                    retrain_time_start = default_timer()
                    retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, retrain_train_loader, loss, retrain_clip, 1, retrain_n_temp, decoder, transform, lr_sched_iter)

                    if retrain_ep % validation_freq == 0 or retrain_ep == retrain_epochs - 1:
                        retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, retrain_val_loader, loss, metric, retrain_n_val, decoder, transform, fsd_epoch=retrain_ep if args.dataset == 'FSD' else None)

                        retrain_time_end = default_timer()
                        time_retrain += retrain_time_end - retrain_time_start
                        score.append(retrain_val_score)

                        if compare_metrics(score) == retrain_val_score:
                            try:
                                retrain_model.save_arch(os.path.join(args.save_dir, 'arch.th'))
                                torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, 'network_weights.pt'))
                                np.save(os.path.join(args.save_dir, 'retrain_score.npy'), retrain_score)
                                np.save(os.path.join(args.save_dir, 'retrain_time.npy'), retrain_time)
                            except AttributeError:
                                pass
                        
                        if (retrain_ep + 1) % args.print_freq == 0  or retrain_ep == retrain_epochs - 1: 
                            print("[retrain", retrain_ep, "%.6f" % retrain_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (retrain_time_end - retrain_time_start), "\ttrain loss:", "%.4f" % retrain_loss, "\tval loss:", "%.4f" % retrain_val_loss, "\tval score:", "%.4f" % retrain_val_score, "\tbest val score:", "%.4f" % compare_metrics(score))
                            
                        if retrain_ep == retrain_epochs - 1:
                            retrain_score.append(score)
                            retrain_time.append(time_retrain)

                            np.save(os.path.join(args.save_dir, 'retrain_score.npy'), retrain_score)
                            np.save(os.path.join(args.save_dir, 'retrain_time.npy'), retrain_time)

                print("\n------- Start Test --------")
                test_scores = []
                test_model = retrain_model
                test_time_start = default_timer()
                test_loss, test_score = evaluate(test_model, args.device, retrain_test_loader, loss, metric, retrain_n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
                test_time_end = default_timer()
                test_scores.append(test_score)

                print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

                test_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'network_weights.pt')))
                test_time_start = default_timer()
                test_loss, test_score = evaluate(test_model, args.device, retrain_test_loader, loss, metric, retrain_n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
                test_time_end = default_timer()
                test_scores.append(test_score)

                print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
                np.save(os.path.join(args.save_dir, 'test_score.npy'), test_scores)
                print("====================================================================================")


def asha_inner_search(tune_config, arch_retrain, sample_shape, num_classes, config_kwargs, ks, ds, decoder, transform, args, optimizer_type, weight_sched_hpo, lr_sched_iter, loss, metric, search_epochs, subsampling_ratio, retrain_clip, batch_size, n_train, compare_metrics):
    if args.split_state:
        search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.root_dir, args.dataset, batch_size, arch_retrain, True, args.split_state)
    else:
        search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.root_dir, args.dataset, batch_size, arch_retrain, True)

    random_seed = tune_config['random_seed']
    lr = tune_config['lr']
    drop_rate = tune_config['drop_rate']
    weight_decay = tune_config['weight_decay']
    momentum = tune_config['momentum']

    search_n_temp = int(subsampling_ratio * search_n_train) + 1
    set_random_seed(random_seed)

    retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
    retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
    retrain_model = retrain_model.to(args.device)
    retrain_model.set_arch_requires_grad(False)
    model_base = copy.deepcopy(retrain_model)
    retrain_optimizer = get_optimizer(momentum=momentum, weight_decay=weight_decay, type=optimizer_type)(retrain_model.parameters(), lr=lr)
    retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda=weight_sched_hpo)
    torch.save(model_base.state_dict(), os.path.join(args.save_dir, f'init_{random_seed}_{lr}_{drop_rate}_{weight_decay}_{momentum}.pt'))
    
    best_val_score = -100.0 if compare_metrics == np.max else 100.0
    for _ in range(search_epochs):
        retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, search_train_loader, loss, retrain_clip, 1, search_n_temp, decoder, transform, lr_sched_iter)
        retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
        best_val_score = compare_metrics([retrain_val_score, best_val_score])
        train.report({"val_score": best_val_score, "random_seed": random_seed, "lr": lr, "drop_rate": drop_rate, "weight_decay": weight_decay, "momentum": momentum})
    del retrain_model


def initialize_asha_search(arch_retrain, sample_shape, num_classes, config_kwargs, ks, ds, decoder, transform, args, optimizer_type, weight_sched_hpo, lr_sched_iter, loss, metric, retrain_clip, compare_metrics, batch_size, n_train):
    search_space, search_epochs, subsampling_ratio = get_hp_configs(args.dataset, n_train, arch_retrain)
    search_program = partial(asha_inner_search, arch_retrain=arch_retrain, sample_shape=sample_shape, num_classes=num_classes, config_kwargs=config_kwargs, ks=ks, ds=ds, 
                             decoder=decoder, transform=transform, args=args, optimizer_type=optimizer_type, weight_sched_hpo=weight_sched_hpo, lr_sched_iter=lr_sched_iter, 
                             loss=loss, metric=metric, search_epochs=search_epochs, subsampling_ratio=subsampling_ratio, retrain_clip=retrain_clip, batch_size=batch_size, 
                             n_train=n_train, compare_metrics=compare_metrics)

    mode = "max" if compare_metrics == np.max else "min"
    search_with_resources = tune.with_resources(search_program, {"cpu": 6, "gpu": 1})
    
    tuner = tune.Tuner(
        search_with_resources,
        tune_config=tune.TuneConfig(
            num_samples=120,
            scheduler=ASHAScheduler(metric="val_score", mode=mode, reduction_factor=2.0, max_t=20),
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_score", mode=mode)
    return best_result


def train_one_epoch(model, optimizer, scheduler, device, loader, loss, clip, accum, temp, decoder=None, transform=None, lr_sched_iter=False, min_lr=-1, scale_grad=False, fix_order=None):
    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    if fix_order != None:
        torch.manual_seed(fix_order)

    for i, data in enumerate(loader):
        if transform is not None:
            x, y, z = data
            z = z.to(device)
        else:
            x, y = data 
        
        x, y = x.to(device), y.to(device)
            
        out = model(x)

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)
                        
        l = loss(out, y)
        l.backward()

        if scale_grad:
            model.scale_grad()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += l.item()

        if lr_sched_iter and optimizer.param_groups[0]['lr'] > min_lr:
            scheduler.step()

        if i >= temp - 1:
            break

    if (not lr_sched_iter) and optimizer.param_groups[0]['lr'] > min_lr:
        scheduler.step()

    return train_loss / temp


def evaluate(model, device, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0
    
    if fsd_epoch is None:
        with torch.no_grad():
            for i, data in enumerate(loader):
                if transform is not None:
                    x, y, z = data
                    z = z.to(device)
                else:
                    x, y = data
                                    
                x, y = x.to(device), y.to(device)
                out = model(x)
                
                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)
                
                outs.append(out) 
                ys.append(y) 
                n_data += x.shape[0]
            
                if i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys).item()
                    
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

        eval_loss /= n_eval
        eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                if fsd_epoch < 100:
                    if ix > 2000: break

                x, y = loader[ix]
                x, y = x.to(device), y.to(device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score


if __name__ == '__main__':
    main()