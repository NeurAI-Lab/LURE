import argparse
import os
import random
import shutil
import time
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from pruning import Pruner

torch.multiprocessing.set_sharing_strategy("file_system")


from dataset import (Setup_RestrictedImageNet,
                     generate_anytime_cifar10_dataloader,
                     generate_anytime_cifar100_dataloader,
                     generate_anytime_res_img_dataloader,
                     generate_anytime_res_img_dataloader_few,
                     setup__cifar10_dataset, setup__cifar100_dataset)
from pruner import *
from utils import evaluate_cer, setup_model
from wb import WandBLogger

parser = argparse.ArgumentParser(description="PyTorch Anytime Training")


##################################### Dataset #################################################
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
parser.add_argument(
    "--meta_batch_size",
    type=int,
    default=5000,
    help="data number in each meta batch_size",
)
parser.add_argument("--meta_batch_number", type=int, default=10)

##################################### Architecture ############################################
parser.add_argument("--arch", type=str, default="resnet20s", help="model architecture")
parser.add_argument(
    "--imagenet_arch",
    action="store_true",
    help="architecture for imagenet size samples",
)
parser.add_argument(
    "--imagenet_path",
    type=str,
    default="../imagenet",
    help="location of the imagenet folder",
)

##################################### General setting ############################################
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument(
    "--workers", type=int, default=0, help="number of workers in dataloader"
)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
parser.add_argument(
    "--save_dir",
    help="The directory used to save the trained models",
    default=None,
    type=str,
)
parser.add_argument("-no_replay", action="store_true", help="Flag for No Replay")
parser.add_argument("-one_replay", action="store_true", help="Flag for No Replay")
parser.add_argument("-buffer_replay", action="store_true", help="Flag for No Replay")

parser.add_argument(
    "--buffer_size_train",
    default=182,
    type=int,
    help="number of Random Train examples to add in buffer",
)
parser.add_argument(
    "--buffer_size_val",
    default=182,
    type=int,
    help="number of Random Valid examples to add in buffer",
)
parser.add_argument("-snip_no_replay", action="store_true", help="Flag for No Replay")
parser.add_argument("-few_shot", action="store_true", help="Flag for No Replay")
parser.add_argument(
    "--n_shots",
    default=100,
    type=int,
    help="number of Random Valid examples to add in buffer",
)
##################################### Training setting #################################################
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate") #0.1
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
parser.add_argument("--gamma", default=1.0, type=float, help="weight decay")

parser.add_argument(
    "--epochs", default=182, type=int, help="number of total epochs to run"
)
parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")

##################################### Pruning setting #################################################
parser.add_argument(
    "--snip_size", default=0.20, type=float, help="the size for the snip"
)
parser.add_argument("--sparsity_level", default=1, type=float, help="sparsity level")
parser.add_argument("--use_snip", action="store_true",default=False, help="Flag for No Replay")
parser.add_argument("--use_randinit", action="store_true",default=False, help="Flag for No Replay")
parser.add_argument("--weight_pruning", action="store_true",default=False, help="Flag for No Replay")
parser.add_argument("--use_llf", action="store_true",default=False, help="Flag for No Replay")
parser.add_argument("--use_fc", action="store_true",default=False, help="Flag for No Replay")
parser.add_argument("--use_shrink_perturb", action="store_true", default=False, help="shrink and perturb")
parser.add_argument("--shrink", default=0.3, type=float, help="shrink weights")
parser.add_argument("--perturb", default=0.001, type=float, help="add noise")

##################################### W&B Logging setting #################################################
parser.add_argument("-wb", action="store_true", help="Flag for using W&B logging")
parser.add_argument(
    "--project_name", default="APP", type=str, help="Name of the W&B project"
)
parser.add_argument(
    "--run", default="snip_fs-fr-im", type=str, help="Name for the W&B run"
)


best_sa = 0

args = parser.parse_args()
print(args)
os.makedirs(args.save_dir, exist_ok=True)



def main():


    global args, best_sa
    args = parser.parse_args()
    print(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    model = setup_model(args)

    if args.dataset == "cifar10":
        whole_trainset = setup__cifar10_dataset(args)
    elif args.dataset == "cifar100":
        whole_trainset = setup__cifar100_dataset(args)
    elif args.dataset == "restricted_imagenet":
        whole_trainset, test_set = Setup_RestrictedImageNet(args, args.imagenet_path)



    torch.save(model.state_dict(), os.path.join(args.save_dir, "randinit.pth.tar"))

    # setup initialization and mask

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=1 #0.1
    )
    if args.wb:
        # wandb login --relogin 764541e22cebfb1d71f71825bb28708a0ef3fb65

        wandb_logger = WandBLogger(
            project_name=args.project_name,
            run_name=args.run,
            dir=args.save_dir,

            config=vars(args),
            model=model,
            params={"resume": args.resume},
        )
    else:
        wandb_logger = None

    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]
        start_state = checkpoint["state"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(
            "loading from state: {} epoch: {}, best_sa = {}".format(
                start_state, start_epoch, best_sa
            )
        )

    else:
        start_state = 1
        all_result = {}
        all_result["gen_gap"] = []
        all_result["train_ta"] = []
        all_result["val_ta"] = []
        all_result["best_sa"] = []
        all_result["gen_gap"] = []
        all_result["train_loss"] = []
        all_result["test_acc"] = []
        all_result["test_loss"] = []
        all_result["test_acc_perbatch_{}".format(str(start_state))] = []
        all_result["lr"] = []
        all_result["val_loss"] = []
        start_epoch = 0




    time_list = []
    CER = []
    CER_diff = []
    for current_state in range(start_state, args.meta_batch_number + 1):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=args.gamma #0.1
        )
        print("Current state = {}".format(current_state))
        start_time = time.time()
        if args.dataset == "cifar10":
            print("Loading cifar10 dataset in anytime setting")
            (
                train_loader,
                val_loader,
                test_loader,
                train_snip_set,
            ) = generate_anytime_cifar10_dataloader(args, whole_trainset, current_state)
        elif args.dataset == "cifar100":
            print("Loading cifar100 dataset in anytime setting")
            (
                train_loader,
                val_loader,
                test_loader,
                train_snip_set,
            ) = generate_anytime_cifar100_dataloader(
                args, whole_trainset, current_state
            )
        elif args.dataset == "restricted_imagenet":
            print("Loading Restricted Imagenet dataset in anytime setting")
            if args.meta_batch_number in [1,3,53]:
                (
                    train_loader,
                    val_loader,
                    test_loader,
                    train_snip_set,
                ) = generate_anytime_res_img_dataloader(
                    args, whole_trainset, test_set, 80565, current_state
                )

            elif args.meta_batch_number in [1,10,30,70] and args.few_shot == True:
                # Few Shot Dataloader Example
                (
                    train_loader,
                    val_loader,
                    test_loader,
                    train_snip_set,
                ) = generate_anytime_res_img_dataloader_few(
                    args, whole_trainset, test_set, 6800, current_state
                )

        # Generate Mask using SNIP
        sparsity_level = 1
        save_mask = (
            args.save_dir
            + f"/{current_state}mask_snip_{sparsity_level}.pth.tar"
        )

        if current_state == 1:
            model_load_dir = (
                args.save_dir + "/randinit.pth.tar"
            )  # 1st Meta Batch Randomly initialized model
        else:
            if args.use_randinit:
                model_load_dir = (
                        args.save_dir + "/randinit.pth.tar"
                )
            else:
                model_load_dir = args.save_dir + f"/{current_state-1}model_SA_best.pth.tar"



        model.cpu()
        # Load the Model by applying above mask
        print("loading mask from {}".format(save_mask))
        print("loading model from {}".format(model_load_dir))
        checkpoint = torch.load(model_load_dir, map_location="cpu")
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        model.cuda()
        if current_state>1:
            data_loader = DataLoader(
                train_snip_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
            )
            if args.use_snip:
                print('sniping the params')
                prune = Pruner(model,data_loader , args.device, silent=False)
                mask_file = prune.snip(args.snip_size)
                torch.save(mask_file.state_dict(), os.path.join(args.save_dir, "snip_mask_{}.pth".format(current_state-1)))
                model = reparameterize_non_sparse(args.device, model, mask_file)

            if args.use_llf:
                print('resetting last few layers')
                model = ll_reinitialize(args.device, model)

            if args.use_fc:
                print('resetting last fc layer')
                model = fc_reinitialize(args.device, model)

            if args.use_shrink_perturb:
                print('shrinking weights')
                model = shrink_perturb(args.device, model, args.shrink, args.perturb)

        # model.cuda()
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()["param_groups"][0]["lr"])
            acc, loss = train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            tacc, vloss = validate(val_loader, model, criterion)
            tstacc, tstloss = validate(test_loader, model, criterion)

            scheduler.step()

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            gen_gap = acc - tacc
            all_result["gen_gap"].append(gen_gap)
            all_result["train_ta"].append(acc)
            all_result["val_ta"].append(tacc)
            all_result["best_sa"].append(best_sa)
            all_result["train_loss"].append(loss)
            all_result["val_loss"].append(vloss)
            all_result["test_acc"].append(tstacc)
            all_result["test_loss"].append(tstloss)
            all_result["test_acc_perbatch_{}".format(str(start_state))].append(tstacc)

            all_result["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])

            save_checkpoint(
                {
                    "state": current_state,
                    "result": all_result,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_sa": best_sa,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_SA_best=is_best_sa,
                data_state=current_state,
                save_path=args.save_dir,
            )
            if wandb_logger:
                wandb_logger.log_metrics(all_result)
                wandb_logger.log_metrics({"test loss":  all_result["test_acc_perbatch_{}".format(str(start_state))],
                                          "epoch": epoch})
        # report result
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* State = {} best SA = {} Epoch = {}".format(
                current_state,
                all_result["val_ta"][val_pick_best_epoch],
                val_pick_best_epoch + 1,
            )
        )

        all_result = {}
        all_result["train_ta"] = []
        all_result["val_ta"] = []
        all_result["best_sa"] = []
        all_result["gen_gap"] = []
        all_result["train_loss"] = []
        all_result["val_loss"] = []
        all_result["test_acc"] = []
        all_result["test_loss"] = []
        all_result["test_acc_perbatch_{}".format(str(start_state))] = []
        all_result["lr"] = []
        best_sa = 0
        start_epoch = 0
        best_checkpoint = torch.load(
            os.path.join(args.save_dir, "{}model_SA_best.pth.tar".format(current_state))
        )
        print("Loading Best Weight")
        model.load_state_dict(best_checkpoint["state_dict"])
        end_time = time.time() - start_time
        print("Total time elapsed: {:.4f}s".format(end_time))
        time_list.append(end_time)
        if args.dataset == "restricted_imagenet":
            CER.append(evaluate_cer(model, args, test_loader))
        else:
            CER.append(evaluate_cer(model, args))

        if current_state != 1:
            diff = (CER[current_state - 1] - CER[current_state - 2]) / 10000
            CER_diff.append(diff)
            print("CER diff: {}".format(diff))

        # Reset LR to 0.1 after each state
        for g in optimizer.param_groups:
            g["lr"] = 0.1
        print("LR reset to 0.1")
        print(optimizer.state_dict()["param_groups"][0]["lr"])

    test_tacc, _ = validate(test_loader, model, criterion)

    print("Test Acc = {}".format(test_tacc))
    print("CER = {}".format(sum(CER)))
    wandb_logger.log_metrics({"Test/test_acc": test_tacc})
    wandb_logger.log_metrics({"Test/CER": sum(CER)})

    print("Final Test Accuracy: ")
    print(test_tacc)
    print("CER")
    print(CER)
    print("Anytime Relative Error")
    print(CER_diff)
    print("Total time")
    print(time_list)

def reparameterize_non_sparse(cfg, net, net_sparse_set):
    # Re-initialize those params that were never part of sparse set
    with torch.no_grad():
        for (name, param), mask_param in zip(net.named_parameters(), net_sparse_set.parameters()):
            if 'weight' in name : #and 'bn' not in name and 'downsample' not in name
                re_init_param = re_init_weights(param.data.shape, cfg)
                param.data = param.data * mask_param.data
                re_init_param.data[mask_param.data == 1] = 0
                param.data = param.data + re_init_param.data
    return net

def ll_reinitialize(cfg, net): #LLF
    # Re-initialize those params that were never part of sparse set
    with torch.no_grad():
        for (name, param) in (net.named_parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                if name not in ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight' ,'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight','layer2.1.conv1.weight','layer2.1.conv2.weight' ]:
                    re_init_param = re_init_weights(param.data.shape, cfg)
                    param.data = re_init_param.data

    return net

def fc_reinitialize(cfg, net): #RIFLE
    # Re-initialize those params that were never part of sparse set
    with torch.no_grad():
        for (name, param) in (net.named_parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                if 'linear' in name:
                    re_init_param = re_init_weights(param.data.shape, cfg)
                    param.data = re_init_param.data

    return net


def shrink_perturb(cfg, net, shrink, perturb ):
    # Re-initialize those params that were never part of sparse set
    with torch.no_grad():
        for param in net.parameters():
            re_init_param = re_init_weights(param.data.shape, cfg)
            param.mul_(shrink).add_(re_init_param, alpha=perturb)

    return net


def re_init_weights(shape, device, reinint_method='kaiming'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        renint_usnig_method(mask, reinint_method)
        mask = torch.squeeze(mask, 1)
    else:
        renint_usnig_method(mask, reinint_method)
    return mask

def renint_usnig_method(mask, method='kaiming'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))

    else:
        nn.init.xavier_uniform_(mask)


def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(
    state, is_SA_best, data_state, save_path, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(data_state) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath,
            os.path.join(save_path, "{}model_SA_best.pth.tar".format(data_state)),
        )


def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_dense_mask_0(net, device, value):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net

def count_parameters(model):
    total =0
    layer_idx_dict = {}
    idx = {}
    for (name, p) in model.named_parameters():
        k = p.numel()
        total+= k
        layer_idx_dict[name] = k
    keys = [k for k in layer_idx_dict.keys()]
    values = [v for v in layer_idx_dict.values()]
    for i in range(len(keys)):
        cumilative = sum(values[:i])
        idx[keys[i]] = cumilative

    return layer_idx_dict, idx   #if p.requires_grad




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
