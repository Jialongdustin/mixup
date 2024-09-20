import time
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import Averager, DAverageMeter, acc_utils, count_acc, count_per_cls_acc

def get_optimizer_standard(model, args):

    optimizer = torch.optim.SGD([{'params': model.module.encoder.parameters(), 'lr': args.lr.lr_std}], 
                                momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)

    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)

    return optimizer, scheduler

def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
    num_base = args.num_base
    tl = Averager()
    ta = Averager()
    model = model.train()
    # model.module.mode = 'encoder'
    # standard classification for pretrain

    tqdm_gen = tqdm(trainloader)
 
    for i, batch in enumerate(trainloader, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits, feat, lam, gt_label, gt_label_aux = model(data, train_label, aug=True, stochastic=False)#args.stochastic
        logits = logits[:, :num_base]
        loss = F.cross_entropy(logits, gt_label)
        if gt_label_aux is not None:
            loss_main = F.cross_entropy(logits, gt_label_aux) * lam
            loss_aux = F.cross_entropy(logits, gt_label_aux) * (1 - lam)
            total_loss = loss + loss_main + loss_aux
        else:
            total_loss = loss 
        acc = count_acc(logits, gt_label)


        lrc = scheduler.get_last_lr()[0]
        # tqdm_gen.set_description(
            # 'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def standard_test(args, model, testloader, epoch, session):
    num_base = args.num_base
    num_session = args.num_session
    test_class = num_base + session * args.way
    model = model.eval()
    # model.module.mode = 'encoder'
    vl = Averager()
    va = Averager()
    da = DAverageMeter()
    ca = DAverageMeter()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits, _, _ = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            acc = count_acc(logits, test_label)
            per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            da.update(per_cls_acc)
            ca.update(cls_sample_count)
        vl = vl.item()
        va = va.item()
        da = da.average()
        ca = ca.average()
        acc_dict = acc_utils(da, num_base,num_session, args.way, session)
    print(acc_dict)
    print('epo {}, standard test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    return vl, va, acc_dict, ca
