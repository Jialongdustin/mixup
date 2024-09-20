# import new Network name here and add in model_class args
from .Network import MYNET
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F


def base_train_pit(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    treg = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        # 为方便操作reshape
        samples_per_cls = args.episode.episode_shot + args.episode.episode_query
        data = data.view(samples_per_cls, args.episode.episode_way, -1)
        train_label = train_label.view(samples_per_cls, args.episode.episode_way)
        audio_samples = data.size(-1)
        #
        base_data = data[:, :args.episode.base, :].reshape(-1, audio_samples)
        base_feat, _ = model.module.encode(base_data)
        base_lb = train_label[:, :args.episode.base].reshape(-1)
        # 将10个基类分成两组mixup合成新类
        
        syn_new_data = data[:, args.episode.base:, :].view(samples_per_cls, 2,args.episode.syn_new,-1)
        syn_new_label = train_label[:, args.episode.base:].view(samples_per_cls, 2,args.episode.syn_new,-1)
        lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        mixed_data = lam * syn_new_data[:, 0, :, :] + (1 - lam) * syn_new_data[:, 1, :, :]
        mixed_data = mixed_data.reshape(-1, audio_samples)
        syn_new_label_ori = syn_new_label[:, 0, :].reshape(-1) # 
        syn_new_label_aux = syn_new_label[:, 1, :].reshape(-1)
        mixed_feat, _ = model.module.encode(mixed_data)
        syn_proto = mixed_feat.view(samples_per_cls, args.episode.syn_new, -1)[:args.episode.episode_shot, :, :].mean(0)
        # 
        picked_new_cls=torch.Tensor(np.random.choice(args.num_all-args.num_base,5,replace=False) + args.num_base).long()
        # start_cls = np.random.choice(args.num_all-args.num_base,1,replace=False) + args.num_base
        # start_cls = start_cls if start_cls < args.num_all - args.episode.syn_new else args.num_all - args.episode.syn_new - 1
        # picked_new_cls=torch.Tensor(np.arange(start_cls, start_cls + args.episode.syn_new)).long().cuda()
        novel_mask=torch.Tensor(np.zeros((args.num_all - args.num_base, model.module.num_features)))
        novel_mask[picked_new_cls - args.num_base, :] = syn_proto.cpu()
        model.module.fc.mu.data[args.num_base:, :] = novel_mask.cuda()
        base_logits = model.module.fc(base_feat, args.stochastic)
        mixed_logits = model.module.fc(mixed_feat[args.episode.episode_shot * args.episode.syn_new:, :], args.stochastic)
        syn_lbs = torch.tile(picked_new_cls, (args.episode.episode_query, )).cuda()
        logits_ = torch.cat([base_logits, mixed_logits], dim=0)
        labels = torch.cat([base_lb, syn_lbs], dim=0)
        total_loss = F.cross_entropy(logits_, labels) 
        acc = count_acc(logits_, labels)
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    treg = 0
    return tl, ta, treg


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    treg = Averager()
    model = model.train()
    
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
    
        logits, _, _ = model(data, stochastic = False) # True
        logits = logits[:, :args.num_base]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    treg = 0
    return tl, ta, treg

def replace_base_fc(trainset, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding, _ = model(data, stochastic = False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.num_base):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu.data[:args.num_base] = proto_list

    return model

def replace_fc(trainset, model, args, session):
    present_class = (args.base_class + session * args.way)
    previous_class = (args.base_class + (session-1) * args.way)
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            # data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
            # data = data.view(-1, 3, 32, 32)
            # label = torch.stack([label * 4 + k for k in range(4)], 1).view(-1)
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(previous_class, present_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu[previous_class:present_class] = proto_list

    return model

def update_sigma_protos_feature_output(trainloader, trainset, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    # trainloader.dataset.transform = transform
    
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            #print(data.shape)
            #model.module.mode = 'encoder'
            _,embedding, _ = model(data, stochastic=False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    if session == 0:
        
        for class_index in range(args.num_base):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        
        args.radius = np.sqrt(np.mean(radius)) 
        args.proto_list = torch.stack(proto_list, dim=0)
    else:
        for class_index in  np.unique(trainset.targets):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)

def update_sigma_novel_protos_feature_output(support_data, support_label, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        data, label = support_data, support_label
        #model.module.mode = 'encoder'
        _,embedding, _ = model(data, stochastic=False)

        embedding_list.append(embedding.cpu())
        label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    assert session > 0
    for class_index in  support_label.cpu().unique():

        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
        #print('dim of emd', embedding_this.shape)
        #print(c)
        feature_class_wise = embedding_this.numpy()
        cov = np.cov(feature_class_wise.T)
        radius.append(np.trace(cov)/64)
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)
        
def test(model, testloader, epoch, args, session, print_numbers=False, save_pred=False):
    test_class = args.num_base + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    da = DAverageMeter()
    ca = DAverageMeter()
    pred_list = []
    label_list = []
    with torch.no_grad():
        #tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]
            logits, features, _ = model(data, stochastic = False)
            logits = logits[:, :test_class]
            pred = torch.argmax(logits, dim=1)
            if session == args.num_session - 1:
                pred_list.append(pred)
                label_list.append(test_label)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
 
            vl.add(loss.item())
            va.add(acc)
            per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
            da.update(per_cls_acc)
            ca.update(cls_sample_count)        

        vl = vl.item()
        va = va.item()
        da = da.average()
        ca = ca.average()
        acc_dict = acc_utils(da, args.num_base, args.num_session, args.way, session)
    if print_numbers:
        print(acc_dict)         
    return vl, va, acc_dict


def save_features(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    class_means = model.module.fc.mu.data[:test_class*4:4, :].detach()
    class_means = F.normalize(class_means, p=2, dim=-1)
    data_features = []
    labels = []
    predictions = []

    with torch.no_grad():
        #tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)# teja
            logits, features, _ = model(data, stochastic = False)
            logits_original = logits[0::4, :test_class*4:4]
            loss = F.cross_entropy(logits_original, test_label)
            acc = count_acc(logits_original, test_label)
            vl.add(loss.item())
            va.add(acc)

            data_features.append(features[::4])
            labels.append(test_label)
            predictions.append(torch.argmax(logits, dim=1))

        vl = vl.item()
        va = va.item()

    data_features = torch.stack(data_features).view(-1, 64)
    data_features = F.normalize(data_features, p=2, dim=-1)
    labels = torch.stack(labels).view(-1,1)
    predictions = torch.stack(predictions).view(-1,1)     
    
    with open(args.save_path+'/S3C_features_session_'+ str(session)+'.npy', 'wb' ) as f:
        np.save(f, class_means.cpu().detach().numpy())
        #np.save(f, Glove_inputs.cpu().detach().numpy())
        np.save(f, data_features.cpu().detach().numpy())
        np.save(f, labels.cpu().detach().numpy())
        np.save(f, predictions.cpu().detach().numpy())

def mixup_feat(feat, gt_labels, alpha=1.0):
    if alpha > 0:
        lam = alpha
    else:
        lam = 0.5

    batch_size = feat.size()[0]
    index = torch.randperm(batch_size).to(device=feat.device)

    mixed_feat = lam * feat + (1 - lam) * feat[index, :]
    gt_a, gt_b = gt_labels, gt_labels[index]

    return mixed_feat, gt_a, gt_b, lam