import argparse
import shutil
import numpy as np
from datetime import datetime
import torch
import yaml
import time
from copy import deepcopy


from sympy.physics.units import frequency
from utils.losses import SupConLoss
#from modelscope.models.cv.action_detection.modules.action_detection_pytorch import build_action_detection_model
#from datashader import first
from prompt_toolkit import prompt
from tqdm import tqdm
from helper import Helper
from sklearn import preprocessing
import PIL.Image as Image
import cv2
from tensorflow.keras.datasets import cifar10
from hashlib import md5
from utils.utils import *
import torch.nn as nn
from tasks.batch import Batch
from torch.utils.data import Dataset, DataLoader,RandomSampler,TensorDataset,random_split
from torch.autograd import Variable
import copy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from torch import optim
import torch.nn.functional as F
logger = logging.getLogger('logger')
import torch.optim.lr_scheduler as lr_scheduler
import random
from models.resnet import SupConResNet18
from PIL import Image
import torchvision.transforms as transforms
import os
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
batch_size=64


import numpy as np
import torch


# 固定种子函数
from pytorch_lightning import seed_everything

# Set seed
seed = 42
seed_everything(seed)


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.to(device)
    else:
        return tensor
def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / torch.norm(v))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v

def cifar100_trigger(helper, target_model, noise_trigger,intinal_trigger):
    logger.info("start trigger fine-tuning")
    init = False
    # load model
    model = copy.deepcopy(target_model)
    model.eval()
    pre_trigger = torch.tensor(noise_trigger).to(device)
    aa = copy.deepcopy(intinal_trigger).to(device)
    for e in range(1):
        corrects = 0
        datasize = 0
        for poison_id in range(helper.task.params.fl_number_of_adversaries):

            data_iterator = helper.task.fl_train_loaders[poison_id]
            for batch_id, (datas, labels) in enumerate(data_iterator):
                datasize += len(datas)
                x = Variable(cuda(datas, True))
                y = Variable(cuda(labels, True))
                y_target = torch.LongTensor(y.size()).fill_(int(helper.task.params.backdoor_label))
                y_target = Variable(cuda(y_target, True), requires_grad=False)
                if not init:
                    noise = copy.deepcopy(pre_trigger)
                    noise = Variable(cuda(noise, True), requires_grad=True)
                    init = True
                for index in range(0, len(x)):
                    for i in [0, 4]:

                        for j in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]:
                            if helper.task.params.task=='MNIST':

                                x[index][0][i][j]=noise[i][j]
                            else:
                                x[index][0][i][j] = noise[0][i][j]
                                x[index][1][i][j] = noise[0][i][j]
                                x[index][2][i][j] = noise[0][i][j]

                output = model((x).float())
                classloss = nn.functional.cross_entropy(output, y_target)
                loss = classloss
                model.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)
                loss.backward(retain_graph=True)

                noise = noise - noise.grad * 0.1

                if helper.task.params.task=='MNIST':
                    mean=torch.tensor(0.1307)
                    std=torch.tensor(0.3081)
                else:
                    mean = torch.tensor([0.4914, 0.4822, 0.4465])
                    std = torch.tensor([0.2023, 0.1994, 0.2010])
                for i in range(len(noise)):
                    for j in range(len(noise[0])):
                        if i in [0, 4] and j in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]:
                            continue
                        else:
                            if helper.task.params.task=='MNIST':
                                noise[i][j]=(0-mean)/std
                            else:
                                noise[0][i][j] = (0-mean[0])/std[0]
                                noise[1][i][j] = (0-mean[1])/std[1]
                                noise[2][i][j] = (0-mean[2])/std[2]

                delta_noise = noise - aa
                noise = aa + proj_lp(delta_noise, 10, 2)

                noise = Variable(cuda(noise.data, True), requires_grad=True)

    if helper.task.params.task=='MNIST':
        noise=torch.clamp(noise,-3,3)
    else:
        noise=torch.clamp(noise,-2,2)
    return noise
def poison_per_batch(task,images,labels,evaluation,noise_trigger,user_id):

    new_images = images
    new_targets = labels

    for index in range(0, len(images)):
        if evaluation:  # poison all data when testing
            new_targets[index] = task.params.backdoor_label

            new_images[index] = add_pixel_pattern(images[index], noise_trigger,task,user_id)


        else:  # poison part of data when training
            if index < batch_size*task.params.poisoning_proportion:
                new_targets[index] = task.params.backdoor_label
                new_images[index] = add_pixel_pattern(images[index], noise_trigger,task,user_id)

    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets

def model_cosine_similarity(model, target_params_variables,
                            ):

    cs_list = list()
    cs_loss = torch.nn.CosineSimilarity(dim=0)
    for name, data in model.named_parameters():
        if name == 'decoder.weight':
            continue

        model_update = (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
            name].view(-1)

        cs = F.cosine_similarity(model_update,
                                 target_params_variables[name].view(-1), dim=0)
        cs_list.append(cs)
    return sum(cs_list) / len(cs_list)
def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var = sum_var.to(device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)
def transform_selected_images(image_path):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        raise ValueError("No images found in the specified directory.")
    selected_image = random.choice(image_files)
    selected_image_path = os.path.join(image_path, selected_image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    image = Image.open(selected_image_path).convert('RGB')  # 确保图片是RGB格式
    transformed_image = transform(image)
    return transformed_image
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, global_model=None):
    criterion = hlpr.task.criterion

    model.to(device)
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        optimizer.step()

        if i == hlpr.params.max_batch_id:
            break
    return


def get_one_vec(model_or_state_dict):
    # Check if the input is a model or a state_dict
    if isinstance(model_or_state_dict, torch.nn.Module):
        state_dict = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        raise ValueError("Input must be a PyTorch model or state_dict.")

    size = sum(p.numel() for p in state_dict.values())
    sum_var = torch.zeros(size, device=device)
    index = 0
    for name, param in state_dict.items():
        #if 'fc' in name:
            numel = param.numel()
            sum_var[index:index + numel] = param.view(-1)
            index += numel

    return sum_var

def get_one_vec_variable(task,model, variable=False):
    size = 0
    if task.params.task=='GTSRB':
        s='fc2'
    else:
        s='fc'
    for name, layer in model.named_parameters():
        if s in name:
            size += layer.view(-1).shape[0]
    if variable:
        sum_var = Variable(torch.cuda.FloatTensor(size,device=device).fill_(0))
    else:
        sum_var = torch.cuda.FloatTensor(size,device=device).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        if s in name:
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

    return sum_var

def compute_cos_sim_loss(local_model,task,client_id):
    loss=0
    global_model=task.model
    global_vec = get_one_vec_variable(task, global_model, False)
    for i in range(client_id):

        local_vec = get_one_vec_variable(task,local_model,True)

        update_vec=local_vec-global_vec
        updates_name = f'{task.params.folder_path}/saved_updates/update_{i}.pth'
        loaded_params = torch.load(updates_name)
        other_model = deepcopy(global_model)
        other_model.load_state_dict(loaded_params)
        other_vec=get_one_vec_variable(task,other_model,False)
        cs_sim=F.cosine_similarity(update_vec,other_vec,dim=0)
        cs_sim=cs_sim **2
        loss+=cs_sim
    return loss

def compute_cos_sim_loss_1(local_model,task,client_id,fake_model):
    loss=0

    global_model=task.model
    global_vec = get_one_vec_variable(task, global_model, False)
    local_vec = get_one_vec_variable(task, local_model, True)
    update_vec = local_vec - global_vec
    for i in range(client_id):

        updates_name = f'{task.params.folder_path}/saved_updates/update_{i}.pth'
        loaded_params = torch.load(updates_name)
        other_model = deepcopy(global_model)
        other_model.load_state_dict(loaded_params)
        other_vec=get_one_vec_variable(task,other_model,False)
        cs_sim=F.cosine_similarity(update_vec,other_vec,dim=0)
        cs_sim=(cs_sim) **2
        loss+=cs_sim


    fake_vec=get_one_vec_variable(task,fake_model,False)
    fake_norm_update_vec=fake_vec-global_vec
    cs_sim = F.cosine_similarity(update_vec,fake_norm_update_vec,dim=0)
    cs_sim=(cs_sim)**2
    loss+=cs_sim

    return loss

def compute_cos_sim(local_model,global_model):
    local_vec=get_one_vec(local_model)
    global_vec=get_one_vec(global_model)
    cs_sim=F.cosine_similarity(local_vec,global_vec,dim=0)
    return cs_sim
def compute_euclidean_loss(model,fixed_model):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size, device=device).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (layer - fixed_model.state_dict()[name]).view(-1)
        size += layer.view(-1).shape[0]
    loss = torch.norm(sum_var, p=2)
    return loss
def get_adv_model(model, dl, trigger, mask):
    adv_model = copy.deepcopy(model)
    adv_model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
    for _ in range(5):
        for inputs, labels in dl:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = trigger*mask +(1-mask)*inputs
            outputs = adv_model(inputs)
            loss = ce_loss(outputs, labels)
            adv_opt.zero_grad()
            loss.backward()
            adv_opt.step()

    sim_sum = 0.
    sim_count = 0.
    cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    for name in dict(adv_model.named_parameters()):
        if 'conv' in name:
            sim_count += 1
            sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                dict(model.named_parameters())[name].grad.reshape(-1))
    return adv_model, sim_sum/sim_count
def search_trigger(model, dl,hlpr,trigger,mask):
    model.eval()
    adv_models = []
    adv_ws = []

    def val_asr(model, dl, t, m):
        ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        correct = 0.
        num_data = 0.
        total_loss = 0.
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = t * m + (1 - m) * inputs
                labels[:] = hlpr.task.params.backdoor_label
                output = model(inputs)
                loss = ce_loss(output, labels)
                total_loss += loss
                pred = output.data.max(1)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        asr = correct / num_data
        return asr, total_loss

    ce_loss = torch.nn.CrossEntropyLoss()
    alpha = 0.01

    K = 200
    t = trigger.clone()
    m = mask.clone()
    normal_grad = 0.
    count = 0
    for iter in range(K):
        if iter % 10 == 0:
            asr, loss = val_asr(model, dl, t, m)
        if iter % 1 == 0 and iter != 0:
            if len(adv_models) > 0:
                for adv_model in adv_models:
                    del adv_model
            adv_models = []
            adv_ws = []
            for _ in range(1):
                adv_model, adv_w = get_adv_model(model, dl, t, m)
                adv_models.append(adv_model)
                adv_ws.append(adv_w)

        for inputs, labels in dl:
            count += 1
            t.requires_grad_()
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = t * m + (1 - m) * inputs
            labels[:] = hlpr.task.params.backdoor_label
            outputs = model(inputs)
            loss = ce_loss(outputs, labels)

            if len(adv_models) > 0:
                for am_idx in range(len(adv_models)):
                    adv_model = adv_models[am_idx]
                    adv_w = adv_ws[am_idx]
                    outputs = adv_model(inputs)
                    nm_loss = ce_loss(outputs, labels)
                    if loss == None:
                        loss = 0.01 * adv_w * nm_loss / 1
                    else:
                        loss += 0.01 * adv_w * nm_loss / 1
            if loss != None:
                loss.backward()
                normal_grad += t.grad.sum()
                new_t = t - alpha * t.grad.sign()
                t = new_t.detach_()
                t = torch.clamp(t, min=-2, max=2)
                t.requires_grad_()
    t = t.detach()
    return t,m
def poison_input(hlpr,trigger,mask,inputs,labels, eval=False):
    if eval:
        bkd_num = inputs.shape[0]
    else:
        bkd_num = int(hlpr.task.params.poisoning_proportion * inputs.shape[0])
    inputs[:bkd_num] = trigger*mask + inputs[:bkd_num]*(1-mask)
    labels[:bkd_num] = hlpr.task.params.backdoor_label
    return inputs, labels







def copy_params(own_state, state_dict):
    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param.clone())

def test(hlpr: Helper, epoch, backdoor=False, model=None):
    if model is None:
        model = hlpr.task.model
    
    model.to(device)
    model.eval()
    hlpr.task.reset_metrics()
    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)
            inputs,labels=batch.inputs,batch.labels
            inputs=inputs.to(device)
            labels=labels.to(device)
            #inputs,labels=inputs.to(device),labels.to(device)
            outputs = model(inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric


def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants,idx = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        if user.compromised:
            optimizer = optim.SGD(local_model.parameters(),
                                              lr=0.055,
                                              weight_decay=0.0005,
                                              momentum=0.9)
            if not user.user_id == 0:
                continue
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, global_model=global_model)

        else:
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        local_update = hlpr.attack.get_fl_update(local_model, global_model)



        hlpr.save_update(model=local_update, userID=user.user_id)
        if user.compromised:
            hlpr.attack.local_dataset = deepcopy(user.train_loader)

    hlpr.attack.perform_attack(global_model, epoch)

    hlpr.defense.aggr(weight_accumulator, global_model,idx)

    hlpr.task.update_global_model(weight_accumulator, global_model)
def add_pixel_pattern( ori_image, noise_trigger,task,user_id):
    image = copy.deepcopy(ori_image)
    noise = torch.tensor(noise_trigger).cpu()
    poison_patterns = []
    _0poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    _1poison_pattern = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
    _2poison_pattern = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
    _3poison_pattern = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
    poison_patterns = poison_patterns + _0poison_pattern + _1poison_pattern + _2poison_pattern + _3poison_pattern
    if user_id==5:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            if task.params.task=='MNIST':
                image[0][pos[0]][pos[1]]=noise[pos[0]][pos[1]]
            else:
                image[0][pos[0]][pos[1]] = noise[0][pos[0]][pos[1]]
                image[1][pos[0]][pos[1]] = noise[1][pos[0]][pos[1]]
                image[2][pos[0]][pos[1]] = noise[2][pos[0]][pos[1]]
    else:
        pos=poison_patterns[user_id]
        if task.params.task=='MNIST':
            if task.params.task=='MNIST':
                image[0][pos[0]][pos[1]]=noise[pos[0]][pos[1]]
            else:
                image[0][pos[0]][pos[1]] = noise[0][pos[0]][pos[1]]
                image[1][pos[0]][pos[1]] = noise[1][pos[0]][pos[1]]
                image[2][pos[0]][pos[1]] = noise[2][pos[0]][pos[1]]



    return image
def poison_per_batch(task,images,labels,evaluation,noise_trigger,user_id):

    new_images = images
    new_targets = labels

    for index in range(0, len(images)):
        if evaluation:  # poison all data when testing
            new_targets[index] = task.params.backdoor_label

            new_images[index] = add_pixel_pattern(images[index], noise_trigger,task,user_id)


        else:  # poison part of data when training
            if index < batch_size*task.params.poisoning_proportion:
                new_targets[index] = task.params.backdoor_label
                new_images[index] = add_pixel_pattern(images[index], noise_trigger,task,user_id)

    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets
def CerP_test(hlpr:Helper,epoch,noise_trigger,backdoor=False,model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    dataloader = hlpr.task.test_loader
    if not backdoor:
        for inputs1, targets1 in dataloader:
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            outputs = model(inputs1)

            hlpr.task.accumulate_metrics(outputs=outputs, labels=targets1)
    else:
        for inputs1, targets1 in dataloader:
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs_bd,targets_bd=poison_per_batch(hlpr.task,inputs1,targets1,True,noise_trigger,5)
            inputs1 = inputs_bd
            targets1 = targets_bd
            outputs = model(inputs1)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=targets1)
    metric = hlpr.task.report_metrics(epoch,
                                      prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric
def Cerp_train_with_grad_control(model,  trainloader, criterion, optimizer, task,user_id,pretrained_normal_model,noise_trigger):

    for images,labels in trainloader:
        images,labels=images.to(device),labels.to(device)
        images,labels=poison_per_batch(task,images,labels,False,noise_trigger,5)
        output = model(images)
        model.zero_grad()
        normal_loss=criterion(output,labels)
        distance_loss=model_dist_norm_var(model,pretrained_normal_model.state_dict())
        sum_cs=0.0
        for i in range(user_id):
            updates_name = f'{task.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name)
            sum_cs+=model_cosine_similarity(model,loaded_params)
        loss=normal_loss+0.001*distance_loss+0.001*sum_cs
        loss.backward()
        optimizer.step()
def Cerp_run_fl_round(hlpr: Helper, epoch,noise_trigger,intinal_trigger):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants,idx = hlpr.task.sample_users_for_round(epoch)  # 返回的是client列表，每个client包含id，是否恶意，dataloader
    weight_accumulator = hlpr.task.get_empty_accumulator()
    tuned_trigger = cifar100_trigger(hlpr,  global_model, noise_trigger, intinal_trigger)
    noise_trigger=tuned_trigger
    #noise_trigger=intinal_trigger
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)    #把全局模型赋值给本地模型
        make_optimizer = hlpr.task.make_optimizer(local_model)
        optimizer = make_optimizer
        normal_model=deepcopy(local_model)
        normal_optimizer=hlpr.task.make_optimizer(normal_model)
        normal_model.train()
        local_model.train()
        if user.compromised:
            continue
        else:
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        hlpr.save_update(model=local_update, userID=user.user_id)
    for i in range(hlpr.task.params.fl_number_of_adversaries):
        user_id=i
        hlpr.task.copy_params(global_model, local_model)  # 把全局模型赋值给本地模型
        make_optimizer = hlpr.task.make_optimizer(local_model)
        optimizer = make_optimizer
        normal_model = deepcopy(local_model)
        normal_optimizer = hlpr.task.make_optimizer(normal_model)
        normal_model.train()
        local_model.train()
        train_loader =hlpr.task.fl_train_loaders[user_id]
        for local_epoch in range(hlpr.params.fl_local_epochs):
            train(hlpr, local_epoch, normal_model, normal_optimizer,
                  train_loader, attack=False)
        for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
            Cerp_train_with_grad_control(local_model, train_loader, torch.nn.CrossEntropyLoss(), optimizer,
                                         hlpr.task, user_id, normal_model, noise_trigger)
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        hlpr.save_update(model=local_update, userID=user_id)


    hlpr.defense.aggr(weight_accumulator, global_model, idx)

    hlpr.task.update_global_model(weight_accumulator, global_model)
    return noise_trigger

def run(hlpr: Helper):
    if hlpr.task.params.attack == 'CerP':
        poison_patterns = []
        _0poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        _1poison_pattern = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
        _2poison_pattern = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
        _3poison_pattern = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
        poison_patterns = poison_patterns + _0poison_pattern + _1poison_pattern + _2poison_pattern + _3poison_pattern
        if hlpr.task.params.task == 'MNIST':
            intinal_trigger = torch.zeros([28, 28])
        else:
            intinal_trigger = torch.zeros([3, 32, 32])
        triggervalue = 1
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            if hlpr.task.params.task == 'MNIST':
                intinal_trigger[pos[0]][pos[1]] = triggervalue
            else:
                intinal_trigger[0][pos[0]][pos[1]] = triggervalue  # +delta i  #？
                intinal_trigger[1][pos[0]][pos[1]] = triggervalue  # +delta i
                intinal_trigger[2][pos[0]][pos[1]] = triggervalue  # +delta i
            # 这里需要对noise进行cifar10数据集的正则化
        if hlpr.task.params.task == 'MNIST':
            mean = torch.tensor(0.1307)
            std = torch.tensor(0.3081)
            intinal_trigger = (intinal_trigger - mean[None, None]) / std[None, None]
        else:
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            intinal_trigger = (intinal_trigger - mean[:, None, None]) / std[:, None, None]
        noise_trigger = copy.deepcopy(intinal_trigger)
        for epoch in range(hlpr.params.start_epoch,
                           hlpr.params.epochs + 1):
            if hlpr.task.params.defense == 'Indicator':
                hlpr.defense.pre_process(hlpr.task.test_loader, epoch, hlpr.task.model)
            Cerp_run_fl_round(hlpr, epoch, noise_trigger, intinal_trigger)
            metric = CerP_test(hlpr, epoch, noise_trigger, False)
            hlpr.record_accuracy(metric, CerP_test(hlpr, epoch, noise_trigger, True), epoch)
            hlpr.save_model(hlpr.task.model, epoch, metric)
    else:
        for epoch in range(hlpr.params.start_epoch,
                           hlpr.params.epochs + 1):

            run_fl_round(hlpr, epoch)
            metric = test(hlpr, epoch, backdoor=False)
            hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)
            #torch.save(hlpr.task.model.state_dict(),'imagenet_model.pth')
            hlpr.save_model(hlpr.task.model, epoch, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))
    try:
        run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. ")
        else:
            logger.error(f"Aborted training. No output generated.")
    helper.remove_update()