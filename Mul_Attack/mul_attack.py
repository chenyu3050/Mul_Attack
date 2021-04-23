
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from PIL import Image
from torchvision.utils import save_image

import inversefed

import os 


defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR10', defs)

def mul_attack(local_lr:float,local_steps:int,use_updates:bool,dm,ds,arch,trained_model:False,num_images:int)->None:

    defs = inversefed.training_strategy('conservative')

    loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR10', defs)

    model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=3)
    model.to(**setup)
    if trained_model:
        epochs = 120
        file = f'{arch}_{epochs}.pth'
        try:
            model.load_state_dict(torch.load(f'models/{file}'))
        except FileNotFoundError:
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), f'models/{file}')
    model.eval()
    ground_truth,labels=show_groundtruth(num_images)
    batch_attack(model,ground_truth,labels,local_lr,local_steps,use_updates,dm,ds,num_images)
def plot(tensor,name:str):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1: # shape[0]=batch_size
        return plt.show(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12)) # 子画布是1*batchsize的模式画图
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
        plt.savefig('output/'+name+'_batch_{}.jpg'.format(tensor.shape[0]))


# show batch attack effect 

def batch_attack(model,ground_truth,labels,local_lr:float,local_steps:int,use_updates:bool,dm,ds,num_images:int)->None: 
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels, 
                                                        lr=local_lr, local_steps=local_steps,
                                                                   use_updates=use_updates)
    input_parameters = [p.detach() for p in input_parameters]

    config = dict(signed=True, # 符号
              boxed=True,      # 盒约束 
              cost_fn='sim',   # 损失函数
              indices='def',   # 未知 
              weights='equal', # 初始化等权重
              lr=0.1,          # 梯度下降的学习率
              optim='adam',    # 优化方法的选择
              restarts=1,
              max_iterations=200, # 迭代的轮次
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                                use_updates=use_updates,num_images=num_images) # fix by alex 
    output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=(3, 32, 32))

    test_mse = (output.detach() - ground_truth).pow(2).mean() # 做差平方 符合DLG的思路 
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    plot(output,'adv')  

# show ground truth  however the figure quailty is not good 
def show_groundtruth(num_images)->tuple:
    
    ground_truth, labels = [], []
    idx = 25 # choosen randomly ... just whatever you want
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    plot(ground_truth,'origin')

    return ground_truth,labels

def mkdir():

    file_path = os.getcwd() #获得当前工作目录

    if os.path.exists(file_path + "\output"):
        print("output已存在")
    else:
        os.makedirs(file_path + "\output")
if __name__ == "__main__":
    # make  dir output
    mkdir()
    # init dm and ds 
    setup = inversefed.utils.system_startup()
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    # mul_attack
    mul_attack(local_lr=0.01,local_steps=4,use_updates=True,dm=dm,ds=ds,arch="ResNet20",trained_model=False,num_images=2) # 修改图片数量
