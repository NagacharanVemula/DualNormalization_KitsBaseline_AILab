import os
import random
import datetime
import argparse
import numpy as np
import medpy.metric.binary as mmb
from tqdm import tqdm
from model.unetdsbn import Unet2D
from utils.loss import dice_loss1
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel
from test_utils import get_bn_statis, cal_distance
import torch
import torchvision.transforms as tfs
from torch import optim
from torch.optim import Adam
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import logging
import copy
import time
from torch.utils.tensorboard import SummaryWriter

def get_args():
  
    parser = argparse.ArgumentParser(description='Dual Normalization U-Net Training Script')
    
    # Data parameters
    parser.add_argument('--data_dir', 
                       type=str, 
                       default='./data/brats/npz_data',
                       help='Directory containing the training data')
    
    parser.add_argument("--train_domain_list_1", 
                       nargs='+', 
                       required=True,
                       help='List of domains for first training set (e.g., KiTS_ss)')
    
    parser.add_argument("--train_domain_list_2", 
                       nargs='+', 
                       required=True,
                       help='List of domains for second training set (e.g., KiTS_sd)')
    
    parser.add_argument('--result_dir', 
                       type=str, 
                       default='./results/unet_dn',
                       help='Directory to save training results')
    
    # Model parameters
    parser.add_argument('--n_classes', 
                       type=int, 
                       default=2,
                       help='Number of segmentation classes')
    
    # Training parameters
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=64,
                       help='Training batch size')
    
    parser.add_argument('--n_epochs', 
                       type=int, 
                       default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--lr', 
                       type=float, 
                       default=0.001,
                       help='Learning rate')
    
    # System parameters
    parser.add_argument('--seed', 
                       type=int, 
                       default=1337,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--gpu_ids', 
                       type=str, 
                       default='0,1',
                       help='Comma-separated list of GPU IDs to use')
    
    parser.add_argument('--deterministic', 
                       action='store_true',
                       help='Enable deterministic training')
    
    args = parser.parse_args()
    return args

args = get_args()

def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x


def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)



if __name__== '__main__':
    start_time1 = datetime.datetime.now()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    base_dir = args.data_dir
    batch_size = args.batch_size
    lr = args.lr
    train_domain_list_1 = args.train_domain_list_1
    train_domain_list_2 = args.train_domain_list_2
    max_epoch = args.n_epochs
    result_dir = args.result_dir
    n_classes = args.n_classes
    log_dir = os.path.join(result_dir, 'log')
    model_dir =  'model'

    writer = SummaryWriter("runs/exp_kits_DN")


    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)



    dataloader_train = []
    model = Unet2D(num_classes=n_classes, norm='dsbn', num_domains=2)
    params_num = sum(p.numel() for p in model.parameters())
    print("Training started!")
    print("\nModle's Params: %.3fM" % (params_num / 1e6))
    model = DataParallel(model).cuda()

    optimizer = Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))

    exp_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)





    #Train datasets and data loaders for each domain's data
    dataset_1 = Dataset(base_dir=base_dir, split='train', domain_list=train_domain_list_1, 
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=n_classes),
                            ToTensor()
                        ]))
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    dataloader_train.append(dataloader_1)
    dataset_2 = Dataset(base_dir=base_dir, split='train', domain_list=train_domain_list_2, 
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=n_classes),
                            ToTensor()
                        ]))
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    dataloader_train.append(dataloader_2)


    #Validation dataset and data loader
    dataset = Dataset(base_dir= base_dir, split='val', domain_list= "KiTS",
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=n_classes),
                            ToTensor()
                        ]))
    dataloader_val = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)





    
    best_mean_dice = -1
    
    for epoch_num in range(max_epoch):
        data_iter = [repeat_dataloader(dataloader_train[i]) for i in range(2)]
        start_time = time.time()
        print('Epoch: {}, LR: {}'.format(epoch_num+1, round(exp_lr.get_last_lr()[0], 6)))
        epoch_loss = 0
        model.train()
        batch_loss = 0

        for i, batch in enumerate(dataloader_train[0]):
            ### get all domains' sample_batch ###
            sample_batches = [batch]
            other_sample_batches = [next(data_iter[i]) for i in range(1, 2)]
            sample_batches += other_sample_batches

            total_loss = 0
            count = 0
            for train_idx in range(2):
                count += 1
                sample_data, sample_label = sample_batches[train_idx]['image'].cuda(), sample_batches[train_idx]['onehot_label'].cuda()
                # print("shapes of input batch and masks:",sample_data.shape, sample_label.shape)
                outputs_soft = model(sample_data, domain_label=train_idx*torch.ones(sample_data.shape[0], dtype=torch.long))
                loss = dice_loss1(outputs_soft, sample_label)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            batch_loss+= (total_loss / count) 
        epoch_loss = batch_loss / (i+1)
        print("number of batches:", i+1)
        print(f"epoch-{epoch_num+1} - Average Loss:", epoch_loss)
        writer.add_scalar("Loss/train", epoch_loss, epoch_num)



        exp_lr.step()


        ######Validation part#########

        #Calculating the estimates of two domain specific batchnorms of the trained model
        means_list = []
        vars_list = []
        
        for i in range(2):
            means, vars = get_bn_statis(model, i)
            means_list.append(means)
            vars_list.append(vars)

        
        total_dice = 0

        model_copy = copy.deepcopy(model)
        model_copy.train()   

        with torch.no_grad():
            for idx, (batch, id) in enumerate(dataloader_val):
                model.train()            
                sample_data = batch['image'].cuda()
                onehot_mask = batch['onehot_label'].detach().numpy()
                mask = batch['label'].detach().numpy()
                dis = 99999999
                best_out = None
                for domain_id in range(2):
                    output = model_copy(sample_data, domain_label=domain_id*torch.ones(sample_data.shape[0], dtype=torch.long))
                    means, vars = get_bn_statis(model_copy, domain_id)
                    new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
                    if new_dis < dis:
                        selected_domain = domain_id
                        dis = new_dis

                model_copy = copy.deepcopy(model)

                model.eval()
                output_selected = model(sample_data,  domain_label=selected_domain*torch.ones(sample_data.shape[0], dtype=torch.long) )
                output = output_selected
      
                pred_y = output.cpu().detach().numpy()

                pred_y = np.argmax(pred_y, axis=1)

                total_dice += mmb.dc(pred_y, mask)


            print('Mean Dice: {}, HD: {}, ASD: {}'.format(
                round(total_dice / (idx + 1), 4)
            ))
            

        epoch_mean_dice = round(total_dice / (idx + 1), 4)
        writer.add_scalar("Mean Dice/val", epoch_mean_dice, epoch_num)
        writer.flush()

        #Saving the best model
        if epoch_mean_dice > best_mean_dice:
            best_mean_dice = epoch_mean_dice
            best_metric_epoch = epoch_num + 1
            torch.save({"model_dict": model.module.state_dict(), "optimizer_dict": optimizer.state_dict()}, "best_model_DN.pth")
            print(f"Saved new best model at epoch {best_metric_epoch} with dice {best_mean_dice:.4f}")

        print(f"Epoch duration: {time.time() - start_time:.2f} seconds")

    end_time = datetime.datetime.now()
    print('Finish running. Cost total time: {} hours'.format((end_time - start_time1).seconds / 3600))
    writer.close()



