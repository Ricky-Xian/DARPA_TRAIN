import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import VisionTransformer
import torchvision
from config import get_eval_config
from checkpoint import load_checkpoint
from data_loaders import VIDEODataLoader,UAVHUMANDataLoader
from utils import accuracy, setup_device


def main():

    config = get_eval_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # create model
    # model = VisionTransformer(
    #          image_size=(config.image_size, config.image_size),
    #          patch_size=(config.patch_size, config.patch_size),
    #          emb_dim=config.emb_dim,
    #          mlp_dim=config.mlp_dim,
    #          num_heads=config.num_heads,
    #          num_layers=config.num_layers,
    #          num_classes=config.num_classes,
    #          attn_dropout_rate=config.attn_dropout_rate,
    #          dropout_rate=config.dropout_rate)
    
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, config.num_classes)
    print('Change the final output to ' + str(config.num_classes))

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    data_loader = VIDEODataLoader(
                    datasample = 'normal',
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')
    total_batch = len(data_loader)

    # starting evaluation
    print("Starting evaluation")
    acc1s = []
    acc5s = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to(device)
            # print('data_size: {}'.format(data.size()))
            target = target.to(device)
            # print('target_size: {}'.format(target.size()))
            
            logits = []
            for every_batch in data:
                # print('batch_size: {}'.format(every_batch.size()))
                pred_logits = model(every_batch.permute(1,0,2,3))
                # print('PRED_LOGITS_size: {}'.format(pred_logits.size()))  
                # pred_logits = model(every_batch)

                # # meanpooling
                # avg_logits = torch.mean(pred_logits,0)
                # # logsumexp
                # avg_logits = torch.logsumexp(pred_logits,0)
                # # maxpooling
                avg_logits = torch.max(pred_logits,0)
                # print('AVG_LOGITS_size: {}'.format(avg_logits.size())) 
                logits.append(avg_logits.values)

            logits = torch.stack(logits,0)     
            # print('LOGITS_size: {}'.format(logits.size()))           
                
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format(config.model_arch, config.dataset, np.mean(acc1s), np.mean(acc5s)))


if __name__ == '__main__':
    main()
    
# python src/eval_new.py --n-gpu 8 --model-arch b16 --checkpoint-path /fs/nexus-scratch/rxian/research/vision-transformer-pytorch/experiments/save/uav_UAVHUMAN_bs448_lr0.005_wd0.0_231101_223404/checkpoints/current.pth --image-size 224 --batch-size 32 --data-dir /fs/nexus-projects/AerialAI/ssl_data_cropped/UAVHuman --dataset VIDEO --num-classes 155
