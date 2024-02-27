import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import itertools
import json
import matplotlib.pyplot as plt
from vivit import ViT
from model import VisionTransformer
from config import get_train_config
from checkpoint import load_checkpoint
from data_loaders import VIDEODataLoader
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter

np.random.seed(0)
torch.manual_seed(0)
# FIG_SAVE_DIR = ''

def plot_confusion_matrix(plt_dir, cm, epoch, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix " + 'Epoch:'+str(epoch))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = cm.numpy()
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plt_dir,'cm_epoch_'+str(epoch)+'.png'))
    # return figure


def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        metrics.update('acc1', acc1.item())
        metrics.update('acc5', acc5.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                    .format(epoch, batch_idx, len(data_loader), loss.item(), acc1.item(), acc5.item()))
    
    return metrics.result()


def valid_epoch(plt_dir, epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    nb_classes = 155
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            batch_pred = model(batch_data)
            _, preds = torch.max(batch_pred,1)
            loss = criterion(batch_pred, batch_target)

            for t, p in zip(batch_target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
          
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    ##  plot the heatmap for per class accuracy
    # class_names = [str(x) for x in range(nb_classes)]
    # plot_confusion_matrix(plt_dir, confusion_matrix,epoch=epoch, class_names=class_names)

    per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    per_class_acc = per_class_acc.numpy()

    # print(per_class_acc.numpy())
    with open(os.path.join(plt_dir,'confusion_matrix.csv'),'a') as cmfile:
        for prob in per_class_acc:
            cmfile.write(str(prob)+' ')
        cmfile.write('\n')

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)

    return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")

    model = ViT(
        image_size = 224,          # image size
        frames = 16,               # number of frames
        image_patch_size = 16,     # image patch size
        frame_patch_size = 2,      # frame patch size
        num_classes = config.num_classes,
        dim = 1024,
        spatial_depth = 6,         # depth of the spatial transformer
        temporal_depth = 6,        # depth of the temporal transformer
        heads = 8,
        mlp_dim = 2048
    )

    print(model)


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

    # model = torchvision.models.video.r3d_18(weights=None)
    # model.fc = nn.Linear(512, config.num_classes)
    # print('Change the final output to ' + str(config.num_classes))

    # # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        # print(state_dict['embedding.weight'].size)
        # if config.num_classes != state_dict['classifier.weight'].size(0):
        #     del state_dict['classifier.weight']
        #     del state_dict['classifier.bias']
        #     print("re-initialize fc layer")
        #     model.load_state_dict(state_dict, strict=False)
        # else:
        #     del state_dict['transformer.pos_embedding.pos_embedding']
        model.load_state_dict(state_dict,strict=True)

        print("Load pretrained weights from {}".format(config.checkpoint_path))
    
    # model = torchvision.models.video.r3d_18(weights='DEFAULT')

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    print("create dataloaders")
    train_dataloader = VIDEODataLoader(
                    datasample = config.datasample,
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train')
    valid_dataloader = VIDEODataLoader(
                    datasample = config.datasample,
                    data_dir=config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.CrossEntropyLoss()

    # create optimizers and learning rate scheduler
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    # start training
    print("start training")
    best_acc = 0.0
    epochs = config.train_steps // len(train_dataloader)
    for epoch in range(1, epochs + 1):
        log = {'epoch': epoch}

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, device)
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(config.plt_dir, epoch, model, valid_dataloader, criterion, valid_metrics, device)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        best = False
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best = True

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    main()


# python src/train_video.py --exp-name r3d_cropped_randomnize --n-gpu 8 --tensorboard  --model-arch b16 --image-size 224 --checkpoint-path /fs/nexus-scratch/rxian/research/vision-transformer-pytorch/experiments/save/r3d_cropped_randomnize_ImageNet_bs64_lr0.005_wd0.0_231109_230803/checkpoints/best.pth --batch-size 64 --data-dir /fs/nexus-projects/AerialAI/data/UAVHuman/frames --num-classes 155 --train-steps 10000 --lr 0.005 --wd 0.0

# python src/train.py --exp-name uav --n-gpu 8 --tensorboard  --model-arch b16 --checkpoint-path weights/pytorch/imagenet21k+imagenet2012_ViT-B_16.pth --image-size 224 --batch-size 32 --data-dir /fs/nexus-projects/AerialAI/data/UAVHuman/frames --dataset UAVHUMAN --num-classes 155 --train-steps 10000 --lr 0.03 --wd 0.0