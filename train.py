import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from dataloader.dataset_floder import FloderData
from config import opt
from models.base_model import get_network
from utils.warmup_lr import WarmUpLR


# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 随机种子，保证一致性
# seed_everything(opt.seed)


def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(dataloaders['train']):
        if epoch <= args.warm:
            warmup_scheduler.step()
        labels = labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(dataloaders['train']) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(dataloaders['train'].dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def computer_loss(outputs, labels,class_correct,class_total,class_loss):
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(outputs.shape[0]):
        loos_temp = loss_function(outputs[i].view(1,3), torch.tensor([labels[i].data]).to(device))
        label = labels[i].data
        class_correct[label] += c[i].item()
        class_total[label] += 1
        class_loss[label]+=loos_temp.data
    return class_correct,class_total,class_loss



def eval_valid(epoch):
    net.eval()
    
    classes = ('A', 'B', 'C')
    test_loss = 0.0  # cost function error
    correct = 0.0
    N_CLASSES = 3
    class_correct = list(0. for i in range(N_CLASSES))
    class_loss = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))
    for (images, labels) in dataloaders['valid']:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        class_correct,class_total,class_loss = computer_loss(outputs, labels,class_correct,class_total,class_loss)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(dataloaders['valid'].dataset),
        correct.float() / len(dataloaders['valid'].dataset)
    ))
    
    for i in range(N_CLASSES):
        print('Accuracy of %5s : %2d %%  Loss:%2d ' % (
            classes[i], 100 * class_correct[i] / class_total[i], class_loss[i] / class_total[i]))
        writer.add_scalar(classes[i]+'Test'+'/Average loss', class_loss[i] / class_total[i], epoch)
        writer.add_scalar(classes[i]+'Test'+'/Accuracy',class_correct[i] / class_total[i], epoch)

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(dataloaders['valid'].dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(dataloaders['valid'].dataset), epoch)
    return correct.float() / len(dataloaders['valid'].dataset)

def eval_training(epoch):
    net.eval()
    
    classes = ('A', 'B', 'C')
    test_loss = 0.0  # cost function error
    correct = 0.0
    N_CLASSES = 3
    class_correct = list(0. for i in range(N_CLASSES))
    class_loss = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))
    for (images, labels) in dataloaders['valid']:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        class_correct,class_total,class_loss = computer_loss(outputs, labels,class_correct,class_total,class_loss)
    
    for i in range(N_CLASSES):
        writer.add_scalar(classes[i]+'Train/'+'Average loss', class_loss[i] / class_total[i], epoch)
        writer.add_scalar(classes[i]+'Train/'+'Accuracy',class_correct[i] / class_total[i], epoch)
    # add informations to tensorboard
    writer.add_scalar('Train/Average loss', test_loss / len(dataloaders['train'].dataset), epoch)
    writer.add_scalar('Train/Accuracy', correct.float() / len(dataloaders['train'].dataset), epoch)
    return correct.float() / len(dataloaders['train'].dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    args = parser.parse_args()

    # 1.load net
    net = get_network(net_name=args.net, num_class=opt.num_class, use_gpu=args.gpu)
    print(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = nn.DataParallel(net)
    # 2.load data
    fd = FloderData(batch_size=args.b, train_path=opt.floder_data_dict['train'],
                    valid_path=opt.floder_data_dict['valid'], test_path=opt.floder_data_dict['test'],
                    num_w=args.w,
                    load_model='train')
    dataloaders = fd.get_dataloader()
    # 3.set init function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milepochs,
                                                     gamma=0.2)  # learning rate decay
    # 4.set warmup
    iter_per_epoch = len(dataloaders['train'])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    # 5.save pth outpath and tensorboard outpath
    checkpoint_path = os.path.join(opt.checkpoint_path, args.net, opt.time_now)
    # use tensorboard
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(
        opt.log_dir, args.net, opt.time_now))
    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    # 6. train and valid
    best_acc = 0.0
    for epoch in range(1, opt.epoch):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(epoch)
        train_acc = eval_training(epoch)
        acc = eval_valid(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > opt.milepochs[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % opt.save_epoch:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
