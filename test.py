import argparse
from matplotlib import pyplot as plt

import torch
from config import opt
from models.base_model import get_network
from dataloader.dataset_floder import FloderData





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50', help='net type')
    parser.add_argument('-weights', type=str, default='checkpoint_80/resnet50/20200614_113817/resnet50-30-regular.pth',
                        help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    args = parser.parse_args()
    # 1.load net
    net = get_network(args.net, opt.num_class)
    print(net)
    # 2.load data
    fd = FloderData(batch_size=args.b, train_path=opt.floder_data_dict['train'],
                    valid_path=opt.floder_data_dict['valid'], test_path=opt.floder_data_dict['test'],
                    num_w=args.w,
                    load_model='test')
    dataloaders = fd.get_dataloader()
    # 3.load weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(args.weights, map_location=device))

    # net = nn.DataParallel(net)
    # print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    # 针对每种类别计算
    classes = ('0', '1', '2','3','4','5','6','7','8','9')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # 添加用来计算每一类别的ACC
    for n_iter, (images, labels) in enumerate(dataloaders['test']):
        # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloaders['test'])))
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        _, predicted = torch.max(output, 1)
        c = (predicted == labels).squeeze()
        label = labels.item()
        if c.item():
            class_correct[label] +=1
        class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # 常规计算top-1/top-5
    for n_iter, (images, labels) in enumerate(dataloaders['test']):

        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        # # top 5 方式
        # -----------------------------------------------------
        _, pred = output.topk(3, 1, largest=True, sorted=True)
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()
        # compute top 5
        correct_5 += correct[:, :3].sum()
        # compute top1
        correct_1 += correct[:, :1].sum()
        # -----------------------------------------------------
    #print("Top 1 err: ", 1 - correct_1 / len(dataloaders['test'].dataset))
    mean_acc = (correct_1 / len(dataloaders['test'].dataset)).item()
    # print(mean_acc)
    print("Accuracy of mean all : %2d %%" %(100*mean_acc))
    # print("Top 3 err: ", 1 - correct_5 / len(dataloaders['test'].dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

