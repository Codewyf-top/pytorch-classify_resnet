import argparse
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms

from config import opt
from models.base_model import get_network
from PIL import Image

import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50', help='net type')
    parser.add_argument('-weights', type=str, default='checkpoint/resnet50/20200612_113350/resnet50-49-best.pth',
                        help='the weights file you want to test')
    parser.add_argument('-source', type=str, default='images/A_1.bmp',
                        help='the images path')

    args = parser.parse_args()
    classes = ('A','B','C')
    # 1.load net
    net = get_network(args.net, opt.num_class, use_gpu=False)
    print(net)
    # 2.load weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(args.weights, map_location=device))
    # net = nn.DataParallel(net)
    # print(net)
    net.to(device)
    net.eval()
    img_path = args.source
    img = Image.open(img_path)

    trans = transforms.Compose([
        # transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    output = net(img)

    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    value, predicted = torch.max(output.data, 1)
    pred_class = classes[predicted.item()]
    pred_score = prob[0][predicted.item()].item()
    print('输入图片为 ：{}'.format(img_path))
    print('预测的结果为 : {}, 准确率为 : {}'.format(pred_class, str(pred_score)))
