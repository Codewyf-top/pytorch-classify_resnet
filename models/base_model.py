import torch
import sys


def get_network(net_name, num_class, use_gpu=True):
    """ return given network
    """
    if net_name == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_class)
    elif net_name == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_class)
    elif net_name == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_class)
    elif net_name == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_class)
    elif net_name == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_class)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    if use_gpu:
        net = net.cuda()

    return net
