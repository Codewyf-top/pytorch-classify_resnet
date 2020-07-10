import os
import time


class ConfigSetting(object):
    # directory to save weights file
    checkpoint_path = 'checkpoint'

    # total training epoches
    # step lr 的设置
    epoch = 50
    milepochs = [25, 35, 40]

    # initial learning rate
    init_lr = 0.001

    # time of we run the script
    time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    # tensorboard log dir
    log_dir = 'runs'

    # save weights file per SAVE_EPOCH epoch
    # 每多少个epoch保存epoch
    save_epoch = 5

    # 分类类别数
    # label  number
    num_class = 3

    floder_data_dict = {
        'train': 'data/train',
        'valid': 'data/valid',
        'test': 'data/test',
    }
    seed = 10


opt = ConfigSetting()
