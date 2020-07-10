#### pytorch从头训练

#####1.安装依赖包
```shell script
pip3 install -r requirements.txt
```

#####2.数据增强
```shell script
cd utils
# 注意打开函数里面改改路劲配置
python3 enhance_data.py
```

#####3.数据划分
```shell script
cd utils
# 注意打开函数里面修改路劲
python3 split_floder_data.py
```
>注意将不同尺寸的图片放在x32同级目录下,然后新训练一种尺寸的图片则修改split_floder_data.py的文件参数，并删除data文件夹下面的图片
>最好再执行数据划分
>新的训练数据集会自动生成在data下

---
>注意第二步可以不做，因为本身数据够了
>

#####4.训练模型
```shell script
# resnet50
python3 train.py -net resnet50 -b 4 -w 4
# 默认train.py里面写好了默认的配置，resnet50的
# python3 train.py 会默认执行renset50，注意修改一下-b（batch size参数） 
```
>支持resnet18,resnet50,resnet34,resnet101,resnet152,只需要修改-net 后面的参数
>也注意下显存的使用，bs设置

#####5.测试模型
```shell script
python3 test.py -net resnet50 -weights checkpoint/resnet50/20200612_113350/resnet50-49-best.pth
# 默认使用resnet50从头训练的模型
```
> -weights 就是测试时需要加载的权重,默认位置在checkpoint/net/时间戳/下面

#####6.文件夹说明
- checkpoint : 训练权重保存文件夹
- data : 随机切分的数据保存路劲
- dataloader : 数据加载模块
- models ：网络模块
- x32 : 原始图片
- runs : tensorboard 保存的训练过程图
- utils : 数据处理脚本

#####7.测试单张图片c
```shell script
# 修改-source 的图片路劲
python3 pred_img.py -source images/1_006.jpg -weights checkpoint/resnet50/20200612_113350/resnet50-49-best.pth
```
>注意一下因为是多尺度的训练，所以dataloader没有统一resie图片，注意测试时候，保证这个图片的大小和训练时的图片一致

#####8.tensorboard查看训练过程图
```shell script
tensorboard --logdir runs/resnet50/20200612_113350
```



#####9.需要注意的点
>注意训练不同尺寸图片时间，权重和tensorboard日志都是根据时间戳保存的
>每次训练新的尺寸的数据时，注意把原data下上一次分割后的图片删除
>x32的batch-size能设置为34，x256最大就4了，注意一下batch-size的设置，如果报显存不够的错误
>一些参数的设置在config.py里面
