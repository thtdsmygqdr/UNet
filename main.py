import torch
from torchvision.transforms import transforms as T
import argparse #argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
import unet
from torch import optim
from dataset import LiverDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()

def train_model(model,criterion,optimizer,dataload,num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0 #minibatch数
        for x, y in dataload:# 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()#每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)#前向传播
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#梯度下降,计算出梯度
            optimizer.step()#更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    # torch.save(model.state_dict(),'weights_%d.pth' % epoch)# 返回模型的所有内容
    ablation_str = str(model.ablation_mode).lower().replace(" ", "_")
    torch.save(model.state_dict(), f'weights_{ablation_str}_{epoch}.pth')
    return model

#训练模型
def train():
    model = unet.UNet(3,1, ablation_mode=args.ablation).to(device)
    batch_size = args.batch_size
    #损失函数
    criterion = torch.nn.BCELoss()
    #梯度下降
    optimizer = optim.Adam(model.parameters())#model.parameters():Returns an iterator over module parameters
    #加载数据集
    liver_dataset = LiverDataset("data/train", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度 
    train_model(model,criterion,optimizer,dataloader)

#测试
# main.py (修改后的test函数)
def test():
    model = unet.UNet(3, 1, ablation_mode=args.ablation, action='test')
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()

    liver_dataset = LiverDataset("data/val", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)

    # 创建单个图像显示窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()  # 交互模式

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloaders):
            # 处理输入图像显示（反归一化）
            input_img = torch.squeeze(x).permute(1, 2, 0)
            input_img = (input_img * 0.5) + 0.5  # 从[-1,1]转换到[0,1]

            if args.action == 'test':
                pred, features = model(x)
                features['input_img'] = input_img.numpy()  # 存储输入用于可视化
                if i == 0:  # 只可视化第一个样本的特征
                    visualize_features(features)
            else:
                pred = model(x)

            # 更新图像而不创建新窗口
            ax1.clear()
            ax2.clear()

            ax1.imshow(input_img.numpy())
            ax1.set_title(f"Original Image ({i + 1}/{len(dataloaders)})")
            ax1.axis('off')

            pred_mask = torch.squeeze(pred).numpy()
            ax2.imshow(pred_mask, cmap='gray')
            ax2.set_title("Predicted Mask")
            ax2.axis('off')

            plt.tight_layout()
            plt.pause(5)  # 更短的暂停时间

    plt.ioff()
    plt.close()


def visualize_features(feature_dict):
    """可视化网络中的特征图"""
    # 浅层特征（细节信息）
    shallow_feat = feature_dict['shallow']
    channel_mean = torch.mean(torch.abs(shallow_feat), dim=1)[0]  # 通道绝对值平均

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Input Image")
    plt.imshow(feature_dict['input_img'], cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title("Shallow Feature Activation")
    plt.imshow(channel_mean.detach().numpy(), cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 深层特征（语义信息）
    deep_feat = feature_dict['deep']
    channel_mean = torch.mean(torch.abs(deep_feat), dim=1)[0]  # 通道绝对值平均

    plt.subplot(133)
    plt.title("Deep Feature Activation")
    plt.imshow(channel_mean.detach().numpy(), cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.close()  # 关闭图形避免堆积


if __name__ == '__main__':
    #参数解析
    parser = argparse.ArgumentParser() #创建一个ArgumentParser对象
    parser.add_argument('action', type=str, help='train or test')#添加参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['none', 'no_skip', 'freeze_deep'],
                        help='Ablation study mode')

    args = parser.parse_args()
    
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
