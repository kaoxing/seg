import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from MyGAN import MyGAN as GAN
from MultiTaskAttentionUNet import MultiTasAttentionUNet as UNet
from DatasetGAN import MyDataSetPre
from DatasetUNet import MyDataSetTra
from torch.utils.data import DataLoader
from LossFunction import DLossFunction, GLossFunction

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取参数
    # argv = sys.argv[1:]
    # print(argv)
    batch_size = int(input("batch_size="))
    learning_rate = float(input("learning_rate="))
    epoch = int(input("epoch="))
    result_path = os.path.join("./dataset/result/", input("result_dir="))
    os.mkdir(result_path)
    label_path = "./dataset/train_little/label/"
    raw_path = "./dataset/train_little/raw/"
    d_path = os.path.join("./models/d/", input("d_name="))
    g_path = os.path.join("./models/g/", input("g_name="))

    # 加载模型
    D = UNet()
    D.to(device)
    G = GAN()
    G.to(device)

    # 加载数据集
    dataset = MyDataSetTra(raw_path, label_path)
    dataset = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14
    )
    dataset_test = MyDataSetTra(raw_path, label_path)
    dataset_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
    )
    # 优化器定义
    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=learning_rate
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=learning_rate
    )

    # 损失函数定义
    loss_D = DLossFunction()
    loss_G = GLossFunction()

    # 训练
    for cnt in range(epoch):
        print("迭代轮次[{}/{}]".format(cnt + 1, epoch))
        Loss_D_True = 0
        Loss_D_Fake = 0
        Loss_G = 0
        for i, data in enumerate(dataset):
            input_data, labels = data
            input_data = input_data.to(device)
            labels = labels.to(device)
            optimizer_D.zero_grad()  # 梯度置零
            optimizer_G.zero_grad()
            # 输入假数据
            fake_img = G(labels, torch.randn(input_data.size()))  # 数据输入生成网络
            predict_fake, _ = D(fake_img.detach())  # 生成假数据送入辨别网络
            loss_g = loss_G(False, predict_fake)  # 计算生成网络损失
            loss_g.backward(retain_graph=True)  # 误差逆传播
            optimizer_G.step()
            loss_d_fake = loss_D(False, predict_fake)  # 计算辨别网络损失z
            # loss_d_fake.backward()
            # optimizer_D.step()
            # optimizer_D.zero_grad()
            # 输入真数据
            predict_true, _ = D(input_data)
            loss_d_true = loss_D(True, predict_true)
            # loss_d_true.backward()
            loss_d = (loss_d_true + loss_d_fake) / 2
            loss_d.backward()
            optimizer_D.step()  # 通过梯度调整参数
            Loss_G += loss_g.item()
            Loss_D_True += loss_d_true.item()
            Loss_D_Fake += loss_d_fake.item()
            # print("loss_g.item()", loss_g.item())
            # print("loss_d_fake.item():", loss_d_fake.item())
            # print("loss_d_true.item():", loss_d_true.item())
        print("loss_g:", Loss_G)
        print("loss_d_fake:", Loss_D_Fake)
        print("loss_d_true:", Loss_D_True)

    # 测试
    with torch.no_grad():
        for i, data in enumerate(dataset_test):
            input_data, label = data
            input_data = input_data.to(device)
            label = label.to(device)
            fake_img = G(label, torch.randn(label.size()))
            predict_fake, _ = D(fake_img)
            predict_true, _ = D(input_data)
            fake_img = torch.reshape(fake_img, (512, 512)).cpu().detach().numpy() * 255
            label_img = torch.reshape(label, (512, 512)).cpu().detach().numpy() * 255
            input_img = torch.reshape(input_data, (512, 512)).cpu().detach().numpy() * 255
            predict_fake = predict_fake.cpu().detach()
            predict_true = predict_true.cpu().detach()
            fake_img = np.array(Image.fromarray(fake_img), dtype='uint8')
            label_img = np.array(Image.fromarray(label_img), dtype='uint8')
            input_img = np.array(Image.fromarray(input_img), dtype='uint8')

            filename_fake = os.path.join(result_path, "fake_{}_{}.png".format(i, predict_fake[0][0]))
            filename_label = os.path.join(result_path, "label_{}.png".format(i))
            filename_input = os.path.join(result_path, "input_{}_{}.png".format(i, predict_true[0][0]))
            cv2.imwrite(filename_fake, fake_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(filename_label, label_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(filename_input, input_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 保存模型
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print("模型已保存")
