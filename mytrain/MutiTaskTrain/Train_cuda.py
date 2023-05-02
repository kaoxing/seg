import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from MyGAN import MyGAN as GAN
# from MultiTaskAttentionUNet import MultiTasAttentionUNet as UNet
from Discriminator import MultiTasAttentionUNet as UNet
from DatasetGAN import MyDataSetPre
from DatasetUNet import MyDataSetTra
from torch.utils.data import DataLoader
import LossFunction as lf

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 获取参数
    # argv = sys.argv[1:]
    # print(argv)
    # batch_size = int(input("batch_size="))
    batch_size = 1
    # learning_rate = float(input("learning_rate="))
    learning_rate_g = 0.006
    learning_rate_d = 0.04
    # epoch = int(input("epoch="))
    epoch = 10
    clip = 0.01
    n_critic = 2
    # result_path = os.path.join("./dataset/result/", input("result_dir="))
    result_path = os.path.join("./dataset/result/", "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    label_path = "./dataset/train/label/"
    raw_path = "./dataset/train/raw/"
    # d_path = os.path.join("./models/d/", input("d_name="))
    # g_path = os.path.join("./models/g/", input("g_name="))
    d_path = os.path.join("./models/d/", "10_d.pth")
    g_path = os.path.join("./models/g/", "10_g.pth")

    # 加载模型
    D = UNet()
    # D.load_state_dict(torch.load(os.path.join("./models/d/", "30_d.pth")))
    total = sum([param.nelement() for param in D.parameters()])
    print("Parameter of Dis: %.2fK" % (total / 1e3))
    D.to(device)
    G = GAN()
    total = sum([param.nelement() for param in G.parameters()])
    print("Parameter of Gen: %.2fM" % (total / 1e6))
    # G.load_state_dict(torch.load(os.path.join("./models/g/", "30_g.pth")))
    G.to(device)

    # 加载数据集
    dataset = MyDataSetTra(raw_path, label_path)
    dataset = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    dataset_test = MyDataSetTra(raw_path, label_path)
    dataset_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
    )
    # 优化器定义
    optimizer_G = torch.optim.SGD(
        G.parameters(),
        lr=learning_rate_g
    )
    optimizer_D = torch.optim.SGD(
        D.parameters(),
        lr=learning_rate_d
    )
    # lrd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=5, eta_min=0.000005)
    # lrg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=5, eta_min=0.00001)
    # 损失函数定义
    # loss_D = DLossFunction()
    # loss_G = GLossFunction()
    # lossFuncD = torch.nn.BCELoss()
    # lossFuncG = torch.nn.BCELoss()
    lossFuncD = torch.nn.BCEWithLogitsLoss()
    lossFuncG = lf.Wasserstein()

    # 训练
    for cnt in range(epoch):
        print("迭代轮次[{}/{}]".format(cnt + 1, epoch))
        Loss_D = 0
        Loss_G = 0
        for i, data in enumerate(dataset):
            input_data, labels = data
            input_data = input_data.to(device)
            labels = labels.to(device)
            rand = np.random.normal(loc=0.0, scale=0.1, size=input_data.size()).astype(np.float32)
            rand = torch.tensor(rand).to(device)
            optimizer_D.zero_grad()  # 梯度置零
            # 输入假数据
            fake_img = G(labels, rand)  # 数据输入生成网络
            predict_fake = D(fake_img.detach())  # 生成假数据送入辨别网络
            # print("predict_fake:", predict_fake)
            loss_d = lossFuncD(predict_fake, torch.zeros_like(predict_fake))
            # print("Loss:", loss_d)
            loss_d.backward()
            Loss_D += loss_d.cpu().detach().numpy().sum()
            optimizer_D.step()
            optimizer_D.zero_grad()
            predict_real = D(input_data)
            # print("predict_real:", predict_real)
            loss_d = lossFuncD(predict_real, torch.ones_like(predict_real))
            # print("Loss:", loss_d)
            loss_d.backward()
            optimizer_D.step()
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                fake_img = G(labels, rand)
                predict_fake = D(fake_img)
                # print(predict_fake)
                loss_g = lossFuncG(predict_fake)
                # print("loss_g", loss_g)
                loss_g.backward()
                optimizer_G.step()
                Loss_G += loss_g.cpu().detach().numpy().sum()
            Loss_D += loss_d.cpu().detach().numpy().sum()
            # print("loss_g.item()", loss_g.item())
            # print("loss_d_fake.item():", loss_d_fake.item())
            # print("loss_d_true.item():", loss_d_true.item())
        print("loss_g:", Loss_G)
        print("loss_d:", Loss_D)

    # 测试
    with torch.no_grad():
        for i, data in enumerate(dataset_test):
            input_data, label = data
            input_data = input_data.to(device)
            label = label.to(device)
            fake_img = G(label, torch.randn(label.size()).to(device))
            predict_fake = D(fake_img)
            predict_true = D(input_data)
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
    # torch.save(G.state_dict(), g_path)
    # torch.save(D.state_dict(), d_path)
    print("模型已保存")
