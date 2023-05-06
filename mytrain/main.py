# from AttentionUNet.AttentionUNet import AttentionUNet
import os

import torch
from ModelClass.myModel import Model
from ModelClass.modelTrainer import ModelTrainer
from ModelClass.modelTester import ModelTester

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = Model()
    # model_path = "./AttentionUNet/MultiTaskAttentionUNet.py"
    dict_path = "./AttentionUNet51.pth"
    # model_path = "./UNet/UNet.py"
    # dict_path = "./UNet.pth"
    model_path = "./AttentionUNet/AttentionUNet.py"
    # dict_path = "./AttentionUNet/AttentionUNet30.pth"
    net.load_model(model_path, has_net=True, dict_path=dict_path, multiple_gpu=False)

    trainer = ModelTrainer()
    trainer.set_model(net)
    # raw_path = "./HeartData/small/train/raw/"
    # label_path = "./HeartData/small/train/label/"
    raw_path = "./HeartData/test/raw/"
    label_path = "./HeartData/test/label/"
    # trainer.load_train_data(raw_path, label_path, "nii", max_size=44)
    # trainer.train_model(10, 4, 0.01,num_workers=0,optim="SGD")

    tester = ModelTester()
    tester.set_model(net)
    # raw_path = "./HeartData/small/test/raw/"
    # label_path = "./HeartData/small/test/label/"
    # result_path = "./HeartData/small/test/result/"
    raw_path = "./HeartData/test/raw/"
    label_path = "./HeartData/test/label/"
    result_path = "./HeartData/test/result/"
    tester.load_test_data(raw_path, label_path, "nii", max_size=44,remove_black=True)
    tester.test_model(result_path)
    # net.save_model("./AttentionUNet51.pth")
