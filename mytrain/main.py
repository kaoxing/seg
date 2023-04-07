# from AttentionUNet.AttentionUNet import AttentionUNet
import torch
from ModelClass.myModel import Model
from ModelClass.modelTrainer import ModelTrainer
from ModelClass.modelTester import ModelTester

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = Model()
    # model_path = "./AttentionUNet/MultiTaskAttentionUNet.py"
    # dict_path = "./AttentionUNet.pth"
    model_path = "./UNet/UNet.py"
    dict_path = "./UNet.pth"
    net.load_model(model_path, has_net=True, dict_path=dict_path, multiple_gpu=True)
    trainer = ModelTrainer()
    trainer.set_model(net)
    raw_path = "./data/train/raw/"
    label_path = "./data/train/label/"
    # raw_path = "./data/test/raw/"
    # label_path = "./data/test/label/"
    trainer.load_train_data(raw_path, label_path)
    trainer.train_model(20, 8, 0.00001)
    tester = ModelTester()
    tester.set_model(net)
    raw_path = "./data/test/raw/"
    label_path = "./data/test/label/"
    result_path = "./data/test/result/"
    tester.load_test_data(raw_path, label_path)
    tester.test_model(result_path)
    net.save_model("./UNet.pth")
