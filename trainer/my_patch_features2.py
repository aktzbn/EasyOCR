import pathlib
import torch
import yaml
import importlib

from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patch_model(src_model1: pathlib.Path, src_model1_config_path: pathlib.Path, 
                src_model2_config_path: pathlib.Path,
                destination: pathlib.Path):

    with open(src_model1_config_path, 'r', encoding="utf8") as stream1:
        opt1 = yaml.safe_load(stream1)

    with open(src_model2_config_path, 'r', encoding="utf8") as stream2:
        opt2 = yaml.safe_load(stream2)

    model_pkg = importlib.import_module("easyocr.model.vgg_model")

    network_params = {
        'input_channel': 1,
        'output_channel': 256,
        'hidden_size': 256
    }

    # FeatureExtraction

    pretrained_dict1 = torch.load(str(src_model1))
    num_class1 = len(opt1['character_list'])
    model1 = model_pkg.Model(num_class=num_class1, **network_params)
    model1.load_state_dict(pretrained_dict1)

    num_class2 = len(opt2['character_list'])
    model2 = model_pkg.Model(num_class=num_class2, **network_params)
    model2.FeatureExtraction = model1.FeatureExtraction

    torch.save(model2.state_dict(), destination)
    pass

if __name__ == '__main__':
    patch_model('D:\\Temp\\2023_05_09_fs_0.pth', 'D:\\Temp\\2023_05_09_fs_0.yaml', 'D:\\Temp\\2023_05_18_htr_pl_0_0.yaml', 'D:\\Temp\\xxx.pth')

