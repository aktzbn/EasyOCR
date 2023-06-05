import pathlib
import torch
import yaml
import importlib
import torch.nn.init as init

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
    num_class1 = len(opt1['character_list']) + 1
    model1 = model_pkg.Model(num_class=num_class1, **network_params)

    # model1 = torch.nn.DataParallel(model1).to(device)
    # model1.load_state_dict(pretrained_dict1)

    # Strict = False to fix error during loading of dict
    # where keys contains 'module.'
    model1.load_state_dict(pretrained_dict1, strict=False)

    num_class2 = len(opt2['character_list']) + 1
    model2 = model_pkg.Model(num_class=num_class2, **network_params)

    for name, param in model2.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    model2.FeatureExtraction = model1.FeatureExtraction
    model2.SequenceModeling = model1.SequenceModeling

    torch.save(model2.state_dict(), destination)
    pass

if __name__ == '__main__':
    patch_model('D:\\Temp\\2023_05_09_fs_0.pth', 'D:\\Temp\\2023_05_09_fs_0.yaml', 'D:\\Temp\\2023_05_18_htr_pl_0_0.yaml', 'F:\\Aktz\\Projects\\hryc\\ai\\htr_pl\\start_model\\ru_pl_features_seq.pth')

