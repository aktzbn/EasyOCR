# Need to be located in trainer folder
# https://stackoverflow.com/a/59403771/2064680
if __name__ == '__main__':
    import os
    import torch.backends.cudnn as cudnn
    import yaml
    from train import train
    from utils import AttrDict
    import pandas as pd
    import sys

    cudnn.benchmark = True
    cudnn.deterministic = False

    def get_config(file_path):
        with open(file_path, 'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        if opt.lang_char == 'None':
            characters = ''
            for data in opt['select_data'].split('-'):
                csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
                df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
                all_char = ''.join(df['words'])
                characters += ''.join(set(all_char))
            characters = sorted(set(characters))
            opt.character= ''.join(characters)
        else:
            opt.character = opt.number + opt.symbol + opt.lang_char
        os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
        return opt

    # Run training
    if len(sys.argv) < 2:
        print("Usage: mytrain <PATH_TO_CONFIG>")
        sys.exit(1)
    model=sys.argv[1]
    print("model: " + model)
    opt = get_config(model)
    train(opt, amp=False, show_number=4)