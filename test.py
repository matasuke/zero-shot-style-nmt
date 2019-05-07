import os
import json
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.translator as module_translator
from train import get_instance


def main(config, test_config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        src_path=test_config['data_loader']['args']['src_path'],
        tgt_path=test_config['data_loader']['args']['tgt_path'],
        src_preprocessor_path=config['data_loader']['args']['src_preprocessor_path'],
        tgt_preprocessor_path=config['data_loader']['args']['tgt_preprocessor_path'],
        batch_size=test_config['data_loader']['args']['batch_size'],
        num_workers=test_config['data_loader']['args']['num_workers'],
        shuffle=False,
        validation_split=0.0,
    )
    # data_loader = get_instance(module_data, 'data_loader', test_config)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if test_config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans_args = {
        'model': model,
        'src_preprocessor': data_loader.src_text_preprocessor,
        'tgt_preprocessor': data_loader.tgt_text_preprocessor,
    }
    test_config['sampler']['args'] = {**trans_args, **test_config['sampler']['args']}

    translator = get_instance(module_translator, 'sampler', test_config)
    model = get_instance(module_arch, 'arch', config)
    # model.summary()

    pred_score_total, pred_words_total, gold_score_total, gold_words_total = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (src, tgt, lengths) in enumerate(tqdm(data_loader)):
            src, tgt = src.to(device), tgt.to(device)
            pred_batch, pred_score, gold_score = translator.translate(src, tgt, lengths)
            # save sample images, or do something with output here

            pred_score_total += sum(score[0] for score in pred_score)
            pred_words_total += sum(len(x[0]) for x in pred_batch)
            gold_score_total += sum(gold_score)
            gold_words_total += sum(len(x) for x in tgt)

            for b in range(len(pred_batch)):
                pass

    # print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str, required=True,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-c', '--test_config', default=None, type=str, required=True,
                        help='path to config for testing')
    parser.add_argument('-d', '--device', default=[], type=str, nargs='+', required=True,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.device)

    config = torch.load(args.resume)['config']

    with open(args.test_config, 'r') as f:
        test_config = json.load(f)

    main(config, test_config, args.resume)
