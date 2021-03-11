from train import load_data
from train import auc
import torch
import os
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import utils
# from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import numpy as np

try:
    from apex import amp
except ImportError:
    amp = None


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")
    utils.init_distributed_mode(args)
    print(args)

    # device = torch.device(args.device)
    # model.to(device)
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Creating model")
    num_classes = len(data_loader.dataset.classes)

    print('creat mobilenet')
    mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=args.pretrained)
    mobilenet_v2.classifier = nn.Sequential(
        nn.Linear(1280, num_classes),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
    )
    mobilenet_v2.to(device)

    mobilenet_v2_checkpoint = torch.load(args.mobilenet_v2_resume, map_location='cpu')
    mobilenet_v2.load_state_dict(mobilenet_v2_checkpoint['model'])

    print('creat densenet161')
    densenet161 = torchvision.models.densenet161(pretrained=args.pretrained)
    densenet161.classifier = nn.Sequential(
                nn.Linear(2208, num_classes),
            )
    densenet161.to(device)

    densenet161_checkpoint = torch.load(args.densenet161_resume, map_location='cpu')
    densenet161.load_state_dict(densenet161_checkpoint['model'])

    print('predection')
    average_precision_dict, recall_dict, precision_dict = auc(mobilenet_v2, data_loader_test, num_class=num_classes)
    average_precision_dict2, recall_dict2, precision_dict2 = auc(densenet161, data_loader_test, num_class=num_classes)
    # 绘制所有类别平均的pr曲线
    x = list(np.arange(0., 1.2, 0.2))
    y = x
    print('show picture')
    plt.figure()
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post', color='r', label='mobilenet_v2')
    plt.step(recall_dict2['micro'], precision_dict2['micro'], where='post', color='g', label='densenet161')
    plt.plot(x, y, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.title(
        'Average precision score, micro-averaged over all classes:\n'
        ' AP_mobile={0:0.2f}, AP_dens={0:0.2f}'.format(
            average_precision_dict["micro"], average_precision_dict2["micro"]))
    plt.savefig("set113_pr_curve.jpg")
    plt.show()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='./rpc_data', help='dataset')
    parser.add_argument('--model', default='mobilenet_v2', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./resnet50_model', help='path where to save')
    parser.add_argument('--resume', default='./mobilenet_v2_model/model_6.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--auc', default=True, help='start epoch')
    parser.add_argument('--mobilenet_v2_resume', default='./mobilenet_v2_model/model_6.pth',
                        help='resume from checkpoint')
    parser.add_argument('--densenet161_resume', default='./densenet161_model/model_8.pth',
                        help='resume from checkpoint')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

