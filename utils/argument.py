import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    # whether to use gpu
    parser.add_argument('--gpu', dest='cuda', action='store_true', help='use gpu')
    parser.add_argument('--cpu', dest='cuda', action='store_false', help='use cpu')
    parser.set_defaults(cuda=True)
    # training ralated
    parser.add_argument('--dataset', type=str, choices=['cifar10-5k', 'cifar10'], default='cifar10-5k', help='choose dataset')
    parser.add_argument('--lr_denom', type=int, default=50, help='learning rate denominator. 2 / denom')
    parser.add_argument('--epoch', type=int, default=7000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--hessian_batch_size', type=int, default=5000, help='batch size used to evaluate hessian')
    parser.add_argument('--optim', type=str, choices=['GD', 'SGD', 'Adam', 'RMSprop', 'RK2', 'RK4'], default='GD', help='optimization method')
    parser.add_argument('--momentum', type=float, default=0, help='momentum parameter for GD, SGD and RMSprop optimizer')
    parser.add_argument('--nesterov', dest='nesterov', action='store_true', help='use Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--rms_alpha', type=float, default=0.99, help='smoothing parameter for RMSprop optimizer')
    parser.add_argument('--adam_betas', type=float, nargs='+', default=[0.9, 0.999], help='betas parameters for RMSprop optimizer')
    parser.add_argument('--chunk_size', type=int, default=None, help='Size of each chunk for large batch.')
    parser.add_argument('--eval_gap', type=int, default=100, help='Sharpness and test error will be calculated every eval_gap iterations.')
    # model related
    parser.add_argument('--model', type=str, default='FCN', help='model to be trained')
    parser.add_argument('--loss', type=str, choices=['CE', 'MSE'], default='CE', help='loss function to be used')
    parser.add_argument('--fix_up', dest='fix_up', action='store_true', help='Use fix up init when training resnet')
    parser.set_defaults(fix_up=False)
    # directory related
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='dataset directory')
    parser.add_argument('--save_dir', type=str, default='./logs', help='directory to save the experimental result')
    parser.add_argument('--model_name', type=str, default=None, help='sub-directory name. If none, automatically generated by arguments')
    # others
    parser.add_argument('--random_seed', type=int, default=8, help='random seed for reproducibility')

    # parse
    args = parser.parse_args()

    # check whether valid
    assert(args.lr_denom >= 0)
    assert(args.epoch >= 0)
    assert(args.batch_size >= 0)
    assert(args.hessian_batch_size >= 0)

    if args.model in ('CNN'):
        raise NotImplementedError

    return args
