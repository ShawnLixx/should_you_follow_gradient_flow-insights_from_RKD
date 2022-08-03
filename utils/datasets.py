import torchvision
import torch
import torch.utils.data
from torchvision import transforms


def get_dataloader(args):
    # dataset
    if args.dataset in ('cifar10-5k', 'cifar10'):
        # transform
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2470, 0.2435, 0.2616))
             ])

        # we store the tensors to another dataset
        # to keep it in cuda (avoiding transfering cost).
        temp_dataset = torchvision.datasets.CIFAR10(
                root=args.dataset_dir,
                train=True,
                download=True,
                transform=transform)
        temp_loader = torch.utils.data.DataLoader(temp_dataset,
                batch_size=len(temp_dataset),
                shuffle=False)
        for data, label in temp_loader:
            tensor_data = data.cuda() if args.cuda else data
            tensor_label = label.cuda() if args.cuda else label
        dataset = torch.utils.data.TensorDataset(tensor_data, tensor_label)
        dataset_5k = torch.utils.data.Subset(dataset,
                indices=list(range(5000)))
        # test dataset
        temp_dataset = torchvision.datasets.CIFAR10(
                root=args.dataset_dir,
                train=False,
                download=False,
                transform=transform)
        temp_loader = torch.utils.data.DataLoader(temp_dataset,
                batch_size=len(temp_dataset),
                shuffle=False)
        for data, label in temp_loader:
            tensor_data = data.cuda() if args.cuda else data
            tensor_label = label.cuda() if args.cuda else label
        test_dataset = torch.utils.data.TensorDataset(tensor_data, tensor_label)
        if args.dataset == 'cifar10-5k':
            dataset = dataset_5k

        if args.optim != 'SGD':
            args.batch_size = len(dataset)
        train_loader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size,
                shuffle=args.optim == 'SGD')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size=len(test_dataset) if args.chunk_size is None \
                        else args.chunk_size,
                shuffle=False)
        hessian_batch_loader = torch.utils.data.DataLoader(dataset_5k,
                batch_size=args.hessian_batch_size,
                shuffle=False)

    else:
        raise NotImplementedError

    return train_loader, test_loader, hessian_batch_loader
