from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import logging
from datetime import datetime

from malicious_clients import *
from server import Server

def main(args):
    results_directory=f'results/{args.AR}/{args.dataset}/{args.loader_type}'
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)
    FORMAT = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)s:: %(message)s'
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    FILENAME = '{0}/train_{1}_{2}_{3}_{4}_{5}.log'.format(results_directory, args.AR,
        args.dataset, args.loader_type, args.experiment_name, start_time)
    LOG_LVL = logging.DEBUG if args.verbose else logging.INFO

    fileHandler = logging.FileHandler(FILENAME, mode='w')
    fileHandler.setFormatter(logging.Formatter(FORMAT))
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger('')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(LOG_LVL)
    logging.info("#" * 64)

    for i in vars(args):
        logging.info(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    logging.info("#" * 64)
    logging.info(args)
    logging.info('*********************')
    logging.info('*********************')
    
    logging.info(f'Algorithm Used:\t{args.AR}')
    logging.info(f'Data distribution:\t{args.loader_type}')
    logging.info(f'Attacks:\t{args.attacks} ')
    logging.info('*********************')
    logging.info('*********************')

    torch.manual_seed(args.seed)

    device=args.device
    attacks=args.attacks
    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')


    if args.dataset == 'mnist':
        from models import mnist
        trainData = mnist.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar':
        from models import cifar10 as cifar
        trainData = cifar.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar100':
        from models import cifar100
        trainData = cifar100.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = cifar100.test_dataloader(args.test_batch_size)
        Net = cifar100.Net
        criterion = F.cross_entropy
    elif args.dataset == 'imdb':
        from models import imdb
        trainData = imdb.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = imdb.test_dataloader(args.test_batch_size)
        Net = imdb.Net
        criterion = F.cross_entropy
    elif args.dataset == 'fashion_mnist':
        from models import fashion_mnist
        trainData = fashion_mnist.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = fashion_mnist.test_dataloader(args.test_batch_size)
        Net = fashion_mnist.RockburstNet
        criterion = F.cross_entropy
    elif args.dataset == 'custom':
        from models import custom as c
        trainData = c.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = c.test_dataloader(args.test_batch_size)
        Net = c.RockburstNet
        criterion = F.cross_entropy
    
    model=Net()
    server=Server(model,testData,criterion,device)
    server.set_AR(args.AR)
    if args.dataset == 'cifar':
        #server.set_AR_param(dbscan_eps=35.) # OK for untargeted attack, but does not detect potential targeted attack
        server.set_AR_param(dbscan_eps=18, min_samples=12) #20 is still OK without sign-flipping, 30. is OK without sign-flipping, min_samples=10 is failed.
    elif args.dataset == 'fashion_mnist':
        server.set_AR_param(dbscan_eps=1.3, min_samples=5)

    server.path_to_aggNet = args.path_to_aggNet

    label = torch.ones(args.num_clients)
    for i in args.attacker_list_labelFlipping:
        label[i] = 0
    for i in args.attacker_list_labelFlippingDirectional:
        label[i] = 0
    for i in args.attacker_list_omniscient:
        label[i] = 0
    for i in args.attacker_list_backdoor:
        label[i] = 0
    for i in args.attacker_list_semanticBackdoor:
        label[i] = 0
    for i in args.list_uatk_add_noise:
        label[i] = 0
    for i in args.list_uatk_flip_sign:
        label[i] = 0
    for i in args.list_tatk_multi_label_flipping:
        label[i] = 0
    for i in args.list_tatk_label_flipping:
        label[i] = 0
    for i in args.list_tatk_backdoor:
        label[i] = 0

    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        torch.save(label, f'{server.savePath}/label.pt')
    # attacker_list_labelFlipping = args.attacker_list_labelFlipping
    # attacker_list_omniscient = args.attacker_list_omniscient
    # attacker_list_backdoor = args.attacker_list_backdoor
    # attacker_list_labelFlippingDirectional = args.attacker_list_labelFlippingDirectional
    # attacker_list_semanticBackdoor = args.attacker_list_semanticBackdoor
    for i in range(args.num_clients):
        model = Net()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        if i in args.list_uatk_flip_sign:
            client_i = Attacker_Omniscient(i, model, trainData[i], optimizer,
                criterion, device, args.omniscient_scale, args.inner_epochs)
        elif i in args.list_uatk_add_noise:
            client_i = Attacker_AddNoise_Grad(i, model, trainData[i], optimizer,
                criterion, device, args.mean_add_noise, args.std_add_noise,
                args.inner_epochs)
        elif i in args.list_unreliable:
            channels = 3 if "cifar" in args.dataset else 1
            kernel_size=5 if 'fashion_mnist' in args.dataset else 7

            client_i = Unreliable_client(i, model, trainData[i], optimizer,
                criterion, device, args.mean_unreliable, args.max_std_unreliable,
                args.unreliable_fraction, args.unreliable_fracTrain,
                args.blur_method, args.inner_epochs, channels=channels,
                kernel_size=kernel_size)
        else:
            client_i = Client(i, model, trainData[i], optimizer, criterion,
                device, args.inner_epochs)
        server.attach(client_i)
    server.set_log_path(results_directory, args.experiment_name, start_time)
    loss, accuracy = server.test()
    steps = 0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)

    for j in range(args.epochs):
        steps = j + 1

        logging.info('########EPOCH %d ########' % j)
        logging.info('###Model distribution###')
        server.distribute()
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)
        print("----group---",group)
        server.train(group)
        #         server.train_concurrent(group)

        loss, accuracy = server.test()

        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)
    server.close()
    writer.close()