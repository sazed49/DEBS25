from __future__ import print_function
from sklearn.cluster import dbscan

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
    log_dir = f'logfiles/{args.AR}/{args.dataset}/{args.loader_type}'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    FORMAT = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)s:: %(message)s'
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    FILENAME = '{0}/train_{1}_{2}_{3}_{4}_{5}.log'.format(log_dir, args.AR,
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
    logging.info('#####################')
    logging.info('#####################')
    logging.info('#####################')
    logging.info(f'Aggregation Rule:\t{args.AR}')
    logging.info(f'Data distribution:\t{args.loader_type}')
    logging.info(f'Attacks:\t{args.attacks} ')
    logging.info('#####################')
    logging.info('#####################')
    logging.info('#####################')

    torch.manual_seed(args.seed)

    device = args.device

    attacks = args.attacks

    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')

    if args.dataset == 'mnist':
        from models import mnist
        trainData = mnist.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar':
        from models import cifar as cifar
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
        Net = fashion_mnist.Net
        criterion = F.cross_entropy
    elif args.dataset == 'rockburst':
        from models import rockburst as r
        #print("args.num_clients ",args.num_clients)
        trainData = r.train_dataloader(args.num_clients,
            loader_type=args.loader_type, path=args.loader_path, store=False)
        print(">>>>>>>>>>>>>>This is the length of traindata",len(trainData))
        
        testData = r.test_dataloader(args.test_batch_size)
        print("<<<<<<<<<<<<<<<This is the length of testdata",len(testData))
        Net = r.ImprovedClassifier
        criterion = F.cross_entropy

    # create server instance
    model0 = Net()
    server = Server(model0, testData, criterion, device)
    server.set_AR(args.AR)
    if args.dataset == 'cifar':
        #server.set_AR_param(dbscan_eps=35.) # OK for untargeted attack, but does not detect potential targeted attack
        server.set_AR_param(dbscan_eps=18, min_samples=12) #20 is still OK without sign-flipping, 30. is OK without sign-flipping, min_samples=10 is failed.
    elif args.dataset == 'fashion_mnist':
        server.set_AR_param(dbscan_eps=1.3, min_samples=5)
    else :
        server.set_AR_param(dbscan_eps =1.3, min_samples=10)

    server.path_to_aggNet = args.path_to_aggNet
    '''
    honest clients are labeled as 1, malicious clients are labeled as 0
    '''
    label = torch.ones(args.num_clients)
    
    
    
    
    for i in args.list_uatk_add_noise:
        label[i] = 0
    for i in args.list_uatk_flip_sign:
        label[i] = 0
    
    logging.info("[1-normal, 0-malicious] label ={}".format(label))

    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        torch.save(label, f'{server.savePath}/label.pt')
    # create clients instance

    
    
    
    
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
            client_i = Sign_Flip_Attacker(i, model, trainData[i], optimizer,
                criterion, device, args.omniscient_scale, args.inner_epochs)
        elif i in args.list_uatk_add_noise:
            client_i = Additive_Noise_Attacker(i, model, trainData[i], optimizer,
                criterion, device, args.mean_add_noise, args.std_add_noise,
                args.inner_epochs)
        elif i in args.list_unreliable:
            channels = 3 if "cifar" in args.dataset else 1
            kernel_size=5 if 'fashion_mnist' in args.dataset else 7

            client_i = Unreliable_client(i, model, trainData[i], optimizer,
                criterion, device, args.mean_unreliable, 30,
                args.unreliable_fraction, args.unreliable_fracTrain,
                args.blur_method, args.inner_epochs, channels=channels,
                kernel_size=kernel_size)
        else:
            #print(trainData[i])
            client_i = Client(i, model, trainData[i], optimizer, criterion,
                device, args.inner_epochs)
        server.attach(client_i)

    server.set_log_path(log_dir, args.experiment_name, start_time)
    
    loss, accuracy = server.test()
    steps = 0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)

    if args.attacks and 'BACKDOOR' in args.attacks.upper():
        if 'SEMANTIC' in args.attacks.upper():
            loss, accuracy, bdata, bpred = server.test_semanticBackdoor()
        else:
            loss, accuracy = server.test_backdoor()

        writer.add_scalar('test/loss_backdoor', loss, steps)
        writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    for j in range(args.epochs):
        steps = j + 1

        logging.info('########EPOCH %d ########' % j)
        logging.info('###Model distribution###')
        server.distribute()
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)
        print("group------", group)
        server.train(group)
        #         server.train_concurrent(group)

        loss, accuracy = server.test()

        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)

        if args.attacks and 'BACKDOOR' in args.attacks.upper():
            if 'SEMANTIC' in args.attacks.upper():
                loss, accuracy, bdata, bpred = server.test_semanticBackdoor()
            else:
                loss, accuracy = server.test_backdoor()

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    server.close()
    writer.close()