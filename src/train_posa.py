# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import os.path as osp
import torch
import time
import yaml
from torch.utils.data import DataLoader
from src.data_parser import NPYDataset
import src.posa_models as models
from src.cmd_parser import parse_config
from src import misc_utils


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    model.train()
    total_recon_loss_dist = 0
    total_recon_loss_semantics = 0
    total_semantics_recon_acc = 0
    total_recon_loss = 0
    total_KLD_loss = 0
    total_train_loss = 0

    for batch_idx, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        x = data['x'].to(device)
        gt_batch = x
        vertices_can = data['vertices_can'].to(device)
        if args.use_semantics:
            x_semantics = data['x_semantics'].to(device)
            gt_batch = torch.cat((gt_batch, x_semantics), dim=-1)

        optimizer.zero_grad()

        pr_batch, mu, logvar = model(gt_batch, vertices_can)
        recon_loss_dist, recon_loss_semantics, semantics_recon_acc = misc_utils.compute_recon_loss(gt_batch, pr_batch,
                                                                                                   **args_dict)
        recon_loss = recon_loss_dist + recon_loss_semantics
        if args.kl_w > 0:
            KLD = args.kl_w * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / args.batch_size
        else:
            KLD = torch.zeros(1, dtype=dtype, device=device)
        loss = KLD + recon_loss

        loss.backward()
        optimizer.step()

        total_recon_loss_dist += recon_loss_dist.item()
        total_recon_loss_semantics += recon_loss_semantics.item()
        total_semantics_recon_acc += semantics_recon_acc.item()
        total_recon_loss += recon_loss.item()
        total_train_loss += loss.item()
        total_KLD_loss += KLD.item()

    total_recon_loss_dist /= n_batches_train
    total_recon_loss_semantics /= n_batches_train
    total_recon_loss /= n_batches_train
    total_train_loss /= n_batches_train
    total_semantics_recon_acc /= n_batches_train

    total_KLD_loss /= n_batches_train
    if args.tensorboard:
        writer.add_scalar('recon_loss_dist/train', total_recon_loss_dist, epoch)
        writer.add_scalar('recon_loss_semantics/train', total_recon_loss_semantics, epoch)
        writer.add_scalar('total_semantics_recon_acc/train', total_semantics_recon_acc, epoch)
        writer.add_scalar('recon_loss/train', total_recon_loss, epoch)
        writer.add_scalar('total/train_total_loss', total_train_loss, epoch)
        writer.add_scalar('KLD/train_KLD', total_KLD_loss, epoch)

    print(
        '====> Total_train_loss: {:.4f}, Recon_loss = {:.4f}, KLD = {:.4f}, Recon_loss_dist = {:.4f}, Recon_loss_semantics = {:.4f} , Semantics_recon_acc = {:.4f}'.format(
            total_train_loss, total_recon_loss, total_KLD_loss, total_recon_loss_dist, total_recon_loss_semantics,
            total_semantics_recon_acc))
    return total_train_loss


def test():
    model.eval()
    with torch.no_grad():
        total_recon_loss_dist = 0
        total_recon_loss_semantics = 0
        total_semantics_recon_acc = 0
        total_recon_loss = 0
        total_KLD_loss = 0
        total_test_loss = 0
        for batch_idx, data in enumerate(test_data_loader):
            x = data['x'].to(device)
            gt_batch = x
            vertices_can = data['vertices_can'].to(device)
            if args.use_semantics:
                x_semantics = data['x_semantics'].to(device)
                gt_batch = torch.cat((gt_batch, x_semantics), dim=-1)

            pr_batch, mu, logvar = model(gt_batch, vertices_can)
            recon_loss_dist, recon_loss_semantics, semantics_recon_acc = misc_utils.compute_recon_loss(gt_batch,
                                                                                                       pr_batch,
                                                                                                       **args_dict)
            recon_loss = recon_loss_dist + recon_loss_semantics
            if args.kl_w > 0:
                KLD = args.kl_w * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / args.batch_size
            else:
                KLD = torch.zeros(1, dtype=dtype, device=device)
            loss = KLD + recon_loss

            total_recon_loss_dist += recon_loss_dist.item()
            total_recon_loss_semantics += recon_loss_semantics.item()
            total_semantics_recon_acc += semantics_recon_acc.item()
            total_recon_loss += recon_loss.item()

            total_KLD_loss += KLD.item()
            total_test_loss += loss.item()

        total_recon_loss_dist /= n_batches_test
        total_recon_loss_semantics /= n_batches_test
        total_semantics_recon_acc /= n_batches_test
        total_recon_loss /= n_batches_test
        total_KLD_loss /= n_batches_test
        total_test_loss /= n_batches_test
        if args.tensorboard:
            writer.add_scalar('recon_loss_dist/test', total_recon_loss_dist, epoch)
            writer.add_scalar('recon_loss_semantics/test', total_recon_loss_semantics, epoch)
            writer.add_scalar('total_semantics_recon_acc/test', total_semantics_recon_acc, epoch)
            writer.add_scalar('recon_loss/test', total_recon_loss, epoch)
            writer.add_scalar('total/test_total_loss', total_test_loss, epoch)
            writer.add_scalar('KLD/test_KLD', total_KLD_loss, epoch)

        print(
            '====> Total_test_loss: {:.4f}, Recon_loss = {:.4f}, KLD = {:.4f}, Recon_loss_dist = {:.4f}, Recon_loss_semantics = {:.4f}, Semantics_recon_acc = {:.4f}'.format(
                total_test_loss, total_recon_loss, total_KLD_loss, total_recon_loss_dist, total_recon_loss_semantics,
                total_semantics_recon_acc))

        return total_test_loss


if __name__ == '__main__':
    torch.manual_seed(0)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        args, args_dict = parse_config()
        args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))
        args_dict['output_dir'] = osp.expandvars(args_dict.get('output_dir'))
        args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
        output_dir = args_dict.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)
        checkpoints_dir = osp.join(output_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        log_dir = osp.join(output_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)

        conf_fn = osp.join(output_dir, 'conf.yaml')
        with open(conf_fn, 'w') as conf_file:
            # delete all None arguments
            conf_data = vars(args)
            keys = [x for x in conf_data.keys() if conf_data[x] is None]
            for key in keys:
                del (conf_data[key])
            yaml.dump(conf_data, conf_file)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)

        device = torch.device("cuda" if args.use_cuda else "cpu")
        dtype = torch.float32

        model = models.load_model(**args_dict).to(device)
        print('Num of trainable param = {}'.format(models.count_parameters(model)))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epoch = 1
        if args.load_checkpoint > 0:
            print('loading stats of epoch {}'.format(args.load_checkpoint))
            checkpoint = torch.load(osp.join(osp.join(args_dict['output_dir'], 'checkpoints'),
                                             'epoch_{:04d}.pt'.format(args.load_checkpoint)))
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch'] + 1

        args_dict['train_data'] = 1
        train_dataset = NPYDataset(**args_dict)
        train_data_loader = DataLoader(train_dataset, batch_size=args_dict.get('batch_size'),
                                       num_workers=args_dict.get('num_workers'), shuffle=args.shuffle,
                                       drop_last=True)
        train_set_size = len(train_dataset)
        n_batches_train = train_set_size // args.batch_size
        print('Number of training example: {}, batches: {}'.format(train_set_size, n_batches_train))
        if args.test:
            args_dict['train_data'] = 0
            test_dataset = NPYDataset(**args_dict)
            test_data_loader = DataLoader(test_dataset, batch_size=args_dict.get('batch_size'),
                                          num_workers=args_dict.get('num_workers'), shuffle=args.shuffle,
                                          drop_last=True)
            test_set_size = len(test_dataset)
            n_batches_test = test_set_size // args.batch_size
            print('Number of testing example: {}, batches: {}'.format(test_set_size, n_batches_test))

        for epoch in range(epoch, args.epochs + 1):
            print('Training epoch {}'.format(epoch))
            start = time.time()
            total_train_loss = train()
            training_time = time.time() - start
            print('training_time = {:.4f}'.format(training_time))

            print('Testing epoch {}'.format(epoch))
            start = time.time()
            total_test_loss = test()
            testing_time = time.time() - start
            print('test_time = {:.4f}'.format(testing_time))

            if args.save_checkpoints and epoch % args.log_interval == 0:
                data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_train_loss': total_train_loss,
                    'total_test_loss': total_test_loss,
                }
                torch.save(data, osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(epoch)))
