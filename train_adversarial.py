import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision.utils import save_image
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time
import matplotlib.pyplot as plt
from custom_dataset.unaligned_dataset import UnAlignedDataSet
from glow_adversarial import *
import options
from loss.penalty import gradient_penalty

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


args = options.train_parser.parse_args()

# DDP
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

# dataloader
train_dataset = UnAlignedDataSet(args)
train_sampler = DistributedSampler(train_dataset)
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, sampler=train_sampler, drop_last=True)
content_examples = []
for i in range(args.n_save_img):
    content_examples.append(train_dataset[i]["content"].unsqueeze(0).to(local_rank))
content_examples = torch.cat(content_examples, dim=0)
z_examples = torch.randn((args.n_save_img, args.z_length))

# models
netG = NetG(
    args.img_size,
    args.input_channel,
    args.n_flow,
    args.n_block,
    args.affine,
    ~args.no_lu,
).to(local_rank)
netG = nn.parallel.DistributedDataParallel(netG, device_ids=[local_rank], output_device=local_rank)
netD = NetD(
    args.img_size,
    args.input_channel,
    args.dc_max,
    args.filter
).to(local_rank)
netD = nn.parallel.DistributedDataParallel(netD, device_ids=[local_rank], output_device=local_rank)

# optimizer
optim_G = torch.optim.Adam(netG.parameters(), lr=args.lr_G, betas=(args.beta1_G, args.beta2_G))
optim_D = torch.optim.Adam(netD.parameters(), lr=args.lr_D, betas=(args.beta1_D, args.beta2_D))

# schedule
schedule_G = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)
schedule_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)



if __name__ == '__main__':
    if local_rank == 0 and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    checkpoints_save_dir = os.path.join(args.save_dir, "checkpoints/")
    if local_rank == 0 and not os.path.exists(checkpoints_save_dir):
        os.mkdir(checkpoints_save_dir)
    images_save_dir = os.path.join(args.save_dir, "images/")
    if local_rank == 0 and not os.path.exists(images_save_dir):
        os.mkdir(images_save_dir)
    # -------------------------------------------------------
    if local_rank == 0:
        argsDict = args.__dict__
        with open(os.path.join(args.save_dir, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    # ---------------------resume training-------------------
    if args.resume_G:
        assert os.path.isfile(args.resume_G), "--------no checkpoint found---------"
        print("--------loading checkpoint----------")
        print("=> loading checkpoint '{}'".format(args.resume_G))
        checkpoint_G = torch.load(args.resume_G)
        netG.module.load_state_dict(checkpoint_G['state_dict'])
        optim_G.load_state_dict(checkpoint_G['optimizer'])
    if args.resume_D:
        assert os.path.isfile(args.resume_D), "--------no checkpoint found---------"
        print("--------loading checkpoint----------")
        print("=> loading checkpoint '{}'".format(args.resume_D))
        checkpoint_D = torch.load(args.resume_D)
        netD.module.load_state_dict(checkpoint_D['state_dict'])
        optim_D.load_state_dict(checkpoint_D['optimizer'])
    # -----------------------training------------------------
    G_losses = []
    D_losses = []
    G_loss = 0.0
    D_loss = 0.0
    for epoch in range(args.start_epoch, args.max_epoch + 1):
        #  if training from scratch, epoch starts from 1 rather than 0
        time_start = time.time()
        g_losses = []
        d_losses = []
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_dataloader):
            z = torch.randn([args.batch_size, args.z_length]).to(local_rank)  # TODO: how to choose z?
            content_images = data["content"].to(local_rank)
            style_images = data["style"].to(local_rank)
            if args.start_epoch == 1 and epoch == args.start_epoch and i == 0:
                with torch.no_grad():
                  _ = netG(content_images, z)
                  continue

            fake_images = netG(content_images, z).detach()
            # Compute adversarial loss toward discriminator
            fake_score = netD(fake_images)
            real_score = netD(style_images)
            # TODO: how to compute d_loss?
            d_loss = - (real_score.mean() - fake_score.mean()) + gradient_penalty(style_images.data, fake_images.data, netD) * args.penalty_weight
            d_losses.append(d_loss.item())
            # Update discriminator
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            if (i + 1) % args.updateG_iter == 0:
                for param in netD.parameters():
                    param.requires_grad = False
                # Compute adversarial loss toward generator
                z2 = torch.randn([args.batch_size, args.z_length]).to(local_rank)  # TODO: how to choose z?
                fake_images2 = netG(content_images, z2)
                fake_score2 = netD(fake_images2)
                g_loss = -fake_score2.mean()
                g_losses.append(g_loss.item())
                # Update generator
                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()
                for param in netD.parameters():
                    param.requires_grad = True

        D_losses.append(np.mean(d_losses))
        G_losses.append(np.mean(g_losses))
        print("epoch: %d  time/epoch: %.2f  loss_G: %.3f  loss_D: %.3f" % (epoch, (time.time() - time_start), float(G_losses[-1]), float(D_losses[-1])))
        schedule_G.step()
        schedule_D.step()
        if epoch % args.save_epoch_freq == 0 and local_rank == 0:
            #  Save the results
            with torch.no_grad():
                generate_examples = netG(content_examples, z_examples)
                output_images = torch.cat((content_examples, generate_examples), dim=0)
                output_name = os.path.join(images_save_dir, "%04d.jpg" % (epoch))
                save_image(output_images, output_name, nrow=args.n_save_img)

            netD_state = {'epoch': epoch, 'state_dict': netD.module.state_dict(), 'optimizer': optim_D.state_dict()}
            netG_state = {'epoch': epoch, 'state_dict': netG.module.state_dict(), 'optimizer': optim_G.state_dict()}
            torch.save(netD_state, os.path.join(checkpoints_save_dir, "%04d_D.pth" % (epoch)))
            torch.save(netG_state, os.path.join(checkpoints_save_dir, "%04d_G.pth" % (epoch)))
            plt.figure(0)
            plt.plot(G_losses)
            plt.savefig(os.path.join(args.save_dir, 'G_loss.eps'))
            plt.figure(1)
            plt.plot(D_losses)
            plt.savefig(os.path.join(args.save_dir, 'D_loss.eps'))
            print("epoch: %d  Results have been saved." % (epoch))
