import argparse
import os
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

from custom_dataset.aligned_dataset import AlignedDataSet
from custom_dataset.unaligned_dataset import UnAlignedDataSet
import net

from math import log, sqrt, pi

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='experiments',
                    help='Directory to save the model, the checkpoints and the intermediate results would be saved at save_dir/checkpoints and save_dir/images respectively')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--aligned', action='store_true', help='use aligned dataset instead of unaligned dataset')

parser.add_argument('--mse_weight', type=float, default=0)
parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--content_weight', type=float, default=0.1)

# save options
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints and images at the end of epochs')
parser.add_argument("--n_save_img", default=5, type=int, help="number of saved images")
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='file name of the latest checkpoint')

# glow parameters
parser.add_argument('--input_channel', default=3, type=int, help='input image channels')
parser.add_argument('--load_size', default=256, type=int, help='scale image to this size')
parser.add_argument('--img_size', default=128, type=int, help='final input image size')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')

local_rank = int(os.environ["LOCAL_RANK"])
args = parser.parse_args()

if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
else:
    raise('Not implemented operator', args.operator)
    
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

# VGG
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
encoder = net.Net(vgg)
encoder = encoder.to(local_rank)

# glow
glow_single = Glow(args.input_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# l1 loss
mseloss = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(glow_single.parameters(), lr=args.lr)

# -----------------------resume training------------------------
if args.resume:
    assert os.path.isfile(args.resume), "--------no checkpoint found---------"
    print("--------loading checkpoint----------")
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cuda')
    args.start_epoch = checkpoint['epoch']
    glow_single.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

glow_single = glow_single.to(local_rank)
glow = nn.parallel.DistributedDataParallel(glow_single, device_ids=[local_rank], output_device=local_rank)
glow.train()

log_c = []
log_s = []
log_mse = []



# -------------------------------------------------------
if not (args.aligned):
    train_dataset = UnAlignedDataSet(args)
else:
    train_dataset = AlignedDataSet(args)
train_sampler = DistributedSampler(train_dataset)
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, sampler=train_sampler)
content_examples = []
style_examples = []
for i in range(args.n_save_img):
    content_examples.append(train_dataset[i]["content"].unsqueeze(0).to(local_rank))
    style_examples.append(train_dataset[i]["style"].unsqueeze(0).to(local_rank))
content_examples = torch.cat(content_examples, dim=0)
style_examples = torch.cat(style_examples, dim=0)

checkpoints_save_dir = os.path.join(args.save_dir, "checkpoints/")
if not os.path.exists(checkpoints_save_dir):
    os.mkdir(checkpoints_save_dir)
images_save_dir = os.path.join(args.save_dir, "images/")
if not os.path.exists(images_save_dir):
    os.mkdir(images_save_dir)

argsDict = args.__dict__
with open(os.path.join(args.save_dir, 'setting.txt'), 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')
# -----------------------training------------------------
Time = time.time()
for epoch in range(args.start_epoch, args.max_epoch):
    train_sampler.set_epoch(epoch)
    for i, data in enumerate(train_dataloader):
        content_images = data["content"].to(local_rank)
        style_images = data["style"].to(local_rank)
        if epoch == args.start_epoch and i == 0:
            with torch.no_grad():
              _ = glow.module(content_images, forward=True)
              continue

        # reverse
        stylized = glow(content_images, style=style_images, entire=True)

        loss_c, loss_s = encoder(content_images, style_images, stylized)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()
        loss_mse = mseloss(content_images, stylized)
        loss_style = args.content_weight * loss_c + args.style_weight * loss_s + args.mse_weight * loss_mse

        # optimizer update
        optimizer.zero_grad()
        loss_style.backward()
        nn.utils.clip_grad_norm(glow.module.parameters(), 5)
        optimizer.step()

        # update loss log
        log_c.append(loss_c.item())
        log_s.append(loss_s.item())
        log_mse.append(loss_mse.item())

    if (epoch + 1) % args.save_epoch_freq == 0 and local_rank == 0:
        with torch.no_grad():
            z_c_examples = glow(content_examples, forward=True)
            stylized = glow(z_c_examples, forward=False, style=style_examples)
            output_images = torch.cat((content_examples, style_examples, stylized), dim=0)
            output_name = os.path.join(images_save_dir, "%05d.jpg" % (epoch + 1))
            save_image(output_images, output_name, nrow=args.n_save_img)

        state_dict = glow.module.state_dict()
        state = {'epoch': epoch, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(checkpoints_save_dir, "%05d.pth" % (epoch + 1)))
        print("epoch %d   time/save_epoch_freq: %.2f   loss_c: %.3f   loss_s: %.3f   loss_mse: %.3f" % (epoch, (time.time() - Time) / args.save_epoch_freq, log_c[-1], log_s[-1], log_mse[-1]))
        log_c = []
        log_s = []
        Time = time.time()
