import argparse

train_parser = argparse.ArgumentParser()
# Basic options
train_parser.add_argument('--content_dir', type=str, required=True,
                          help='Directory path to a batch of content images')
train_parser.add_argument('--style_dir', type=str, required=True,
                          help='Directory path to a batch of style images')

# training options
train_parser.add_argument('--n_threads', type=int, default=8)
train_parser.add_argument('--lr_G', type=float, default=1e-4)
train_parser.add_argument('--lr_D', type=float, default=1e-4)
train_parser.add_argument('beta1_G', type=float, default=0.5)
train_parser.add_argument('beta2_G', type=float, default=0.999)
train_parser.add_argument('beta1_D', type=float, default=0.5)
train_parser.add_argument('beta2_D', type=float, default=0.999)
train_parser.add_argument('--max_epoch', type=int, default=300)
train_parser.add_argument('--batch_size', type=int, default=4)
train_parser.add_argument('--aligned', action='store_true', help='use aligned dataset instead of unaligned dataset')
train_parser.add_argument('--penalty_weight', type=float, default=10.0, help='weight of gradient penalty term in d_loss')
train_parser.add_argument('--updateG_iter', type=int, default=2, help='how many iterations does D need to be updated before G is updated ')


# save options
train_parser.add_argument('--save_dir', default='experiments',
                          help='Directory to save the model, the checkpoints and the intermediate results would be saved at save_dir/checkpoints and save_dir/images respectively')
train_parser.add_argument('--save_epoch_freq', type=int, default=5,
                          help='frequency of saving checkpoints and images at the end of epochs')
train_parser.add_argument("--n_save_img", default=5, type=int, help="number of saved images")
train_parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
train_parser.add_argument('--resume', default='', type=str, metavar='PATH', help='file name of the latest checkpoint')

# model configs
train_parser.add_argument('--input_channel', default=3, type=int, help='input image channels')
train_parser.add_argument('--load_size', default=256, type=int, help='scale image to this size')
train_parser.add_argument('--img_size', default=128, type=int, help='final input image size')
train_parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
train_parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
train_parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
train_parser.add_argument('--affine', action='store_true', help='use affine coupling instead of additive')
train_parser.add_argument('--z_length', default=512, type=int, help='length of the input latent code z')
train_parser.add_argument('--w_length', default=512, type=int, help='length of the intermediate latent code w')
train_parser.add_argument('--dc_max', default=512, type=int, help='max channel number of discriminator')
train_parser.add_argument('--filter', default=None, type=list, help='low-pass filter applied in discriminator (none by default)')