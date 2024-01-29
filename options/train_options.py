import argparse

train_parser = argparse.ArgumentParser()
# Basic options
train_parser.add_argument('--content_dir', type=str, required=True,
                          help='Directory path to a batch of content images')
train_parser.add_argument('--style_dir', type=str, required=True,
                          help='Directory path to a batch of style images')
train_parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
train_parser.add_argument('--save_dir', default='experiments',
                          help='Directory to save the model, the checkpoints and the intermediate results would be saved at save_dir/checkpoints and save_dir/images respectively')
train_parser.add_argument('--log_dir', default='./logs',
                          help='Directory to save the log')
train_parser.add_argument('--lr', type=float, default=1e-4)
train_parser.add_argument('--lr_decay', type=float, default=5e-5)
train_parser.add_argument('--max_epoch', type=int, default=300)
train_parser.add_argument('--batch_size', type=int, default=4)
train_parser.add_argument('--aligned', action='store_true', help='use aligned dataset instead of unaligned dataset')

train_parser.add_argument('--mse_weight', type=float, default=0)
train_parser.add_argument('--style_weight', type=float, default=1)
train_parser.add_argument('--content_weight', type=float, default=0.1)

# save options
train_parser.add_argument('--n_threads', type=int, default=8)
train_parser.add_argument('--save_epoch_freq', type=int, default=5,
                          help='frequency of saving checkpoints and images at the end of epochs')
train_parser.add_argument("--n_save_img", default=5, type=int, help="number of saved images")
train_parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
train_parser.add_argument('--resume', default='', type=str, metavar='PATH', help='file name of the latest checkpoint')

# glow parameters
train_parser.add_argument('--input_channel', default=3, type=int, help='input image channels')
train_parser.add_argument('--load_size', default=256, type=int, help='scale image to this size')
train_parser.add_argument('--img_size', default=128, type=int, help='final input image size')
train_parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
train_parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
train_parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
train_parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')