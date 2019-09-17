import os
import time
import argparse
import torch
import torch.optim as optim

from dataload import Dataset
from torch.utils.data import DataLoader
from models import InpaintingModel


parser = argparse.ArgumentParser(description='Image Inpainting')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--max_iterations', type=int, default=500000, help="max iteration number in one epoch")
parser.add_argument('--batch_size', '-b', type=int, default=4, help="Batch size")
parser.add_argument('--patch_size', type=int, default=256, help="Patch size")

parser.add_argument('--TRAIN_FLIST', type=str, default='./flist/places2_train.flist')
parser.add_argument('--TRAIN_MASK_FLIST', type=str, default='./flist/masks.flist')
parser.add_argument('--TEST_FLIST', type=str, default='./flist/places2_val.flist')
parser.add_argument('--TEST_MASK_FLIST', type=str, default='./flist/masks.flist')

parser.add_argument('--save_model_dir', type=str, default='./save_models')
parser.add_argument('--save_iter_interval', type=int, default=100000, help="interval for saving model")

parser.add_argument('--LR', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('--D2G_LR', type=float, default=0.1, help='discriminator/generator learning rate ratio')
parser.add_argument('--REC_LOSS_WEIGHT', type=float, default=10.0, help='reconstruction loss weight')
parser.add_argument('--ADV_LOSS_WEIGHT', type=float, default=0.1, help='adversarial loss weight')
parser.add_argument('--STYLE_LOSS_WEIGHT', type=float, default=100.0, help='style loss weight')
parser.add_argument('--STEP_LOSS_WEIGHT', type=float, default=2.0, help='step loss weight')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gpu_id', type=str, help='GPU ID')
parser.add_argument('--RESUME', default=False, type=bool, help='load pre-trained weights')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--skip_validation', default=False, action='store_true')

config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id #这里的赋值必须是字符串，list会报错


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
else:
    DEVICE = torch.device("cpu")

inpaint_model = InpaintingModel(config).to(DEVICE)

if not config.skip_training:
    train_set = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
if not config.skip_validation:
    eval_set = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=True, training=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=1, shuffle=False)


def train():
    inpaint_model.discriminator.train()
    inpaint_model.generator.train()
    for epoch in range(config.epoch):
        for images, masks in train_loader:
            batch_t0 = time.time()
            images, masks = images.cuda(), masks.cuda()
            outputs, gen_loss, dis_loss = inpaint_model.process(images, masks)
            inpaint_model.backward(gen_loss, dis_loss)
            batch_t1 = time.time()
            if inpaint_model.iteration == 1 or inpaint_model.iteration % 20 == 0:
                psnr = inpaint_model.cal_psnr(outputs, images)
                print('[TRAIN] Epoch[{}({}/{})]; Loss_G:{:.6f}; Loss_D:{:.6f}; PSNR:{:.4f}; time:{:.4f} sec'.
                    format(epoch + 1, inpaint_model.iteration, len(train_loader), 
                    gen_loss.item(), dis_loss.item(), psnr.item(), batch_t1 - batch_t0))

        inpaint_model.save()


if config.RESUME:
    inpaint_model.load()

if not config.skip_training:
    train()

if not config.skip_validation:
    with torch.no_grad():
        eval()