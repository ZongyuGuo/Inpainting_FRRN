import os
import torch
import torch.nn as nn
import torch.optim as optim
from networks import FRRNet, Discriminator
from loss import AdversarialLoss, StyleLoss



class InpaintingModel(nn.Module):
    def __init__(self, config):
        super(InpaintingModel, self).__init__()
        self.name = 'InpaintingModel'
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.save_model_dir, 'InpaintingModel_gen.pth')
        self.dis_weights_path = os.path.join(config.save_model_dir, 'InpaintingModel_dis.pth')

        self.generator = FRRNet()
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=True)

        if torch.cuda.device_count() > 1:
            device_ids=range(torch.cuda.device_count())
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.discriminator = nn.DataParallel(self.discriminator , device_ids)

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss()  

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(config.LR),
            betas=(0.0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(0.0, 0.9)
        )

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs, mid_x, mid_mask = self(images, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.REC_LOSS_WEIGHT
        gen_loss += gen_l1_loss

        # l1 hole loss is equivalent to l1 loss, rec loss weight = l1 loss weight + l1 hole loss weight
        gen_l1_loss_hole = self.l1_loss(outputs * masks, images * masks) * self.config.REC_LOSS_WEIGHT
        gen_loss += gen_l1_loss_hole

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # generator step loss
        for idx in range(len(mid_x) - 1):
            mid_l1_loss = self.l1_loss(mid_x[idx] * mid_mask[idx], images * mid_mask[idx]) 
            gen_loss += mid_l1_loss * self.config.STEP_LOSS_WEIGHT

        # create logs
        '''
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]
        '''

        return outputs, gen_loss, dis_loss

    def forward(self, images, masks):
        image_with_mask = images * masks
        outputs, mid_x, mid_mask = self.generator(image_with_mask, masks)  
        return outputs, mid_x, mid_mask

    def backward(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()
        gen_loss.backward()
        self.gen_optimizer.step()

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.RESUME == True and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        if not os.path.exists(self.config.save_model_dir):
            os.makedirs(self.config.save_model_dir)
            print('File folder created')
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

    def cal_psnr(self, images1, images2):
        mse_loss = self.l2_loss(images1, images2)
        psnr = 10 * torch.log10(1 / mse_loss)
        return psnr