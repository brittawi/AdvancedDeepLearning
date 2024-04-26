import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F
import torchvision
from Datamodule import MNISTDataModule, BATCH_SIZE


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    
    
class LitVanillaGAN(L.LightningModule):
    def __init__(
        self, 
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        with_BCE = True,
        *args: torch.Any, 
        **kwargs: torch.Any) -> None:
        
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.with_BCE = with_BCE
        
        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)
        
        self.validation_z = torch.randn(8, 100)
        
        self.example_input_array = torch.zeros(2, 100)
    
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        real_imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()
            
        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        # train generator
        # generate images
        g_sample = self(z)
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(g_sample)

        # log sampled images
        sample_imgs = g_sample[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("train_generated_images", grid, 0)

        # discriminator loss and backprop
        self.toggle_optimizer(optimizer_d)
        if self.with_BCE:
            valid = torch.ones(real_imgs.size(0), 1)
            valid = valid.type_as(real_imgs)
            d_loss_real = self.adversarial_loss(d_real, valid)
            
            fake = torch.zeros(real_imgs.size(0), 1)
            fake = fake.type_as(real_imgs)
            d_loss_fake = self.adversarial_loss(d_fake, fake)
            
            d_loss = d_loss_real + d_loss_fake
        else:
            d_loss = -(torch.mean(torch.log(d_real) + torch.log(1. - d_fake)))
            
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        # train generator
        # generate images
        g_sample = self(z)
        d_fake = self.discriminator(g_sample)

        # generator loss and backprop
        self.toggle_optimizer(optimizer_g)
        if self.with_BCE:
            valid = torch.ones(real_imgs.size(0), 1)
            valid = valid.type_as(real_imgs)
            g_loss = self.adversarial_loss(d_fake, valid)
        else:
            g_loss = -(torch.mean(torch.log(d_fake)))
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        # log sampled images
        print("on epoch end")
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
if __name__ == '__main__':
    
    dm = MNISTDataModule()
    model = LitVanillaGAN(*dm.dims, with_BCE=False)
    
    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=20,
    fast_dev_run=False,
    )
    
    trainer.fit(model, dm)