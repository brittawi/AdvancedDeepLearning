import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F
import torchvision
from Datamodule import MNISTDataModule


class SimpleGenerator(nn.Module):
    
    def __init__(self,input_size, hidden_size, output_size,img_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_shape = img_shape
        self.generator_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        img = self.generator_net(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
        
        
class SimpleDiscriminator(nn.Module):
    
    def __init__(self,input_size, hidden_size, output_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.discriminator_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.discriminator_net(img_flat)

        return validity
    
    
class LitVanillaGAN(L.LightningModule):
    def __init__(
        self, 
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        *args: torch.Any, 
        **kwargs: torch.Any) -> None:
        
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # networks
        self.generator = SimpleGenerator(input_size=100, hidden_size=128, output_size=784, img_shape=(1,28,28))
        self.discriminator = SimpleDiscriminator(input_size=784, hidden_size=128, output_size=1)
        
        self.validation_z = torch.randn(8, 100)
        
        self.example_input_array = torch.zeros(2, 100)
        
    # def sample_Z(m, n):
    #     '''Uniform prior for G(Z)'''
    #     return np.random.uniform(-1., 1., size=[m, n])
    
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        imgs, _ = batch
        print("image dims: ", imgs.shape)

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], 100)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        print("valid: ", valid.shape)

        # adversarial loss is binary cross-entropy
        #g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        print("valid: ", valid.shape)
        
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
if __name__ == '__main__':
    
    dm = MNISTDataModule()
    model = LitVanillaGAN()
    
    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=100,
    fast_dev_run=False,
    )
    
    trainer.fit(model, dm)