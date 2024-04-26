import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from Datamodule import MNISTDataModule, BATCH_SIZE


class GeneratorForMNIST(nn.Module):
    def __init__(self, latent_dim, img_shape, num_classes: int = 10):
        super().__init__()
        self.img_shape = img_shape
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        conditional_inputs = torch.cat([z, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_inputs)
        out = out.view(out.size(0), *self.img_shape)
        return out
    
class DiscriminatorForMNIST(nn.Module):
    def __init__(self, img_shape, num_classes: int = 10):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([img_flat, conditional], dim=-1)
        validity = self.main(conditional_inputs)

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
        *args: torch.Any, 
        **kwargs: torch.Any) -> None:
        
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # networks
        data_shape = (channels, width, height)
        self.generator = GeneratorForMNIST(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = DiscriminatorForMNIST(img_shape=data_shape)
        
        self.validation_z = torch.randn(8, 100)
        
        #self.example_input_array = torch.zeros(2, 100)
        
    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        
        inputs, targets = batch
        
        optimizer_g, optimizer_d = self.optimizers()
        
        batch_size = inputs.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype)
        fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype)

        noise = torch.randn([batch_size, self.hparams.latent_dim])
        conditional = torch.randint(0, 10, (batch_size,))

        ##############################################
        # (1) Update D network: max E(x)[log(D(x))] + E(z)[log(1- D(z))]
        ##############################################
        # Set discriminator gradients to zero.
        optimizer_d.zero_grad()

        # Train with real.
        real_output = self.discriminator(inputs, targets)
        d_loss_real = self.adversarial_loss(real_output, real_label)
        d_loss_real.backward()

        # Train with fake.
        fake = self(noise, conditional)
        fake_output = self.discriminator(fake.detach(), conditional)
        d_loss_fake = self.adversarial_loss(fake_output, fake_label)
        d_loss_fake.backward()

        # Count all discriminator losses.
        d_loss = d_loss_real + d_loss_fake
        self.log("d_loss", d_loss, prog_bar=True)
        optimizer_d.step()

        ##############################################
        # (2) Update G network: min E(z)[log(1- D(z))]
        ##############################################
        # Set generator gradients to zero.
        optimizer_g.zero_grad()

        fake_output = self.discriminator(fake, conditional)
        g_loss = self.adversarial_loss(fake_output, real_label)
        self.log("g_loss", g_loss, prog_bar=True)
        g_loss.backward()
        optimizer_g.step()
        
        

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        # log sampled images
        z = self.validation_z.type_as(self.generator.main[0].weight)
        conditional = torch.randint(0, 10, (8,))
        sample_imgs = self(z, conditional)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
if __name__ == '__main__':
    
    Train = False
    if Train:
        dm = MNISTDataModule()
        model = LitVanillaGAN(*dm.dims)
        
        trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=128,
        fast_dev_run=False,
        )
        
        trainer.fit(model, dm)
        
        trainer.save_checkpoint("best_model.ckpt")
    
    # load model
    gan = LitVanillaGAN.load_from_checkpoint("best_model.ckpt")
    noise = torch.randn([1, 100])
    conditional = torch.tensor([4], dtype=torch.int64)

    # disable randomness, dropout, etc...
    gan.eval()
    
    with torch.no_grad():
        generated_image = gan(noise, conditional)
        generated_image = torch.squeeze(generated_image,1)
        plt.imshow(generated_image.permute(1,2,0), cmap='Greys')
        plt.show()
    
    
    
    
    
    
    
    # real_imgs, targets = batch
        
    #     optimizer_g, optimizer_d = self.optimizers()
        
    #     real_label = torch.ones(real_imgs.size(0), 1)
    #     real_label = real_label.type_as(real_imgs)
    #     fake_label = torch.zeros(real_imgs.size(0), 1)
    #     fake_label = fake_label.type_as(real_imgs)
            
    #     # sample noise
    #     z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
    #     z = z.type_as(real_imgs)
        
    #     conditional = torch.randint(0, 10, (real_imgs.size(0),), dtype=targets.dtype)
    #     conditional = conditional.type_as(targets)
        
    #     # update discriminator
    #     self.toggle_optimizer(optimizer_d)
    #     optimizer_d.zero_grad()
    #     real_output = self.discriminator(real_imgs, targets)
    #     d_loss_real = self.adversarial_loss(real_output, real_label)
    #     d_loss_real.backward()
        
    #     # Train with fake.
    #     fake = self.generator(z, conditional)
    #     fake_output = self.discriminator(fake.detach(), conditional)
    #     d_loss_fake = self.adversarial_loss(fake_output, fake_label)
    #     d_loss_fake.backward()
        
    #     # log sampled images
    #     sample_imgs = fake[:6]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("train_generated_images", grid, 0)

    #     # Count all discriminator losses.
    #     d_loss = d_loss_real + d_loss_fake
    #     self.log("d_loss", d_loss, prog_bar=True)
    #     optimizer_d.step()
    #     self.untoggle_optimizer(optimizer_d)
        
    #     # sample noise
    #     z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
    #     z = z.type_as(real_imgs)
        
    #     conditional = torch.randint(0, 10, (real_imgs.size(0),), dtype=targets.dtype)
    #     conditional = conditional.type_as(targets)
        
    #     real_label = torch.ones(real_imgs.size(0), 1)
    #     real_label = real_label.type_as(real_imgs)
        
    #     # Set generator gradients to zero.
    #     self.toggle_optimizer(optimizer_g)
    #     optimizer_g.zero_grad()

    #     fake = self.generator(z, conditional)
    #     fake_output = self.discriminator(fake.detach(), conditional)
    #     g_loss = self.adversarial_loss(fake_output, real_label)
    #     self.log("g_loss", g_loss, prog_bar=True)
    #     self.manual_backward(g_loss)
    #     optimizer_g.step()
    #     self.untoggle_optimizer(optimizer_g)