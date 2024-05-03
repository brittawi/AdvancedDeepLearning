import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
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
            *self.block(num_classes + int(np.prod(img_shape)), 1024, False, True),
            *self.block(1024, 512, True, True),
            *self.block(512, 256, True, True),
            *self.block(256, 128, False, False),
            *self.block(128, 1, False, False),
            nn.Sigmoid(),
        )
        
    def block(self, size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

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
        
        #self.validation_z = torch.randn(8, 100)
        
        #self.example_input_array = torch.zeros(2, 100)
        
    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        
        self.generator.train()
        self.discriminator.train()
        
        inputs, targets = batch
        optimizer_g, optimizer_d = self.optimizers()
        batch_size = inputs.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype)
        fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype)
        
        # Train the generator
        self.generator.zero_grad()
        noise = torch.randn([batch_size, self.hparams.latent_dim])
        conditional = torch.randint(0, 10, (batch_size,))
        x_fake = self(noise, conditional)
        y_fake_g = self.discriminator(x_fake, conditional)
        g_loss = self.adversarial_loss(y_fake_g, real_label)
        self.log("g_loss", g_loss, prog_bar=True)
        g_loss.backward()
        optimizer_g.step()
        
        # Train the discriminator
        self.discriminator.zero_grad()
        y_real = self.discriminator(inputs, targets)
        d_real_loss = self.adversarial_loss(y_real, real_label)
        y_fake_d = self.discriminator(x_fake.detach(), conditional)
        d_fake_loss = self.adversarial_loss(y_fake_d, fake_label)
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        d_loss.backward()
        optimizer_d.step()

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        # log sampled images
        with torch.no_grad():
            noise = torch.randn(8, 100)
            conditional = torch.randint(0, 10, (8,))
            sample_imgs = self(noise, conditional)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
if __name__ == '__main__':
    
    torch.manual_seed(1)
    
    Train = True
    if Train:
        dm = MNISTDataModule()
        model = LitVanillaGAN(*dm.dims)
        
        trainer = L.Trainer(
        accelerator="auto",
        max_epochs=15,
        fast_dev_run=False,
        )
        
        trainer.fit(model, dm)
        
        trainer.save_checkpoint("best_model_2.ckpt")
    
    # load model
    gan = LitVanillaGAN.load_from_checkpoint("best_model_2.ckpt")
    gan.eval()
    z = torch.randn(100, 100)
    print(z.shape)
    labels = torch.LongTensor([i for i in range(10) for _ in range(10)])
    images = gan(z, labels)
    grid = make_grid(images, nrow=10, normalize=True)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
    ax.axis('off')
    plt.show()
    
            
            
        
