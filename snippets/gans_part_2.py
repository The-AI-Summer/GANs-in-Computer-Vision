 
 import torch 
 import itertools

class CycleGAN():
        init(lr_g, lr_d, beta1, fake_pool_X, fake_pool_Y, criterionGAN, criterionCycle):
        """
        Define any G image-to-image model and any D image-to-scalar in [0,1]
        """
                self.netG_X = generator()
                self.netG_Y = generator() 

                self.netD_X = discriminator()
                self.netD_Y = discriminator()

                # Unify generator optimizers 
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_X.parameters(),
                        self.netG_Y.parameters()), lr=lr_g, betas=(beta1, 0.999))

                # Unify discriminator optimizers 
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                         self.netD_B.parameters()), lr=lr_d, betas=(beta1, 0.999))

                self.hyperparameters = (lambda_X,lambda_Y)

                # Abstractly defined Image-pooling and criteria
                # for code-compactness and understanding
                self.fake_pool_X = fake_pool_X
                self.fake_pool_Y = fake_pool_Y
                self.criterionGAN = criterionGAN
                self.criterionCycle = criterionCycle


        def training_cyclye_gan_scheme(real_X_data, real_Y_data):
                """
                real_X_data is a bunch(mini-set) of images from domain X. 
                real_Y_data is a bunch(mini-set) of images from domain Y 
                shape is [batch, (mini-set), channels=3, height, width] 
                """

                # we make input data available as class member to be in the scope of all classes
                self.real_X_data = real_X_data
                self.real_Y_data = real_Y_data

                # 1st cycle starting from Domain X
                self.fake_Y = self.netG_X(real_X_data)       # G_X(x) --> in Y domain
                self.reconstructed_X = self.netG_B(fake_Y)   # G_Y(G_X(x)) --> in X domain, ideally  G_Y(G_X(x))=x 

                # 2nd cycle starting from Domain Y
                self.fake_X = self.netG_Y(real_Y_data)       # G_Y(y) --> in X domain
                self.reconstructed_Y = self.netG_X(fake_X)   # G_X(G_Y(y)) --> in Y domain, ideally G_X(G_Y(y))=y

                # Train netG_X and netG_B
                # To do that, we need to disable gradients for discriminators
                # because discriminators require no gradients when optimizing Generators 
                self.set_requires_grad([self.netD_X, self.netD_Y], requires_grad=False)
                
                self.optimizer_G.zero_grad()
                # calculate losses and gradients for G_A and G_B and update
                self.backward_G()             
                self.optimizer_G.step() 

                # Train netD_A and netD_B
                self.set_requires_grad([self.netD_A, self.netD_B], requires_grad=True)
                self.optimizer_D.zero_grad()   
                self.backward_D()
                self.optimizer_D.step() 
     
        
        def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for net in nets:
                for param in net.parameters():
                        param.requires_grad = requires_grad

        def backward_G(self):
        """
        Calculate the loss for generators G_A and G_B
        Identity loss is skipped for educational purposes!
        """
        lambda_X, lambda_Y = self.hyperparameters
     
        # GAN loss D_X(G_X(x))
        loss_G_X = self.criterionGAN(self.netD_X(self.fake_Y_data))

        # GAN loss D_Y(G_Y(y))
        loss_G_Y = self.criterionGAN(self.netD_Y(self.fake_X_data))

        # Forward cycle loss || G_Y(G_X(x)) - x||
        loss_cycle_X = self.criterionCycle(self.reconstructed_X, self.real_X_data) * lambda_X

        # Backward cycle loss || G_X(G_Y(y)) - y||
        loss_cycle_Y = self.criterionCycle(self.reconstructed_Y, self.real_Y_data) * lambda_Y

        # combined loss and calculate gradients
        loss_G = loss_G_X + loss_G_Y + loss_cycle_X + loss_cycle_Y
        loss_G.backward()

        def backward_D(self):
        """Calculate adverserial loss for the discriminator"""
        pred_real = netD_X(self.real_X_data)
        loss_D_real = self.criterionGAN(pred_real, target=True)
        # Fake
        pred_fake = netD_X(self.fake_X_data.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target=False)
        # Combined adverserial losses and calculate gradients
        loss_D_X = (loss_D_real + loss_D_fake) * 0.5
        loss_D_X.backward()

        # the same for the other model
        pred_real = netD_Y(self.real_Y_data)
        loss_D_real = self.criterionGAN(pred_real, target=True)
        pred_fake = netD_Y(self.fake_Y_data.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target=False)
        # Combined adverserial loss and calculate gradients
        loss_D_Y = (loss_D_real + loss_D_fake) * 0.5
        loss_D_Y.backward()
        return loss_D