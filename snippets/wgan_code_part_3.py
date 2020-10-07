import torch
from torch.autograd import Variable
from torch import autograd


def WGAN_train_step(optimizer_D, optimizer_G, generator, discriminator, real_imgs, clip_value, iteration, n_critic = 5):
  batch = real_imgs.size(0)
  #  Train Discriminator
  optimizer_D.zero_grad()

  # Sample noise for dim=100
  z = torch.rand(batch, 100)

  # Generate a batch of images
  fake_imgs = generator(z).detach()
  # Adversarial loss
  loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
  loss_D.backward()
  optimizer_D.step()

  # Clip weights of discriminator
  for p in discriminator.parameters():
    p.data.clamp_(-clip_value, clip_value)

  # Train the generator every couple of iterations
  if iteration % n_critic == 0:
    optimizer_G.zero_grad()

    # Generate a batch of images
    z = torch.rand(batch, 100)
    gen_imgs = generator(z)
    # Adversarial loss
    loss_G = -torch.mean(discriminator(gen_imgs))

    loss_G.backward()
    optimizer_G.step()



def WGAN_GP_train_step(optimizer_D, optimizer_G, generator, discriminator, real_imgs, iteration, n_critic = 5):
  """
  Keep in mind that for Adam optimizer the official paper sets β1 = 0, β2 = 0.9
  betas = ( 0, 0.9)
  """
  optimizer_D.zero_grad()
  # Sample noise for dim=100
  batch = real_imgs.size(0)
  z = torch.rand(batch, 100)
  # Generate a batch of fake images
  fake_imgs = generator(z).detach()

  # Find penalty between real and fake images
  grad_penalty = gradient_penalty(discriminator, real_imgs, fake_imgs)

  # Adversarial loss + penalty
  loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + grad_penalty

  loss_D.backward()
  optimizer_D.step()
  # Train the generator every couple of iterations
  if iteration % n_critic == 0:
    optimizer_G.zero_grad()

    # Generate a batch of images
    z = torch.rand(batch, 100)
    gen_imgs = generator(z)
    # Adversarial loss
    loss_G = -torch.mean(discriminator(gen_imgs))
    loss_G.backward()
    optimizer_G.step()


def gradient_penalty(discriminator, real_imgs, fake_imgs, gamma=10):
  batch_size = real_imgs.size(0)
  epsilon = torch.rand(batch_size, 1, 1, 1)
  epsilon = epsilon.expand_as(real_imgs)

  interpolation = epsilon * real_imgs.data + (1 - epsilon) * fake_imgs.data
  interpolation = Variable(interpolation, requires_grad=True)


  interpolation_logits = discriminator(interpolation)
  grad_outputs = torch.ones(interpolation_logits.size())

  gradients = autograd.grad(outputs=interpolation_logits,
                            inputs=interpolation,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True)[0]

  gradients = gradients.view(batch_size, -1)
  gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
  return torch.mean(gamma * ((gradients_norm - 1) ** 2))

