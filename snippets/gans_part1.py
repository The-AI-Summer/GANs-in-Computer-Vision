import torch


def ones_target(size):
    '''
    For real data when training D, while for fake data when training G
    Tensor containing ones, with shape = size
    '''
    return torch.ones(size, 1)


def zeros_target(size):
    '''
    For data when training D
    Tensor containing zeros, with shape = size
    '''
    return torch.zeros(size, 1)


def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    target_real = ones_target(N)

    error_real = loss(prediction_real, target_real)
    error_real.backward()

    # Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    target_fake = zeros_target(N)
    
    error_fake = loss(prediction_fake, target_fake)
    error_fake.backward()

    # Update weights with gradients
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, optimizer, fake_data, loss):
    N = fake_data.size(0)  
    # Reset the gradients
    optimizer.zero_grad() 
    # Sample noise and generate fake data
    prediction = discriminator(fake_data) 

    # Calculate error and backpropagate
    target = ones_target(N)
    error = loss(prediction, target)
    error.backward()  
    optimizer.step()
    return error
