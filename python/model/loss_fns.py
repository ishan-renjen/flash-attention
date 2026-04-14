import torch
import torch.nn as nn
import torchvision

def discriminator_loss(real_data, fake_data, criterion):
    target_real_data = torch.ones_like(real_data)
    target_fake_data = torch.zeros_like(fake_data)

    r_loss = criterion(real_data, target_real_data)
    f_loss = criterion(fake_data, target_fake_data)
    loss_D = (r_loss + f_loss)/2
    return loss_D

def generator_loss(fake_data, criterion):
    target_real_data = torch.ones_like(fake_data)

    loss_G = criterion(fake_data, target_real_data)
    return loss_G

def cycleconsistency_loss(real_A, reconstructed_A, real_B, reconstructed_B, cycleconsistency_criterion, cycle_weight=10):
    #A->B->A
    cycle_a_loss = cycleconsistency_criterion(real_A, reconstructed_A)
    #B->A->B
    cycle_b_loss = cycleconsistency_criterion(real_B, reconstructed_B)
    return (cycle_a_loss + cycle_b_loss)*cycle_weight

class StructuralLossModel(nn.Module):
    def __init__(self, layers=('relu1_1', 'relu2_1', 'relu3_1')):
        super(StructuralLossModel, self).__init__()

        self.layer_name_mapping = {
            'relu1_1': 0,
            'relu2_1': 3,
            'relu3_1': 6,
            'relu4_1': 8,
            'relu5_1': 11}
        self.layers = layers

        vgg = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features.eval()

        max_idx = max([self.layer_name_mapping[layer] for layer in layers])
        self.vgg_slice = nn.Sequential(*[vgg[i] for i in range(max_idx+1)]).cuda()

        for param in vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for name, module in self.vgg_slice._modules.items():
            x = module(x)
            # Check if this module corresponds to one of the layers of interest
            for layer in self.layers:
                if int(name) == self.layer_name_mapping[layer]:
                    output[layer] = x
        return output

    def return_layers(self):
        return self.layers

def structural_loss(vgg, real_image, generated_image, weight=3):
    print(type(real_image))
    print(type(generated_image))
    #https://pytorch.org/hub/pytorch_vision_vgg/
    #https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
    #https://pytorch.org/tutorials/advanced/neural_style_tutorial.html?
    real_feature_map = vgg(real_image)
    generated_feature_map = vgg(generated_image)

    layers = vgg.return_layers()
    loss = 0
    for item in layers:
        loss += torch.nn.functional.l1_loss(generated_feature_map[item], real_feature_map[item])
    return loss*weight

def identity_loss(dark, identity_dark, light, identity_light, weight=3):
      identity_loss_A = torch.nn.functional.l1_loss(dark, identity_dark)
      identity_loss_B = torch.nn.functional.l1_loss(real_light_image, identity_light)

      return (identity_loss_A + identity_loss_B)*weight