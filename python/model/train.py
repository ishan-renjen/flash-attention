import torch
import os
import matplotlib.pyplot as plt
import torchvision

def train_eval_loop(dark_loader, light_loader, 
                    D_A, D_B, G_A, G_B, vgg,
                    opt_DA, opt_DB, opt_GA, opt_GB,
                    generator_loss, discriminator_loss, identity_loss,
                    structural_loss, cycleconsistency_loss,
                    n_epoch=10, saveItem = False):

    lowest_gen_loss = float('inf')
    os.makedirs("../../out/models/", exist_ok=True)
    os.makedirs("../../out/images/", exist_ok=True)
    index = 0

    for e, epoch in enumerate(range(n_epoch)):
        for dark in dark_loader:
            dark = dark.cuda()
            disc_iterator = iter(light_loader)
            real_light_image = next(disc_iterator).cuda()
            #train discriminators 1 and 2
            generated_light_image = G_A(dark).cuda()
            disc_real_output = D_A(real_light_image)
            disc_fake_output = D_A(generated_light_image)

            disc_darktolight_loss = discriminator_loss(disc_real_output, disc_fake_output)
            #***************************************************************************************************
            generated_dark_image = G_B(real_light_image)
            disc_real_output = D_B(dark)
            disc_fake_output = D_B(generated_dark_image)

            disc_lighttodark_loss = discriminator_loss(disc_real_output, disc_fake_output)
            #***************************************************************************************************
            opt_DA.zero_grad()
            disc_darktolight_loss.backward()
            opt_DA.step()

            opt_DB.zero_grad()
            disc_lighttodark_loss.backward()
            opt_DB.step()
            #***************************************************************************************************
            #train generators 1 and 2
            generated_light_image = G_A(dark).cuda()
            generated_dark_image = G_B(real_light_image)

            discriminate_light = D_A(generated_light_image)
            discriminate_dark = D_B(generated_dark_image)

            gen_light_loss = generator_loss(discriminate_light)
            gen_dark_loss = generator_loss(discriminate_dark)

            recon_dark  = G_B(generated_light_image)     # dark -> light -> dark
            recon_light = G_A(generated_dark_image)      # light -> dark -> light

            identity_light = G_A(real_light_image)
            identity_dark = G_B(dark)

            id_loss = identity_loss(dark, identity_dark, real_light_image, identity_light)

            cycle_loss  = cycleconsistency_loss(dark, recon_dark, real_light_image, recon_light)
            struct_loss = structural_loss(vgg, dark, generated_light_image)
            #***************************************************************************************************
            gen_total = gen_dark_loss + gen_light_loss + cycle_loss + struct_loss + id_loss

            opt_GA.zero_grad()
            opt_GB.zero_grad()
            gen_total.backward()
            opt_GA.step()
            opt_GB.step()
            #***************************************************************************************************
            gen_loss = gen_total.item()
            if gen_loss < lowest_gen_loss:
                lowest_gen_loss = gen_loss
                saveItem = True

                torch.save({
                    'epoch': epoch,
                    'GA_state_dict': G_A.state_dict(),
                    'GA_optimizer_state_dict': opt_GA.state_dict(),
                    'GB_state_dict': G_B.state_dict(),
                    'GB_optimizer_state_dict': opt_GB.state_dict(),
                    'DA_state_dict': D_A.state_dict(),
                    'DA_optimizer_state_dict': opt_DA.state_dict(),
                    'DB_state_dict': D_B.state_dict(),
                    'DB_optimizer_state_dict': opt_DB.state_dict(),
                    'loss': gen_loss}, os.path.join("../../out/models/", "best_cycleGAN.pth"))
            else:
                saveItem = False
            #***************************************************************************************************
            G_A.eval()

            input  = dark.cpu()
            output = generated_light_image.cpu()
            comparison = torch.cat([input, output], dim=0)
            B = input.size(0)
            grid_img = torchvision.utils.make_grid(comparison, nrow=B, normalize=True, scale_each=True)

            if saveItem == True:
                filename = os.path.join("../../out/images/", f'iteration_{index}_epoch{epoch}.jpg')
                print(f' | Save some samples to {filename}.')

            torchvision.utils.save_image(grid_img, filename, nrow=B)
            plt.figure(figsize=(12,6))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()
            G_A.train()

            index = index+1