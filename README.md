The project is based on the paper 
'Wang, L., Guo, Y., Lin, Z., Deng, X. and An, W., 2018, December. Learning for video super-resolution through HR optical flow estimation. 
In Asian Conference on Computer Vision (pp. 514-529). Springer, Cham.' Only the loss function is changed from MSE to SSIM

train_drive.py is the main program used for training.

modules.py sets up the network architecture

data_utils.py contains preprocessing codes for preprocessing imageser_resolution

The sr_02_SSIM.png are result of SSIM as loss function. The sr_02_mse.png is the original result
