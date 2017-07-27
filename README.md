# alpha-GAN-Toy

Toy MNIST code for alpha-GAN

Small difference from original paper : 
Update generator and encoder with same loss, at the same time.
Since their loss is not exclusive and shares a L1-norm term, it is much better to integrate loss of two nets. 
