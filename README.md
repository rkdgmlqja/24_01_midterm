# 24_01_midterm
## Param size comparison

LeNet : 61,706
CustomMLP: 72,842
Use torchsummary API or calculate by hand

LeNet = 6*5*5+16*5*5+400*120+120*84+84*10
CustomMLP = 8*3*3+16*3*3+32*3*3+32*4*4*128+128*10
