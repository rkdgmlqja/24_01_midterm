# 24_01_midterm
## Param size comparison

LeNet : 61,706
CustomMLP: 72,842
Use torchsummary API or calculate by hand

LeNet = 6 * 5 * 5 + 16 * 5 * 5 + 400 * 120 + 120 * 84 + 84 * 10

CustomMLP = 8 * 3 * 3 + 16 * 3 * 3 + 32 * 3 * 3 + 32 * 4 * 4 * 128 + 128 * 10

LeNet Train Loss: 0.0673, LeNet Train Accuracy: 97.94%
LeNet Test Loss: 0.0679, LeNet Test Accuracy: 97.82%

CustomMLP Train Loss: 0.0657, Train Accuracy: 97.94%
CustomMLP Test Loss: 0.0657, Test Accuracy: 97.92%

![Figure_1](https://github.com/rkdgmlqja/24_01_midterm/assets/33273567/e733f4fe-35b7-466f-b774-117f55743a27)

trained LeNet-5 accuracy is almost the same with Reference LeNet-5 accuracy(98%)
