import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, 14),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)


        x = self.transition(x)
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)#14
        return x
