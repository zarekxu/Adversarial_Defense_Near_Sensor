'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# cfg = {
#     'VGG3': [16, 'M', 16, 'M', 32, 'M'],
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(32, 200)

#     def forward(self, x):
#         out = self.features(x)
#         out = F.avg_pool2d(out, 28)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)




class VGG(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGG, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(256, 1000)

        # Load pretrained layers
        self.load_pretrained_layers()
        # self._initialize_weights()

    def forward(self, image):


        out = F.relu(self.conv1(image))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True
        out = F.avg_pool2d(out, 28)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out 




    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        # print("model structure is:",state_dict)
        param_names = list(state_dict.keys())
        print("model structure is:", param_names)

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        print("pre-trained model structure is:", pretrained_param_names)

        state_dict['conv1.weight'] = pretrained_state_dict[pretrained_param_names[0]]
        state_dict['conv1.bias'] = pretrained_state_dict[pretrained_param_names[1]]
        state_dict['conv2.weight'] = pretrained_state_dict[pretrained_param_names[4]]
        state_dict['conv2.bias'] = pretrained_state_dict[pretrained_param_names[5]]
        state_dict['conv3.weight'] = pretrained_state_dict[pretrained_param_names[8]]
        state_dict['conv3.bias'] = pretrained_state_dict[pretrained_param_names[9]]

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")




def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
