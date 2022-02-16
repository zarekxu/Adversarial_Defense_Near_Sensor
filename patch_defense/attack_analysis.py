import torch 
import os

from patch_utils import*
from utils import*
from torchvision import models
from PIL import Image
from small_net import VGGBase
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# model = models.vgg16(pretrained=True).cuda()


#-------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGGBase()
print(model)
model = model.to(device)
#--------

# #--------
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = VGG('VGG3')
# model = model.to(device)
# if device == 'cuda':
#     # net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


# # Load checkpoint.
# print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/vgg3_tiny224.pth')
# model.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']
# #---------












print(model)
model.eval()


test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



classes = ('English springer', 'French horn', 'cassette player', 'chain saw', 'church',
           'garbage truck', 'gas pump', 'golf ball', 'parachute', 'tench')




def get_features_hook0(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    np.save("./attention/analysis/vgg3_freeze_featuremap3_benign_relu.npy",features)
    # np.save("./attention/analysis/vgg3_freeze_featuremap3_attack_relu.npy",features)







img = Image.open('./original_image.jpg')
# img = Image.open('./perturbated_iamge.jpg')
x = test_transforms(img)
x = x.unsqueeze(0).cuda()



# handle0 = model.features[10].register_forward_hook(get_features_hook0)
handle0 = model.pool3.register_forward_hook(get_features_hook0)

output = model(x)

handle0.remove()

pred = torch.argmax(output, 1)
print(pred)



print('Saving..')
state = {
    'net': model.state_dict(),
    'acc': 0,
    'epoch': 0,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/vgg3_freeze_relu_tiny.pth')