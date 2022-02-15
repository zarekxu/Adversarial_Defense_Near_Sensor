# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation
# Created by Junbo Zhao 2020/3/19
import matplotlib.pyplot as plt
import numpy as np
import torch

# Initialize the patch
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)   
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        # x_location, y_location = 50, 50
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location


def get_features_hook1(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap1st.npy",features)

def get_features_hook2(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap2nd.npy",features)

def get_features_hook3(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap3rd.npy",features)

def get_features_hook4(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap4th.npy",features)

def get_features_hook5(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap5th.npy",features)


def get_features_hook11(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap1sta.npy",features)

def get_features_hook22(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap2nda.npy",features)

def get_features_hook33(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap3rda.npy",features)

def get_features_hook44(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap4tha.npy",features)

def get_features_hook55(self, input, output):
    
    features = output.data.cpu().numpy()
    features = np.squeeze(features)
    # print(features.shape)
    np.save("./attention/featuremap5tha.npy",features)



# # Test the patch on dataset
# def test_patch(patch_type, target, patch, test_loader, model):
#     model.eval()
#     test_total, test_actual_total, test_success = 0, 1, 0
#     for (image, label) in test_loader:
#         test_total += label.shape[0]
#         assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
#         image = image.cuda()
#         label = label.cuda()
#         handle1 = model.features[1].register_forward_hook(get_features_hook1)
#         handle2 = model.features[6].register_forward_hook(get_features_hook2)
#         handle3 = model.features[15].register_forward_hook(get_features_hook3)
#         handle4 = model.features[22].register_forward_hook(get_features_hook4)
#         handle5 = model.features[29].register_forward_hook(get_features_hook5)

#         output = model(image)
#         _, predicted = torch.max(output.data, 1)
#         if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
#             image1 = image.cpu().numpy()
#             image1  = np.squeeze(np.transpose(image1, (2, 3, 1, 0)))
#             image1 = np.clip(image1, 0, 1)
#             # height, width, channel = image.shape
#             # fig, ax = plt.subplots()

#             # fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
#             # plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
#             # plt.margins(0,0)
#             # plt.imshow(image)
#             plt.imshow(image1)
#             plt.savefig("original_image.jpg")
#             test_actual_total += 1
#             applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
#             applied_patch = torch.from_numpy(applied_patch)
#             mask = torch.from_numpy(mask)
#             perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
#             perturbated_image1  = np.squeeze(np.transpose(perturbated_image, (2, 3, 1, 0)))
#             perturbated_image1 = np.clip(perturbated_image1, 0, 1)

#             # # print(perturbated_image.shape)
#             # height, width, channel = perturbated_image.shape

#             # fig, ax = plt.subplots()

#             # fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
#             # plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
#             # plt.margins(0,0)
#             # # plt.savefig(path)

#             plt.imshow(perturbated_image1)
#             plt.savefig("perturbated_iamge.jpg")
#             perturbated_image = perturbated_image.cuda()
#             handle1 = model.features[1].register_forward_hook(get_features_hook11)
#             handle2 = model.features[6].register_forward_hook(get_features_hook22)
#             handle3 = model.features[15].register_forward_hook(get_features_hook33)
#             handle4 = model.features[22].register_forward_hook(get_features_hook44)
#             handle5 = model.features[29].register_forward_hook(get_features_hook55)
#             output = model(perturbated_image)
#             _, predicted = torch.max(output.data, 1)
#             if predicted[0].data.cpu().numpy() == target:
#                 test_success += 1
#     return test_success / test_actual_total


# # Test the patch on dataset
# def test_patch(patch_type, target, patch, test_loader, model):
#     model.eval()
#     test_total, test_actual_total, test_success = 0, 1, 0
#     for (image, label) in test_loader:
#         print("label is:", label)
#         test_total += label.shape[0]
#         assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
#         image = image.cuda()
#         label = label.cuda()
#         handle1 = model.features[1].register_forward_hook(get_features_hook1)
#         handle2 = model.features[6].register_forward_hook(get_features_hook2)
#         handle3 = model.features[15].register_forward_hook(get_features_hook3)
#         handle4 = model.features[22].register_forward_hook(get_features_hook4)
#         handle5 = model.features[29].register_forward_hook(get_features_hook5)

#         output = model(image)
#         handle1.remove()
#         handle2.remove()
#         handle3.remove()
#         handle4.remove()
#         handle5.remove()
#         _, predicted = torch.max(output.data, 1)
#         # if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
#         image1 = image.cpu().numpy()
#         image1  = np.squeeze(np.transpose(image1, (2, 3, 1, 0)))
#         image1 = np.clip(image1, 0, 255)
#         height, width, channel = image1.shape
#         fig, ax = plt.subplots()

#         fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
#         plt.margins(0,0)
#         plt.imshow(image1)
#         plt.savefig("original_image.jpg")
#         test_actual_total += 1
#         applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
#         applied_patch = torch.from_numpy(applied_patch)
#         mask = torch.from_numpy(mask)
#         perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
#         perturbated_image1  = np.squeeze(np.transpose(perturbated_image, (2, 3, 1, 0)))
#         perturbated_image1 = np.clip(perturbated_image1, 0, 255)
#         height, width, channel = perturbated_image1.shape

#         fig, ax = plt.subplots()

#         fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
#         plt.margins(0,0)
#         plt.imshow(perturbated_image1)
#         plt.savefig("perturbated_iamge.jpg")
#         perturbated_image = perturbated_image.cuda()
#         handle11 = model.features[1].register_forward_hook(get_features_hook11)
#         handle22 = model.features[6].register_forward_hook(get_features_hook22)
#         handle33 = model.features[15].register_forward_hook(get_features_hook33)
#         handle44 = model.features[22].register_forward_hook(get_features_hook44)
#         handle55 = model.features[29].register_forward_hook(get_features_hook55)



#         output = model(perturbated_image)
#         handle11.remove()
#         handle22.remove()
#         handle33.remove()
#         handle44.remove()
#         handle55.remove()
#         _, predicted = torch.max(output.data, 1)
#         if predicted[0].data.cpu().numpy() == target:
#             test_success += 1
#     return test_success / test_actual_total

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 1, 0
    k = 0
    for (image, label) in test_loader:
        print("label is:", label)
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        _, predicted = torch.max(output.data, 1)
        # if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
        image1 = image.cpu().numpy()
        image1  = np.squeeze(np.transpose(image1, (2, 3, 1, 0)))
        image1 = np.clip(image1, 0, 255)
        height, width, channel = image1.shape
        fig, ax = plt.subplots()

        fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.imshow(image1)
        plt.savefig("original_image.jpg")
        test_actual_total += 1
        applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image1  = np.squeeze(np.transpose(perturbated_image, (2, 3, 1, 0)))
        perturbated_image1 = np.clip(perturbated_image1, 0, 255)
        height, width, channel = perturbated_image1.shape

        fig, ax = plt.subplots()

        fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.imshow(perturbated_image1)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0].data.cpu().numpy() == target:
            test_success += 1
        k += 1
    return test_success / test_actual_total

def test_patch_ig(patch_type, target, patch, test_loader, model, x_location, y_location):
    model.eval()
    test_total, test_actual_total, test_success = 0, 1, 0
    k = 0
    for (image, label) in test_loader:
        print("label is:", label)
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        _, predicted = torch.max(output.data, 1)
        # if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
        image1 = image.cpu().numpy()
        image1  = np.squeeze(np.transpose(image1, (2, 3, 1, 0)))
        image1 = np.clip(image1, 0, 255)
        height, width, channel = image1.shape
        fig, ax = plt.subplots()

        fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.imshow(image1)
        plt.savefig("original_image.jpg")
        test_actual_total += 1
        applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image1  = np.squeeze(np.transpose(perturbated_image, (2, 3, 1, 0)))
        perturbated_image1 = np.clip(perturbated_image1, 0, 255)
        height, width, channel = perturbated_image1.shape

        fig, ax = plt.subplots()

        fig.set_size_inches(width/100.0,height/100.0)#输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.imshow(perturbated_image1)
        if test_loader == test_loader:
            plt.savefig("./perturbed_image/perturbated_image"+ str(k) + "_" + str(x_location) + "_" + str(y_location) + ".jpg")
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0].data.cpu().numpy() == target:
            test_success += 1
        k += 1
    return test_success / test_actual_total