import argparse
import importlib

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


from har_dataset_train_test_split import train_test_split, HARDataset,dataset_exists
from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='../examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    # result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                   height, width, tensor.size(2))
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      1,-1, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    # result = result.transpose(2, 3).transpose(1, 2)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True
if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    model = importlib.import_module('PointNet_mlp_Vit_0037')
    model = model.PointNet_Vit( num_layers = 2, hidden_size = 2048, k = 7,channel = 5, normal_channel=False, model='rnn',seq_len = 20)
    model.apply(inplace_relu)
    checkpoint = torch.load('/home/zlc/yj/HAR/har_network/pointnet_mlp_transformer/log/classification/2022-02-22_00-37/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])


    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)
    data_path = '../../har_data'
    activity_list = ['walk']
    train_files_point, test_files_point = train_test_split(activity_list, 'point', data_path, 0.75)
    test_dataset = HARDataset(data_path, 'PAT', activity_list, concat_framNum=3,
                              seq_len=20,
                              file_list=test_files_point, pointLSTM=True, dataset_type='test',
                              pc_channel=5)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=False)

    for batch_id, (points, target, y) in enumerate(testDataLoader, 0):
        # sample = np.array(testDataLoader.dataset[0])
        # points_sample = torch.Tensor(sample[0])
        # target_sample = torch.Tensor(sample[1])
        print(batch_id)
        if batch_id == 100:
            points_sample = points
            target_sample = target
            break

    points_sample = points_sample.transpose(3, 2)
    targets = None
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=(points_sample,target_sample),
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)    # grayscale_cam.shape = (1, 1, 20)

    grayscale_cam = grayscale_cam[0, :]
    cv2.imwrite(f'{args.method}_cam_walk.jpg', grayscale_cam)

    # cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                 std=[0.5, 0.5, 0.5])
    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested category.
    # targets = None
    #
    # # AblationCAM and ScoreCAM have batched implementations.
    # # You can override the internal batch size for faster computation.
    # cam.batch_size = 32
    #
    # grayscale_cam = cam(input_tensor=input_tensor,
    #                     targets=targets ,
    #                     eigen_smooth=args.eigen_smooth,
    #                     aug_smooth=args.aug_smooth)
    #
    # # Here grayscale_cam has only one image in the batch
    # grayscale_cam = grayscale_cam[0, :]
    #
    # cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

