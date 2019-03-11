from retinanet.model import resnet18
from retinanet.losses import FocalLoss
import torch
import numpy as np


def test_training():
    fake_img_batch = np.random.uniform(0, 255, size=(2, 3, 512, 512)).astype(np.float32)
    fake_bb_batch = np.random.uniform(0, 400, (2, 4, 4))
    fake_classes_batch = np.random.uniform(0, 10, (2, 4, 1)).astype(np.uint8)
    fake_targets = np.concatenate((fake_bb_batch, fake_classes_batch), axis=-1).astype(np.float32)
    net = resnet18(num_classes=15)
    net.cuda()
    net.eval()

    (classification,  # for focal loss
     regression,  # for focal loss
     anchors,  # for focal loss
     nms_scores,  # for inference
     nms_class,  # for inference
     transformed_anchors  # for inference
     ) = net(torch.from_numpy(fake_img_batch).cuda())
    floss = FocalLoss()

    targets = torch.from_numpy(fake_targets).cuda()
    my_loss = floss(classification, regression, anchors, targets)
    pass
