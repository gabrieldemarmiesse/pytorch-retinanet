from torch.utils.data import DataLoader
from torchvision import transforms
import losses
import model
from dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer


def test_training():
    dataset_train = CocoDataset('/datasets_master/COCO', set_name='train2017',
                                transform=transforms.Compose(
                                    [Normalizer(), Augmenter(), Resizer()]))
    retinanet = model.resnet18(num_classes=dataset_train.num_classes(),
                               pretrained=True)
    retinanet.cuda()
    retinanet.eval()
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater,
                                  batch_sampler=sampler)

    for data in dataloader_train:
        (classification,  # for focal loss
         regression,      # for focal loss
         anchors,         # for focal loss
         nms_scores,      # for inference
         nms_class,       # for inference
         transformed_anchors) = retinanet(data['img'].cuda())
        break
    floss = losses.FocalLoss()

    my_loss = floss(classification, regression, anchors, data['annot'].cuda())
    pass
