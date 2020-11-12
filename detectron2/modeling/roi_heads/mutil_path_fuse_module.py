import torch
from torch import nn
from detectron2.structures import Boxes

def get_selfarea_and_interarea(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute self_area and inter_area
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & N boxes, respectively.
    Returns:
        self_area: proposal_boxes_area, sized [N]
        inter_area: inter_area, sized [N,N].
    """
    self_area = boxes1.area()
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  

    wh = (rb - lt).clamp(min=0)  
    inter_area = wh[:, :, 0] * wh[:, :, 1]  

    return self_area, inter_area


class Mutil_Path_Fuse_Module(nn.Module):
    """
    Mutil path fuse module. given features of word-level texts, character-level texts, 
    global context, and get the richer fused features.

    Args:
        x (Tensor): mask roi features of word-level and character-level text instances, sized [N,256,14,14]
        global context (Tensor): seg roi features of global context, sized [N,256,14,14]
        proposals (list[Instances]): selected positive proposals

    Returns:
        feature_fuse (Tensor): richer fused features
    """
    def __init__(self, cfg):
        super(Mutil_Path_Fuse_Module, self).__init__()

        self.channels = cfg.MODEL.TEXTFUSENET_SEG_HEAD.CHANNELS

        self.char_conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.char_conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1,
                                      padding=0, bias=False)

        self.text_conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.text_conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1,
                                      padding=0, bias=False)

        self.conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x, global_context, proposals):
        result = []

        if self.training:
            proposal_boxes = proposals[0].proposal_boxes
            classes = proposals[0].gt_classes
        else:
            proposal_boxes = proposals[0].pred_boxes
            classes = proposals[0].pred_classes

        if len(proposal_boxes) == 0:
              return x
              
        self_area, inter_area = get_selfarea_and_interarea(proposal_boxes, proposal_boxes)
        self_area = self_area.reshape(1, self_area.shape[0])
        self_area = self_area.repeat(len(proposal_boxes), 1)
        inter_percent = inter_area / self_area
        char_pos = inter_percent > 0.9

        for i in range(len(proposal_boxes)):
            if classes[i] != 0:
                char = x[i]
                char = char.reshape([1, char.shape[0], char.shape[1], char.shape[2]])
                char = self.char_conv3x3(char)
                char = self.char_conv1x1(char)
                result.append(char)
            else:
                if torch.sum(char_pos[i]) > 1:
                    text = x[char_pos[i]]
                    text = torch.sum(text,dim=0)/(text.shape[0])
                    text = text.reshape([1, text.shape[0], text.shape[1], text.shape[2]])
                    text = self.text_conv3x3(text)
                    text = self.text_conv1x1(text)
                    result.append(text)
                else:
                    text = x[i]
                    text = text.reshape([1, text.shape[0], text.shape[1], text.shape[2]])
                    text = self.text_conv3x3(text)
                    text = self.text_conv1x1(text)
                    result.append(text)

        char_context = torch.stack(result)
        char_context = char_context.squeeze(1)

        feature_fuse = char_context + x + global_context

        feature_fuse = self.conv3x3(feature_fuse)
        feature_fuse = self.conv1x1(feature_fuse)
        feature_fuse = self.bn(feature_fuse)
        feature_fuse = self.relu(feature_fuse)

        return feature_fuse


def build_mutil_path_fuse_module(cfg):
    return Mutil_Path_Fuse_Module(cfg)


