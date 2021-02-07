import torch
import torch.nn as nn
# from operations import *
from torch.autograd import Variable
# from utils import drop_path
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
DARTS_image = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
PC_DARTS_sctd = Genotype(normal=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
# from mmdet.apis import get_root_logger
# from mmdet.apis import get_root_logger

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        output=torch.cat([states[i] for i in self._concat], dim=1)
        # print(output.shape)
        return output


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux



@BACKBONES.register_module()
class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(NetworkImageNet, self).__init__()
        genotype = DARTS_image
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        # outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            # if i == 1 * self._layers // 3:
                # outputs.append(s1)
                # print("--------------1:::::",s1.shape)
            # if i == 2 * self._layers // 3:
            #     outputs.append(s1)
            #     print("--------------2:::::", s1.shape)
        outputs.append(s1)
        # print(type(outputs))
        # print(len(outputs))
        # print("--------------3:::::", s1.shape)

        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)


@BACKBONES.register_module()
class MyNetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(MyNetworkImageNet, self).__init__()
        genotype = PC_DARTS_image
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        # outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            # if i == 1 * self._layers // 3:
                # outputs.append(s1)
                # print("--------------1:::::",s1.shape)
            # if i == 2 * self._layers // 3:
            #     outputs.append(s1)
            #     print("--------------2:::::", s1.shape)
        outputs.append(s1)
        # print(type(outputs))
        # print(len(outputs))
        # print("--------------3:::::", s1.shape)

        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)
    # def forward(self, input):
    #     logits_aux = None
    #     s0 = self.stem0(input)
    #     s1 = self.stem1(s0)
    #     for i, cell in enumerate(self.cells):
    #         # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    #         s0, s1 = s1, cell(s0, s1, 0)
    #         if i == 2 * self._layers // 3:
    #             if self._auxiliary and self.training:
    #                 logits_aux = self.auxiliary_head(s1)
    #     out = self.global_pooling(s1)
    #     logits = self.classifier(out.view(out.size(0), -1))
    #     return logits, logits_aux

@BACKBONES.register_module()
class MyNetworkImageNet_FPN(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(MyNetworkImageNet_FPN, self).__init__()
        genotype = PC_DARTS_image
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            if i == 1 * self._layers // 3:
                outputs.append(s0)
                # print("--------------1:::::",s0.shape)
            if i == 2 * self._layers // 3:
                outputs.append(s0)
                # print("--------------2:::::", s0.shape)
        outputs.append(s1)
        # print("--------------3:::::", s1.shape)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)


@BACKBONES.register_module()
class MyNetworkImageNet_FPN_search(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(MyNetworkImageNet_FPN_search, self).__init__()
        genotype = PC_DARTS_sctd
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            if i == 1 * self._layers // 3:
                outputs.append(s0)
                # print("--------------1:::::",s0.shape)
            if i == 2 * self._layers // 3:
                outputs.append(s0)
                # print("--------------2:::::", s0.shape)
        outputs.append(s1)
        # print("--------------3:::::", s1.shape)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)

@BACKBONES.register_module()
class NetworkImageNet_FPN(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(NetworkImageNet_FPN, self).__init__()
        genotype = DARTS_image
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            if i == 1 * self._layers // 3:
                outputs.append(s0)
                # print("--------------1:::::",s1.shape)
            if i == 2 * self._layers // 3:
                outputs.append(s0)
                # print("--------------2:::::", s1.shape)
        outputs.append(s1)
        # print("--------------3:::::", s1.shape)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)


@BACKBONES.register_module()
class NetworkImageNet_FPN3(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary):
        super(NetworkImageNet_FPN3, self).__init__()
        genotype = DARTS_image
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),  # (224-3+2)/2+1=112.5=112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),  # (112-3+2)/2+1=56.5=56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),  # (56-3+2)/2+1=28.5=28
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def init_weights(self, pretrained=None):
      if isinstance(pretrained, str):
          from mmdet.apis import get_root_logger
          logger = get_root_logger()
          load_checkpoint(self, pretrained, strict=False, logger=logger)
      # elif pretrained is None:
      #     for m in self.modules():
      #         if isinstance(m, nn.Conv2d):
      #             kaiming_init(m)
      #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
      #             constant_init(m, 1)

    def forward(self, input):
        outputs = []
        # logits_aux = None
        s0 = self.stem0(input)
        # outputs.append(s0)
        # print("--------------0:::::", s0.shape)
        s1 = self.stem1(s0)
        # if i in [layers // 3, 2 * layers // 3]:
        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0)
            if i == 1 * self._layers // 3:
                outputs.append(s0)
                # print("--------------1:::::",s1.shape)
            if i == 2 * self._layers // 3:
                outputs.append(s0)
                # print("--------------2:::::", s1.shape)
        outputs.append(s1)
        # print("--------------3:::::", s1.shape)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        return tuple(outputs)
