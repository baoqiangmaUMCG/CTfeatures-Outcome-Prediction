import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
        
        
        
def conv1x1x1(in_planes, out_planes, bias=False):
    # 1x1x1 convolution
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def SeparableConv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # the separable convolutions consists of a 1x3x3 conv and a 3x1x1 conv
    # to replace 3x3x3 conv [conv3x3x3(inplanes, planes, kernel_size=3, bias=False)]
    return nn.Sequential(nn.Conv3d(in_planes, 1, kernel_size=[1, 3, 3],
                                   stride=[1, stride, stride], padding=[0, 1, 1], bias=bias),
                         nn.Conv3d(1, out_planes, kernel_size=[3, 1, 1],
                                   stride=[stride, 1, 1], padding=[1, 0, 0], bias=bias))


def SeparableTransConv3x3x3(in_planes, out_planes, stride=1, output_pad=[0, 0, 0], bias=False):
    # the separable transpose convolution 
    return nn.Sequential(nn.ConvTranspose3d(in_planes, 1, kernel_size=[1, 3, 3],
                                            stride=[1, stride, stride], padding=[0, 1, 1],
                                            output_padding=[0, output_pad[1], output_pad[2]], bias=bias),
                         nn.ConvTranspose3d(1, out_planes, kernel_size=[3, 1, 1],
                                            stride=[stride, 1, 1], padding=[1, 0, 0],
                                            output_padding=[output_pad[0], 0, 0], bias=bias))


def SeparableConv7x7x7(in_planes, out_planes):
    # the separable convolutions consists of a 1x7x7 conv and a 7x1x1 conv
    # to replace 7x7x7 conv [nn.Conv3d(input_channel,64,kernel_size=7,stride=(1,2,2),padding=(3,3,3),bias=False)]
    return nn.Sequential(nn.Conv3d(in_planes, 1, kernel_size=[1, 7, 7],
                                   stride=[1, 2, 2], padding=[0, 3, 3], bias=False),
                         nn.Conv3d(1, out_planes, kernel_size=[7, 1, 1],
                                   stride=[2,1,1], padding=[3, 0, 0], bias=False))


def SeparableTransConv7x7x7(in_planes, out_planes, output_pad=[0, 0, 0]):
    # the separable Transpose convolutions consists of a 1x7x7 TransConv and a 7x1x1 TransConv
    # to replace 7x7x7 TransConv 
    return nn.Sequential(nn.ConvTranspose3d(in_planes, 1, kernel_size=[1, 7, 7],
                                            stride=[1, 2, 2], padding=[0, 3, 3],
                                            output_padding=[0, output_pad[1], output_pad[2]], bias=False),
                         nn.ConvTranspose3d(1, out_planes, kernel_size=[7, 1, 1],
                                            stride=[2,1,1], padding=[3, 0, 0], output_padding=[output_pad[0], 0, 0],
                                            bias=False))


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish,self).__init__()
#         print("Swish activation loaded ...")
#     def forward(self,x):
#         x = x*nn.Sigmoid(x)
#         return x


class Swish(object):
    def __call__(self, input):
        input = input * nn.Sigmoid()(input)
        return input


class Mish(object):
    def __call__(self, input):
        input = input * (torch.tanh(F.softplus(input)))
        return input

    # def Mish(input):
#     input = input*(torch.tanh(F.softplus(input)))
#     return input

# class Mish(nn.Module):
#     def __init__(self):
#         super(Mish,self).__init__()
#         print("Mish activation loaded ...")
#     def forward(self,x):
#         x = x*(torch.tanh(F.softplus(x)))
#         return x


class BasicBlock(nn.Module):
    """
    Residual block.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, actfn=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        self.conv1 = SeparableConv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.actfn = actfn
        self.conv2 = SeparableConv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.actfn(out)

        return out


class BasicBlock_Decoder(nn.Module):
    """
    Residual block, potentially with upsampling.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, actfn=nn.ReLU(inplace=True), outputpadding=[0, 0, 0]):
        super(BasicBlock_Decoder, self).__init__()
        self.conv1 = SeparableTransConv3x3x3(inplanes, planes, stride, output_pad=outputpadding)
        # self.conv1_1 = nn.ConvTranspose3d(inplanes,1,kernel_size=[1,3,3],
        #                 stride = [1,stride,stride],padding=[0,1,1],output_padding = [0,1,1],bias = False)
        # self.conv1_2 = nn.ConvTranspose3d(1,planes,kernel_size=[3,1,1],
        #                 stride = [stride,1,1],padding = [1,0,0],output_padding = [1,0,0],bias = False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.actfn = actfn
        self.conv2 = SeparableTransConv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.conv1_1(x)
        # out = self.conv1_2(out)
        out = self.bn1(out)
        out = self.actfn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.actfn(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, actfn=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        self.actfn = actfn
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.actfn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.actfn(out)

        return out


class ResEncoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 extra_featuresize,
                 actfn,
                 shortcut_type='B',
                 input_channel=1,
                 z_size = 512):
        self.inplanes = 64
        super(ResEncoder, self).__init__()
        self.conv1 = SeparableConv7x7x7(input_channel, self.inplanes)
        self.bn1 = nn.BatchNorm3d(self.inplanes)

        if actfn == 'relu':
            self.actfn = nn.ReLU(inplace=True)
        elif actfn == 'swish':
            self.actfn = Swish()
        elif actfn == 'tanh':
            self.actfn = nn.Tanh()
        elif actfn == 'leakyrelu':
            self.actfn = nn.LeakyReLU(inplace=True)
        elif actfn == 'mish':
            self.actfn = Mish()

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], shortcut_type, actfn=self.actfn)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, actfn=self.actfn)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, actfn=self.actfn)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, actfn=self.actfn)
        
        self.conv1x1x1_1_to_top = conv1x1x1(in_planes=64, out_planes=128)
        self.conv1x1x1_2_to_top = conv1x1x1(in_planes=128, out_planes=256)
        self.conv1x1x1_3_to_top = conv1x1x1(in_planes=256, out_planes=512)

        last_duration = int(math.ceil(sample_duration / 32))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512 * block.expansion, z_size)

        # -------Dropout layers
        # self.dropout = nn.Dropout(0.5)
        # -------

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, actfn=nn.ReLU(inplace=True)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':  # at this moment disabled
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, actfn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.actfn(x0)  # shape: (B, 64, 92, 90, 90)
        
        x0_mp = self.maxpool(x0)  # shape: (B, 64, 46, 45, 45)
        
        x1 = self.layer1(x0_mp)  # shape: (B, 64, 46, 45, 45)
        x2 = self.layer2(x1)  # shape: (B, 128, 23, 23, 23)
        x3 = self.layer3(x2)  # shape: (B, 256, 12, 12, 12)
        x4 = self.layer4(x3)  # shape: (B, 512, 6, 6, 6)
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResDecoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 extra_featuresize,
                 actfn,
                 shortcut_type='B',
                 input_channel=512):
        super(ResDecoder, self).__init__()
        self.inplanes = input_channel

        if actfn == 'relu':
            self.actfn = nn.ReLU(inplace=True)
        elif actfn == 'swish':
            self.actfn = Swish()
        elif actfn == 'tanh':
            self.actfn = nn.Tanh()
        elif actfn == 'leakyrelu':
            self.actfn = nn.LeakyReLU(inplace=True)
        elif actfn == 'mish':
            self.actfn = Mish()

        last_duration = int(math.ceil(sample_duration / 32))
        last_size = int(math.ceil(sample_size / 32))

        self.Upsample3D = nn.Upsample(scale_factor=last_duration, mode='trilinear', align_corners=False)
        self.layerup4 = self._make_layer_upsample(
            block, 256, layers[3], shortcut_type, stride=2, actfn=self.actfn, output_pad=[0, 0, 0])
        self.layerup3 = self._make_layer_upsample(
            block, 128, layers[2], shortcut_type, stride=2, actfn=self.actfn, output_pad=[1, 1, 1])
        self.layerup2 = self._make_layer_upsample(
            block, 64, layers[1], shortcut_type, stride=2, actfn=self.actfn, output_pad=[1, 1, 1])
        self.layerup1 = self._make_layer_upsample(
            block, self.inplanes, layers[0], shortcut_type, actfn=self.actfn)

        self.Upsample3D_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv1 = SeparableTransConv7x7x7(self.inplanes, 1, output_pad=[1, 1, 1])
        # self.bn1 = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

        # ------- Dropout layers
        # self.dropout = nn.Dropout(0.5)
        # -------

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_upsample(self, block, planes, blocks, shortcut_type, stride=1, actfn=nn.ReLU(inplace=True),
                             output_pad=[0, 0, 0]):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                upsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                upsample = nn.Sequential(
                    SeparableTransConv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        stride=stride,
                        output_pad=output_pad,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, actfn, outputpadding=output_pad))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        x = self.Upsample3D(x)

        x = self.layerup4(x)
        x = self.layerup3(x)
        x = self.layerup2(x)
        x = self.layerup1(x)

        x = self.Upsample3D_2(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.sigmoid(x)
        
        return x


class Res3DAutoencoder(nn.Module):
    def __init__(self,
                 encoderblock,
                 decoderblock,
                 layers,
                 sample_size,
                 sample_duration,
                 extra_featuresize,
                 actfn,
                 shortcut_type='B',
                 input_channel=1,
                 z_size=512):
        super(Res3DAutoencoder, self).__init__()

        self.ResEncoder = ResEncoder(encoderblock,
                                     layers,
                                     sample_size,
                                     sample_duration,
                                     extra_featuresize,
                                     actfn,
                                     shortcut_type=shortcut_type,
                                     input_channel=input_channel)

        self.ResDecoder = ResDecoder(decoderblock,
                                     layers,
                                     sample_size,
                                     sample_duration,
                                     extra_featuresize,
                                     actfn,
                                     shortcut_type=shortcut_type)

    def forward(self, x):
        x_latent = self.ResEncoder(x)
        x = self.ResDecoder(x_latent)
        print(x_latent.size())

        return x, x_latent
        
        
class Discriminator(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 extra_featuresize,
                 actfn,
                 shortcut_type='B',
                 input_channel=1):
        self.inplanes = 64
        super(Discriminator, self).__init__()
        self.conv1 = SeparableConv7x7x7(input_channel, self.inplanes)
        self.bn1 = nn.BatchNorm3d(self.inplanes)

        if actfn == 'relu':
            self.actfn = nn.ReLU(inplace=True)
        elif actfn == 'swish':
            self.actfn = Swish()
        elif actfn == 'tanh':
            self.actfn = nn.Tanh()
        elif actfn == 'leakyrelu':
            self.actfn = nn.LeakyReLU(inplace=True)
        elif actfn == 'mish':
            self.actfn = Mish()

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], shortcut_type, actfn=self.actfn)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, actfn=self.actfn)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, actfn=self.actfn)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, actfn=self.actfn)
        

        last_duration = int(math.ceil(sample_duration / 32))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.fc1 = nn.Linear(512 * block.expansion, 128)
        self.fc2 = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, actfn=nn.ReLU(inplace=True)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':  # at this moment disabled
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, actfn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.actfn(x0)  # shape: (B, 64, 92, 90, 90)
        
        x0_mp = self.maxpool(x0)  # shape: (B, 64, 46, 45, 45)
        
        x1 = self.layer1(x0_mp)  # shape: (B, 64, 46, 45, 45)
        x2 = self.layer2(x1)  # shape: (B, 128, 23, 23, 23)
        x3 = self.layer3(x2)  # shape: (B, 256, 12, 12, 12)
        x4 = self.layer4(x3)  # shape: (B, 512, 6, 6, 6)
        
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.fc1(x)                    
        x = self.actfn(x)
        x = self.fc2(x)
    
        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    # NEW (actually not NEW code, but a question): why maximum of 5 layers?
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = Res3DAutoencoder(BasicBlock, BasicBlock_Decoder, [1, 1, 1, 1], **kwargs)
    return model
    
def discriminator10(**kwargs):
    """Constructs a discriminator-10 model.
    """
    model = Discriminator(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = Res3DAutoencoder(BasicBlock, BasicBlock_Decoder, [2, 2, 2, 2], **kwargs)
    return model
    
def discriminator18(**kwargs):
    """Constructs a discriminator-18 model.
    """
    model = Discriminator(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = Res3DAutoencoder(BasicBlock, BasicBlock_Decoder, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Res3DAutoencoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Res3DAutoencoder(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = Res3DAutoencoder(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = Res3DAutoencoder(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
