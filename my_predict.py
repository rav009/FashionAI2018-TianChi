import mxnet as mx
import numpy as np
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn

#print(mx.nd.load('G:\My Programs\FashionAI2018-TianChi-master\models\collar_design_labels_resnet50_v2_10_15_final.params').keys())

pretrained_net = gluon.model_zoo.vision.get_model('resnet50_v2', pretrained=True)
#ctx = mx.cpu(0)
net = gluon.model_zoo.vision.get_model('resnet50_v2', classes=5)
net.features = pretrained_net.features
#finetune_net.output.initialize(init.Xavier(), ctx = ctx)
#finetune_net.collect_params().reset_ctx(ctx)
#finetune_net.hybridize()
net.collect_params().load('G:\My Programs\FashionAI2018-TianChi-master\models\collar_design_labels_resnet50_v2_10_15_final.params')

def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, :, ::-1]

    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],#center crop
        img[:, 0:H, 0:W],#left top corner
        img[:, iH - H:iH, 0:W],#left bottom corner
        img[:, 0:H, iW - W:iW],#right top corner
        img[:, iH - H:iH, iW - W:iW],#right bottom corner

        ## new define
        img[:, 0:H, (iW - W) // 2:(iW + W) // 2], #middle top
        img[:, iH - H:iH, (iW - W) // 2:(iW + W) // 2],#middle bottom
        img[:, (iH - H) // 2:(iH + H) // 2, 0:W],#left middle
        img[:, (iH - H) // 2:(iH + H) // 2, iW - W:iW],#right middle


        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],

        img_flip[:, 0:H, (iW - W) // 2:(iW + W) // 2], #middle top
        img_flip[:, iH - H:iH, (iW - W) // 2:(iW + W) // 2],#middle bottom
        img_flip[:, (iH - H) // 2:(iH + H) // 2, 0:W],#left middle
        img_flip[:, (iH - H) // 2:(iH + H) // 2, iW - W:iW],#right middle

    )
    return (crops)

## Test time transform function , 19-crop  ##
def transform_predict(im, size):
    im = im.astype('float32') / 255
    im = image.resize_short(im, size, interp=1)
    # im = image.resize_short(im, 331)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # im = forty_crop(im, (352, 352))
    im = ten_crop(im, (448, 448))
    return (im)

net.hybridize()


img_path = """G:\My Programs\FashionAI\\train_valid_allset\collar_design_labels\\train\\1\\0a6b9e7ddb72561658f388af7570847d.jpg"""
with open(img_path, 'rb') as f:
    img = image.imdecode(f.read())
out_all = np.zeros([5, ])
###### Test Time augmentation (muti-scale test) ######
input_scale = [448,480,512]
for scale in input_scale:
    data = transform_predict(img, scale)
    with ag.predict_mode():
        out = net(data.as_in_context(mx. cpu(0)))  # 随机crop十张图片,所以此处是10张图片的结果
        out = nd.SoftmaxActivation(out).mean(axis=0)  # 取softmax,然后对十个结果取平均
        out_all += out.asnumpy()
out = out_all / len(input_scale)
print(out)
net.export('collar_design_labels_resnet50_v2_10_15_final', epoch=15)
