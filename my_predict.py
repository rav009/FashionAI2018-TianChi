import mxnet as mx
import numpy as np
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag


model_name = "coat_length_labels"
model_path = "C:\\FR Projects\\FashionAI\\models\\coat_length_labels_resnet50_v2_10_15_final.params"
pic_path = "C:\\Users\\01205691\\Pictures\\PAUL01.jpg"
ctx = mx.cpu(0)

if_design = True if "_design_" in model_name else False
if_length = True if "_length_" in model_name else False

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}
assert model_name in task_list.keys()
pretrained_net = gluon.model_zoo.vision.get_model('resnet50_v2', pretrained=True)
net = gluon.model_zoo.vision.get_model('resnet50_v2', classes=task_list[model_name])
net.features = pretrained_net.features
net.collect_params().load(model_path)
net.hybridize()

if if_design:
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
        return crops


    def transform_predict(im, size):
        im = im.astype('float32') / 255
        im = image.resize_short(im, size, interp=1)
        # im = image.resize_short(im, 331)
        im = nd.transpose(im, (2,0,1))
        im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # im = forty_crop(im, (352, 352))
        im = ten_crop(im, (448, 448))
        return im

    with open(pic_path, 'rb') as f:
        img = image.imdecode(f.read())
    out_all = np.zeros([task_list[model_name], ])
    input_scale = [448, 480, 512]
    for scale in input_scale:
        data = transform_predict(img, scale)
        with ag.predict_mode():
            out = net(data.as_in_context(ctx))  # 随机crop十张图片,所以此处是10张图片的结果
            out = nd.SoftmaxActivation(out).mean(axis=0)  # 取softmax,然后对十个结果取平均
            out_all += out.asnumpy()
    out = out_all / len(input_scale)
    print(out)
    net.export(model_path.split('\\')[-1].replace(".params", ""), epoch=15)

elif if_length:
    def two_crop(img):
        img_flip = img[:, :, ::-1]
        crops = nd.stack(
            img,
            img_flip,
        )
        return crops


    def transform_predict(im, size):
        im = im.astype('float32') / 255
        # im = image.resize_short(im, size, interp=1)
        im = image.imresize(im, size, size, interp=1)
        # im = image.resize_short(im, 331)
        im = nd.transpose(im, (2, 0, 1))
        im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # im = forty_crop(im, (352, 352))
        im = two_crop(im)
        return im

    input_scale = 480
    with open(pic_path, 'rb') as f:
        img = image.imdecode(f.read())
    data = transform_predict(img, input_scale)
    with ag.predict_mode():
        out = net(data.as_in_context(ctx))
        out = nd.SoftmaxActivation(out).mean(axis=0)
    print(out)
    net.export(model_path.split('\\')[-1].replace(".params", ""), epoch=15)
else:
    raise NotImplementedError


