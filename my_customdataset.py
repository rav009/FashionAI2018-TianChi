from mxnet import gluon, image, init, nd
import random
import os

class my_custom_dataset(gluon.data.Dataset):
    def unique_index(self, L, e):  # return the index of e
        index = []
        for (i, j) in enumerate(L):
            if j == e:
                index.append(i)
        return index


    def __init__(self, imgroot, labelmasterpath, transform=None):
        self.img_list = {}
        for dirname in os.listdir(imgroot):
            imgdir = os.path.join(imgroot, dirname)
            for fn in os.listdir(imgdir):
                assert fn not in self.img_list.keys()
                self.img_list[fn] = os.path.join(imgdir, fn)
        self._transform = transform
        self.items = []
        with open(labelmasterpath, 'r') as f:
            samplelist = f.readlines()
            #random.shuffle(train)
            for line in samplelist:
                tmp = line.strip().split(',')
                image_short_name = tmp[0]
                if image_short_name in self.img_list.keys():
                    label = tmp[2]
                    label_length = len(label)
                    y = list(label).index('y')
                    m = self.unique_index(list(label), 'm')
                    label_final = [99] * label_length
                    label_final[0] = y
                    n = 1
                    for i in m:
                        label_final[n] = i
                        n += 1
                    self.items.append((self.img_list[image_short_name], label_final))
        #print(self.items)

    def __getitem__(self, idx):
        #print(self.items[idx][0])
        img = image.imread(self.items[idx][0])
        label = self.items[idx][1]
        # if len(label)!=1:
        #     print('mmmmmmm')
        #abel_m = self.items[idx][2]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)