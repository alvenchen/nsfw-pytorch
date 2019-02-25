import os
import numpy as np
import random
from torch.utils.data.dataset import Dataset
import glob
from PIL import Image, ImageFilter, ImageEnhance

def shuffle(datas, labels):
    idx = np.random.permutation(len(datas))
    x = [datas[i] for i in idx]
    y = [labels[i] for i in idx]

    return x,y


"""
the data is like:
    dir:
        sexy:
            xx.jpg
        porn:
            xx.jpg
        ...
        ...
"""
def get_dataset(path, disorder=True):
    classes_names = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    classes_names = sorted(classes_names)
    datas = []
    classes = []

    t = int(0)
    for c in classes_names:
        files = glob.glob(os.path.join(path, c, '*.jpg'))
        datas += files
        classes += [t for i in xrange(len(files))]
        t += 1
    
    if disorder:
        datas, classes = shuffle(datas, classes)


    return datas, classes, classes_names

class DataWrapper(Dataset):
    def __init__(self, x, y, transform=None, image_mode='RGB', augumentation=True):
        self.x = x
        self.y = y
        self.image_mode = image_mode        
        self.transform = transform
        self.length = min(len(x), len(y))
        self.augumentation = augumentation

    def __getitem__(self, index):
        img_name = self.x[index]
        cls = self.y[index]

        for i in range(100):
            try:            
                img = Image.open(img_name)
                img = img.convert(self.image_mode)        
            except Exception, e:
                print("file {} exception: {}".format(img_name, str(e)))                
                img_name = self.x[(index+i+1) % self.length]
                cls = self.y[(index+i+1) % self.length]
                continue
            break

        if self.augumentation:
            img = self.randomFlip(img)
            img = self.randomBlur(img)
            img = self.randomRotation(img)
            img = self.randomColor(img)
            img = self.randomGaussian(img)

        if self.transform is not None:
            img = self.transform(img)
        

        return img, cls, img_name

    def __len__(self):        
        return self.length

    @staticmethod
    def randomFlip(image, prob=0.5):
        rnd = np.random.random_sample()
        if rnd < prob:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    @staticmethod
    def randomBlur(image, prob=0.5):
        rnd = np.random.random_sample()
        if rnd < prob:
            return image.filter(ImageFilter.BLUR)
        return image

    @staticmethod
    def randomRotation(image, prob=0.5, angle=(1, 60)):
        rnd = np.random.random_sample()
        if rnd < prob:
            random_angle = np.random.randint(angle[0], angle[1])
            return image.rotate(random_angle)
        return image

    @staticmethod
    def randomColor(image, prob=0.5, factor=(1, 90)):
        rnd = np.random.random_sample()
        if rnd < prob:
            # Factor 1.0 always returns a copy of the original image, 
            # lower factors mean less color (brightness, contrast, etc), and higher values more
            random_factor = np.random.randint(2, 18) / 10. 
            color_image = ImageEnhance.Color(image).enhance(random_factor)
            random_factor = np.random.randint(5, 18) / 10.
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
            random_factor = np.random.randint(5, 18) / 10.
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
            random_factor = np.random.randint(2, 18) / 10.
            return ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 
        return image

    @staticmethod    
    def randomGaussian(image, prob=0.5, mean=0, sigma=10):
        rnd = np.random.random_sample()
        if rnd < prob:
            img_array = np.asarray(image)
            noisy_img = img_array + np.random.normal(mean, sigma, img_array.shape)
            noisy_img = np.clip(noisy_img, 0, 255)

            return Image.fromarray(np.uint8(noisy_img))
        return image
