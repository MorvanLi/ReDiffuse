import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import re
import matplotlib.pyplot as plt


def extract_number(name):
    num = re.search(r'\((\d+)\)', name)
    return int(num.group(1)) if num else float('inf')

class MFI_Dataset(Dataset):
    def __init__(self, datasetPath, phase, use_dataTransform, resize, imgSzie):
        super(MFI_Dataset, self).__init__()
        self.datasetPath = datasetPath
        self.phase = phase
        self.use_dataTransform = use_dataTransform
        self.resize = resize
        self.imgSzie = imgSzie

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        dirsName = os.listdir(self.datasetPath)
        assert len(dirsName) >= 2, "Please check that the dataset is formatted correctly."
        dirsPath = os.path.join(self.datasetPath, dirsName[0])
        return len(os.listdir(dirsPath))

    def __getitem__(self, index):
        if self.phase == "train":
            # source image1

            # source image1
            sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
            # sourceImg1_names = os.listdir(sourceImg1_dirPath)
            sourceImg1_names = sorted(
                os.listdir(sourceImg1_dirPath),
                key=extract_number
            )
            sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
            sourceImg1 = cv2.imread(sourceImg1_path)

            # convert RGB image to YCbCr
            if len(sourceImg1.shape) == 3 & sourceImg1.shape[-1] > 1:
                # ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2YCR_CB)
                ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_BGR2YCrCb)
                sourceImg1 = ycbcr_sourceImg1[:, :, 0]

            # source image2
            sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
            # sourceImg2_names = os.listdir(sourceImg2_dirPath)
            sourceImg2_names = sorted(
                os.listdir(sourceImg2_dirPath),
                key=extract_number
            )
            sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
            sourceImg2 = cv2.imread(sourceImg2_path)

            # convert RGB image to YCbCr
            if len(sourceImg2.shape) == 3 & sourceImg2.shape[-1] > 1:
                # ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_BGR2YCrCb)
                sourceImg2 = ycbcr_sourceImg2[:, :, 0]

            # full_clear image
            clearImg_dirPath = os.path.join(self.datasetPath, "full_clear")
            # clearImg_names = os.listdir(clearImg_dirPath)
            clearImg_names = sorted(
                os.listdir(clearImg_dirPath),
                key=extract_number
            )
            clearImg_path = os.path.join(clearImg_dirPath, clearImg_names[index])
            clearImg = cv2.imread(clearImg_path)

            # convert RGB image to YCbCr
            if len(clearImg.shape) == 3 & clearImg.shape[-1] > 1:
                # ycbcr_clearImg = cv2.cvtColor(clearImg, cv2.COLOR_RGB2YCR_CB)
                ycbcr_clearImg = cv2.cvtColor(clearImg, cv2.COLOR_BGR2YCrCb)
                clearImg = ycbcr_clearImg[:, :, 0]

            # plt.figure()
            # plt.imshow(clearImg, cmap='gray')
            # plt.show()

            x, y = sourceImg1.shape
            x_dim = random.randint(0, x - 256)
            y_dim = random.randint(0, y - 256)
            sourceImg1 = sourceImg1[x_dim:x_dim + 256, y_dim:y_dim + 256]
            sourceImg2 = sourceImg2[x_dim:x_dim + 256, y_dim:y_dim + 256]
            clearImg = clearImg[x_dim:x_dim + 256, y_dim:y_dim + 256]
            if self.resize:
                sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))
                clearImg = cv2.resize(clearImg, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg1 = self.transform(sourceImg1)
                sourceImg2 = self.transform(sourceImg2)
                clearImg = self.transform(clearImg)

            return [sourceImg1, sourceImg2, clearImg]

        if self.phase == "valid":
            # source image1
            sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
            # sourceImg1_names = os.listdir(sourceImg1_dirPath)
            sourceImg1_names = sorted(
                os.listdir(sourceImg1_dirPath),
                key=lambda x: int(os.path.splitext(x)[0])
            )
            sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
            sourceImg1 = cv2.imread(sourceImg1_path)


            # convert RGB image to YCbCr
            # ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2YCR_CB)
            ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_BGR2YCrCb)
            sourceImg1 = ycbcr_sourceImg1[:, :, 0]
            sourceImg1_cr = ycbcr_sourceImg1[:, :, 1]
            sourceImg1_cb = ycbcr_sourceImg1[:, :, 2]

            # source image2
            sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
            # sourceImg2_names = os.listdir(sourceImg2_dirPath)
            sourceImg2_names = sorted(
                os.listdir(sourceImg2_dirPath),
                key=lambda x: int(os.path.splitext(x)[0])
            )
            sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
            sourceImg2 = cv2.imread(sourceImg2_path)

            # convert RGB image to YCbCr
            # ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
            ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_BGR2YCrCb)
            sourceImg2 = ycbcr_sourceImg2[:, :, 0]
            sourceImg2_cr = ycbcr_sourceImg2[:, :, 1]
            sourceImg2_cb = ycbcr_sourceImg2[:, :, 2]

            if self.resize:
                sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg1 = self.transform(sourceImg1)
                sourceImg2 = self.transform(sourceImg2)


            return [sourceImg1, sourceImg2, sourceImg1_cr, sourceImg1_cb, sourceImg2_cr, sourceImg2_cb]