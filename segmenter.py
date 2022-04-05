import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model


class SegmentPerson:
    def __init__(self, output_type=None):
        self.model = create_model("Unet_2020-07-20")
        self.model.eval()
        valid_list = ['marked', 'cropped']
        if output_type in valid_list:
            self.out_type = output_type
        else:
            print('Select a valid Type : ', valid_list)
            print("Default Mode : marked")
            self.out_type = 'marked'

    def semantic_segment(self, filename):
        image = load_rgb(filename)
        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
        with torch.no_grad():
            prediction = self.model(x)[0][0]
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        mask3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        if self.out_type == 'cropped':
            cropped_img = image * mask3d
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            return cropped_img
        else:
            marked_img = cv2.addWeighted(image, 0.5, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
            return marked_img


def main():
    filename = 'test.jpg'
    sp = SegmentPerson(output_type='cropped')
    out = sp.semantic_segment(filename)
    cv2.imshow('Segmented Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
