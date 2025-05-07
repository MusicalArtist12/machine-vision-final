"""bdd100k dataset."""

import tensorflow_datasets as tfds
import json
import numpy as np
import cv2 as cv
import math
import posixpath


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bdd100k dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(bdd100k): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features = tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(720, 1280, 3)),
                'seg_map': tfds.features.Image(shape=(720, 1280, 1))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys = ('image', 'seg_map'),  # Set tstr(image_path / img_id), str(seg_path / mask_id)o `None` to disable
            homepage = 'http://bdd-data.berkeley.edu/'
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = dl_manager.download_and_extract({
            'images': 'http://128.32.162.150/bdd100k/bdd100k_images_10k.zip',
            'seg_maps': 'http://128.32.162.150/bdd100k/bdd100k_seg_maps.zip'
        })

        return {
            'train': self._generate_examples(
                image_path = path['images'] / '10k/train',
                seg_path = path['seg_maps'] / 'color_labels/train'
            ),
            'val': self._generate_examples(
                image_path = path['images'] / '10k/val',
                seg_path = path['seg_maps'] / 'color_labels/val'
            ),
        }

    def _generate_examples(self, image_path, seg_path):

        image_list = []

        for image in image_path.glob("*.jpg"):
            image_mat = cv.imread(image)
            if image_mat.shape != (720, 1280, 3):
                print(f"removing {image} (shape == {image_mat.shape})")
            else:
                image_list.append(image)

        for image in image_list:


            img_id = posixpath.basename(image)

            img_key = img_id[:-len(".jpg")]

            mask_id = img_key + '_train_color.png'

            img = cv.imread(str(seg_path / mask_id))

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            lower = np.array([
                120,
                0,
                0
            ])
            upper = np.array([
                120,
                255,
                150
            ])
            mask = cv.inRange(hsv, lower, upper)

            res = cv.bitwise_and(img, img, mask = mask)
            res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
            ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY)


            final = np.full((720, 1280, 1), (0), dtype = np.uint8)

            final[res == (255)] = (1)
            # print(f'{img_id} - {mask_id}')

            # cv.imshow("window", final)
            # cv.waitKey(10)


            yield img_id, {
                'image': str(image_path / img_id),
                'seg_map': final
            }