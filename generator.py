"""
Authors : inzapp

Github url : https://github.com/inzapp/lvgan

Copyright 2024 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import numpy as np

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 ae_e,
                 gan_g,
                 image_paths,
                 generate_shape,
                 batch_size,
                 latent_dim,
                 dtype='float32'):
        self.ae_e = ae_e
        self.gan_g = gan_g
        self.image_paths = image_paths
        self.generate_shape = generate_shape
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(self.image_paths)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        from lvgan import LVGAN
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        ae_x = []
        for f in fs:
            img = f.result()
            img = self.resize(img, (self.generate_shape[1], self.generate_shape[0]))
            x = self.normalize(np.asarray(img).reshape(self.generate_shape))
            ae_x.append(x)
        ae_x = np.asarray(ae_x).astype(self.dtype)
        half_ae_x = ae_x[:self.half_batch_size]
        real_dx = np.asarray(LVGAN.graph_forward(model=self.ae_e, x=half_ae_x)).reshape((self.half_batch_size, self.latent_dim)).astype(self.dtype)
        z = self.get_z_vector(size=self.batch_size * self.latent_dim).reshape((self.batch_size, self.latent_dim)).astype(self.dtype)
        half_z = z[:self.half_batch_size]
        fake_dx = np.asarray(LVGAN.graph_forward(model=self.gan_g, x=half_z)).reshape((self.half_batch_size, self.latent_dim)).astype(self.dtype)
        real_dy = np.ones((self.half_batch_size, 1)).astype(self.dtype)
        fake_dy = np.zeros((self.half_batch_size, 1)).astype(self.dtype)
        dx = np.append(real_dx, fake_dx, axis=0)
        dy = np.append(real_dy, fake_dy, axis=0)
        gx = z
        gy = np.ones((self.batch_size, 1)).astype(self.dtype)
        return ae_x, dx, dy, gx, gy

    @staticmethod
    def normalize(x):
        return np.asarray(x).astype('float32') / 255.0

    @staticmethod
    def denormalize(x):
        return np.asarray(np.clip((x * 255.0), 0.0, 255.0)).astype('uint8')

    @staticmethod
    def get_z_vector(size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def resize(self, img, size):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if size[0] > img_width or size[1] > img_height:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        return cv2.resize(img, size, interpolation=interpolation)

    def load_image(self, image_path):
        return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.generate_shape[-1] == 1 else cv2.IMREAD_COLOR)

