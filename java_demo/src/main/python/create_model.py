# -*- coding: utf-8 -*-
#
# File: create_model.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 04.02.2022
#
import time

import tensorflow as tf
from tqdm import tqdm


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(2, name="d")

    def call(self, input):
        a, b = input
        out_a = self.dense(a)
        out_b = self.dense(b)
        out = tf.einsum('bf,bt->bft', out_a, out_b)
        out = tf.reduce_sum(out, axis=1)

        out = tf.argmax(out, -1)
        return out


if __name__ == '__main__':
    data = tf.random.uniform((2, 10))
    model = MyModel()
    start = time.time()
    out = model((data, data))
    model.save("../resources/test")
    # for i in tqdm(range(10000)):
    #     out = model((data, data))
    # print(time.time()-start)
    # print(out)
