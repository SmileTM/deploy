# -*- coding: utf-8 -*-
#
# File: create_bert_model.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 04.03.2022
#
import tensorflow as tf
from transformers import TFAutoModel
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class MyMode(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyMode, self).__init__(**kwargs)
        self.plm = TFAutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.dense = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        input_ids, attention_mask, token_type_ids = inputs
        plm_out = self.plm(input_ids, attention_mask, token_type_ids, return_dict=True)
        sequence_output, pooled_output = plm_out['last_hidden_state'], plm_out['pooler_output']
        out = self.dense(pooled_output)
        return out


if __name__ == '__main__':
    data = tf.ones((1, 512), dtype=tf.int64)
    model = MyMode()
    out = model((data, data, data))
    print(out)
    model.save("../resources/testBert")
    for i in tqdm(range(50)):
        model((data, data, data))
