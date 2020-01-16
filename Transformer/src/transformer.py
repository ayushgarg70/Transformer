import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Softmax, Dense, BatchNormalization, Flatten, Reshape, Input
import pandas as pd
import sys


class Transformer():

    def __init__(self,train_length):
        self.train_length=train_length

    def encoder(self):
        self.a1 = tf.compat.v1.placeholder(shape=[2, 1], dtype=tf.float32, name="a1")
        self.a2 = tf.compat.v1.placeholder(shape=[2, 1], dtype=tf.float32, name="a2")
        self.a3 = tf.compat.v1.placeholder(shape=[2, 1], dtype=tf.float32, name="a3")
        self.a4 = tf.compat.v1.placeholder(shape=[2, 1], dtype=tf.float32, name="a4")
        self.a5 = tf.compat.v1.placeholder(shape=[2, 1], dtype=tf.float32, name="a5")

        self.W11 = tf.compat.v1.get_variable(name="W11", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W12 = tf.compat.v1.get_variable(name="W12", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W13 = tf.compat.v1.get_variable(name="W13", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W14 = tf.compat.v1.get_variable(name="W14", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W15 = tf.compat.v1.get_variable(name="W15", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)

        self.W21 = tf.compat.v1.get_variable(name="W21", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W22 = tf.compat.v1.get_variable(name="W22", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W23 = tf.compat.v1.get_variable(name="W23", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W24 = tf.compat.v1.get_variable(name="W24", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W25 = tf.compat.v1.get_variable(name="W25", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)

        self.W31 = tf.compat.v1.get_variable(name="W31", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W32 = tf.compat.v1.get_variable(name="W32", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W33 = tf.compat.v1.get_variable(name="W33", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W34 = tf.compat.v1.get_variable(name="W34", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W35 = tf.compat.v1.get_variable(name="W35", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)

        self.W41 = tf.compat.v1.get_variable(name="W41", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W42 = tf.compat.v1.get_variable(name="W42", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W43 = tf.compat.v1.get_variable(name="W43", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W44 = tf.compat.v1.get_variable(name="W44", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W45 = tf.compat.v1.get_variable(name="W45", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)

        self.W51 = tf.compat.v1.get_variable(name="W51", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W52 = tf.compat.v1.get_variable(name="W52", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W53 = tf.compat.v1.get_variable(name="W53", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W54 = tf.compat.v1.get_variable(name="W54", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)
        self.W55 = tf.compat.v1.get_variable(name="W55", shape=[2, 2], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer)

        self.a11 = tf.matmul(self.W11, self.a1)
        self.a12 = tf.matmul(self.W12, self.a1)
        self.a13 = tf.matmul(self.W13, self.a1)
        self.a14 = tf.matmul(self.W14, self.a1)
        self.a15 = tf.matmul(self.W15, self.a1)

        self.a21 = tf.matmul(self.W21, self.a2)
        self.a22 = tf.matmul(self.W22, self.a2)
        self.a23 = tf.matmul(self.W23, self.a2)
        self.a24 = tf.matmul(self.W24, self.a2)
        self.a25 = tf.matmul(self.W25, self.a2)

        self.a31 = tf.matmul(self.W31, self.a3)
        self.a32 = tf.matmul(self.W32, self.a3)
        self.a33 = tf.matmul(self.W33, self.a3)
        self.a34 = tf.matmul(self.W34, self.a3)
        self.a35 = tf.matmul(self.W35, self.a3)

        self.a41 = tf.matmul(self.W41, self.a4)
        self.a42 = tf.matmul(self.W42, self.a4)
        self.a43 = tf.matmul(self.W43, self.a4)
        self.a44 = tf.matmul(self.W44, self.a4)
        self.a45 = tf.matmul(self.W45, self.a4)

        self.a51 = tf.matmul(self.W51, self.a5)
        self.a52 = tf.matmul(self.W52, self.a5)
        self.a53 = tf.matmul(self.W53, self.a5)
        self.a54 = tf.matmul(self.W54, self.a5)
        self.a55 = tf.matmul(self.W55, self.a5)

        self.x121 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a11, self.a21))
        self.x131 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a11, self.a31))
        self.x141 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a11, self.a41))
        self.x151 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a11, self.a51))
        input11 = tf.stack([self.x121, self.x131, self.x141, self.x151])

        self.x211 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a21, self.a11))
        self.x231 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a21, self.a31))
        self.x241 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a21, self.a41))
        self.x251 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a21, self.a51))
        input21 = tf.stack([self.x211, self.x231, self.x241, self.x251])

        self.x311 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a31, self.a11))
        self.x321 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a31, self.a21))
        self.x341 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a31, self.a41))
        self.x351 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a31, self.a51))
        input31 = tf.stack([self.x311, self.x321, self.x341, self.x351])

        self.x411 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a41, self.a11))
        self.x421 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a41, self.a21))
        self.x431 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a41, self.a31))
        self.x451 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a41, self.a51))
        input41 = tf.stack([self.x411, self.x421, self.x431, self.x451])

        self.x511 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a51, self.a11))
        self.x521 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a51, self.a21))
        self.x531 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a51, self.a31))
        self.x541 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a51, self.a41))
        input51 = tf.stack([self.x511, self.x521, self.x531, self.x541])

        softmax11 = Softmax()(tf.expand_dims(input11, 1))
        softmax21 = Softmax()(tf.expand_dims(input21, 1))
        softmax31 = Softmax()(tf.expand_dims(input31, 1))
        softmax41 = Softmax()(tf.expand_dims(input41, 1))
        softmax51 = Softmax()(tf.expand_dims(input51, 1))

        preffnn11 = tf.compat.v1.matmul(tf.concat([self.a21, self.a31, self.a41, self.a51], 1), softmax11) + self.a11
        preffnn21 = tf.compat.v1.matmul(tf.concat([self.a11, self.a31, self.a41, self.a51], 1), softmax21) + self.a21
        preffnn31 = tf.compat.v1.matmul(tf.concat([self.a11, self.a21, self.a41, self.a51], 1), softmax31) + self.a31
        preffnn41 = tf.compat.v1.matmul(tf.concat([self.a11, self.a21, self.a31, self.a51], 1), softmax41) + self.a41
        preffnn51 = tf.compat.v1.matmul(tf.concat([self.a11, self.a21, self.a31, self.a41], 1), softmax51) + self.a51

        self.x122 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a12, self.a22))
        self.x132 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a12, self.a32))
        self.x142 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a12, self.a42))
        self.x152 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a12, self.a52))
        input12 = tf.stack([self.x122, self.x132, self.x142, self.x152])

        self.x212 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a22, self.a12))
        self.x232 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a22, self.a32))
        self.x242 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a22, self.a42))
        self.x252 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a22, self.a52))
        input22 = tf.stack([self.x212, self.x232, self.x242, self.x252])

        self.x312 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a32, self.a12))
        self.x322 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a32, self.a22))
        self.x342 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a32, self.a42))
        self.x352 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a32, self.a52))
        input32 = tf.stack([self.x312, self.x322, self.x342, self.x352])

        self.x412 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a42, self.a12))
        self.x422 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a42, self.a22))
        self.x432 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a42, self.a32))
        self.x452 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a42, self.a52))
        input42 = tf.stack([self.x412, self.x422, self.x432, self.x452])

        self.x512 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a52, self.a12))
        self.x522 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a52, self.a22))
        self.x532 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a52, self.a32))
        self.x542 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a52, self.a42))
        input52 = tf.stack([self.x512, self.x522, self.x532, self.x542])

        softmax12 = Softmax()(tf.expand_dims(input12, 1))
        softmax22 = Softmax()(tf.expand_dims(input22, 1))
        softmax32 = Softmax()(tf.expand_dims(input32, 1))
        softmax42 = Softmax()(tf.expand_dims(input42, 1))
        softmax52 = Softmax()(tf.expand_dims(input52, 1))

        preffnn12 = tf.compat.v1.matmul(tf.concat([self.a22, self.a32, self.a42, self.a52], 1), softmax12) + self.a12
        preffnn22 = tf.compat.v1.matmul(tf.concat([self.a12, self.a32, self.a42, self.a52], 1), softmax22) + self.a22
        preffnn32 = tf.compat.v1.matmul(tf.concat([self.a12, self.a22, self.a42, self.a52], 1), softmax32) + self.a32
        preffnn42 = tf.compat.v1.matmul(tf.concat([self.a12, self.a22, self.a32, self.a52], 1), softmax42) + self.a42
        preffnn52 = tf.compat.v1.matmul(tf.concat([self.a12, self.a22, self.a32, self.a42], 1), softmax52) + self.a52

        self.x123 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a13, self.a23))
        self.x133 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a13, self.a33))
        self.x143 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a13, self.a43))
        self.x153 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a13, self.a53))
        input13 = tf.stack([self.x123, self.x133, self.x143, self.x153])

        self.x213 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a23, self.a13))
        self.x233 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a23, self.a33))
        self.x243 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a23, self.a43))
        self.x253 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a23, self.a53))
        input23 = tf.stack([self.x213, self.x233, self.x243, self.x253])

        self.x313 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a33, self.a13))
        self.x323 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a33, self.a23))
        self.x343 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a33, self.a43))
        self.x353 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a33, self.a53))
        input33 = tf.stack([self.x313, self.x323, self.x343, self.x353])

        self.x413 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a43, self.a13))
        self.x423 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a43, self.a23))
        self.x433 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a43, self.a33))
        self.x453 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a43, self.a53))
        input43 = tf.stack([self.x413, self.x423, self.x433, self.x453])

        self.x513 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a53, self.a13))
        self.x523 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a53, self.a23))
        self.x533 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a53, self.a33))
        self.x543 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a53, self.a43))
        input53 = tf.stack([self.x513, self.x523, self.x533, self.x543])

        softmax13 = Softmax()(tf.expand_dims(input13, 1))
        softmax23 = Softmax()(tf.expand_dims(input23, 1))
        softmax33 = Softmax()(tf.expand_dims(input33, 1))
        softmax43 = Softmax()(tf.expand_dims(input43, 1))
        softmax53 = Softmax()(tf.expand_dims(input53, 1))

        preffnn13 = tf.compat.v1.matmul(tf.concat([self.a23, self.a33, self.a43, self.a53], 1), softmax13) + self.a13
        preffnn23 = tf.compat.v1.matmul(tf.concat([self.a13, self.a33, self.a43, self.a53], 1), softmax23) + self.a23
        preffnn33 = tf.compat.v1.matmul(tf.concat([self.a13, self.a23, self.a43, self.a53], 1), softmax33) + self.a33
        preffnn43 = tf.compat.v1.matmul(tf.concat([self.a13, self.a23, self.a33, self.a53], 1), softmax43) + self.a43
        preffnn53 = tf.compat.v1.matmul(tf.concat([self.a13, self.a23, self.a33, self.a43], 1), softmax53) + self.a53

        self.x124 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a14, self.a24))
        self.x134 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a14, self.a34))
        self.x144 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a14, self.a44))
        self.x154 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a14, self.a54))
        input14 = tf.stack([self.x124, self.x134, self.x144, self.x154])

        self.x214 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a24, self.a14))
        self.x234 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a24, self.a34))
        self.x244 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a24, self.a44))
        self.x254 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a24, self.a54))
        input24 = tf.stack([self.x214, self.x234, self.x244, self.x254])

        self.x314 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a34, self.a14))
        self.x324 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a34, self.a24))
        self.x344 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a34, self.a44))
        self.x354 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a34, self.a54))
        input34 = tf.stack([self.x314, self.x324, self.x344, self.x354])

        self.x414 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a44, self.a14))
        self.x424 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a44, self.a24))
        self.x434 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a44, self.a34))
        self.x454 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a44, self.a54))
        input44 = tf.stack([self.x414, self.x424, self.x434, self.x454])

        self.x514 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a54, self.a14))
        self.x524 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a54, self.a24))
        self.x534 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a54, self.a34))
        self.x544 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a54, self.a44))
        input54 = tf.stack([self.x514, self.x524, self.x534, self.x544])

        softmax14 = Softmax()(tf.expand_dims(input14, 1))
        softmax24 = Softmax()(tf.expand_dims(input24, 1))
        softmax34 = Softmax()(tf.expand_dims(input34, 1))
        softmax44 = Softmax()(tf.expand_dims(input44, 1))
        softmax54 = Softmax()(tf.expand_dims(input54, 1))

        preffnn14 = tf.compat.v1.matmul(tf.concat([self.a24, self.a34, self.a44, self.a54], 1), softmax14) + self.a14
        preffnn24 = tf.compat.v1.matmul(tf.concat([self.a14, self.a34, self.a44, self.a54], 1), softmax24) + self.a24
        preffnn34 = tf.compat.v1.matmul(tf.concat([self.a14, self.a24, self.a44, self.a54], 1), softmax34) + self.a34
        preffnn44 = tf.compat.v1.matmul(tf.concat([self.a14, self.a24, self.a34, self.a54], 1), softmax44) + self.a44
        preffnn54 = tf.compat.v1.matmul(tf.concat([self.a14, self.a24, self.a34, self.a44], 1), softmax54) + self.a54

        self.x125 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a15, self.a25))
        self.x135 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a15, self.a35))
        self.x145 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a15, self.a45))
        self.x155 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a15, self.a55))
        input15 = tf.stack([self.x125, self.x135, self.x145, self.x155])

        self.x215 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a25, self.a15))
        self.x235 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a25, self.a35))
        self.x245 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a25, self.a45))
        self.x255 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a25, self.a55))
        input25 = tf.stack([self.x215, self.x235, self.x245, self.x255])

        self.x315 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a35, self.a15))
        self.x325 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a35, self.a25))
        self.x345 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a35, self.a45))
        self.x355 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a35, self.a55))
        input35 = tf.stack([self.x315, self.x325, self.x345, self.x355])

        self.x415 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a45, self.a15))
        self.x425 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a45, self.a25))
        self.x435 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a45, self.a35))
        self.x455 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a45, self.a55))
        input45 = tf.stack([self.x415, self.x425, self.x435, self.x455])

        self.x515 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a55, self.a15))
        self.x525 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a55, self.a25))
        self.x535 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a55, self.a35))
        self.x545 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.a55, self.a45))
        input55 = tf.stack([self.x515, self.x525, self.x535, self.x545])

        softmax15 = Softmax()(tf.expand_dims(input15, 1))
        softmax25 = Softmax()(tf.expand_dims(input25, 1))
        softmax35 = Softmax()(tf.expand_dims(input35, 1))
        softmax45 = Softmax()(tf.expand_dims(input45, 1))
        softmax55 = Softmax()(tf.expand_dims(input55, 1))

        preffnn15 = tf.compat.v1.matmul(tf.concat([self.a25, self.a35, self.a45, self.a55], 1), softmax15) + self.a15
        preffnn25 = tf.compat.v1.matmul(tf.concat([self.a15, self.a35, self.a45, self.a55], 1), softmax25) + self.a25
        preffnn35 = tf.compat.v1.matmul(tf.concat([self.a15, self.a25, self.a45, self.a55], 1), softmax35) + self.a35
        preffnn45 = tf.compat.v1.matmul(tf.concat([self.a15, self.a25, self.a35, self.a55], 1), softmax45) + self.a45
        preffnn55 = tf.compat.v1.matmul(tf.concat([self.a15, self.a25, self.a35, self.a45], 1), softmax55) + self.a55

        preffnnmain1 = tf.concat([preffnn11, preffnn12, preffnn13, preffnn14, preffnn15], 1)
        preffnnmain2 = tf.concat([preffnn21, preffnn22, preffnn23, preffnn24, preffnn25], 1)
        preffnnmain3 = tf.concat([preffnn31, preffnn32, preffnn33, preffnn34, preffnn35], 1)
        preffnnmain4 = tf.concat([preffnn41, preffnn42, preffnn43, preffnn44, preffnn45], 1)
        preffnnmain5 = tf.concat([preffnn51, preffnn52, preffnn53, preffnn54, preffnn55], 1)

        # position-wise feed-forward networks

        preffnnmain1 = Flatten()(tf.expand_dims(preffnnmain1, 0))
        ffnn1 = Dense(10, input_shape=(10,), activation='relu')(preffnnmain1)
        ffnn1 = Dense(10)(ffnn1)
        self.ffnn1 = BatchNormalization()(ffnn1)

        preffnnmain2 = Flatten()(tf.expand_dims(preffnnmain2, 0))
        ffnn2 = Dense(10, input_shape=(10,), activation='relu')(preffnnmain2)
        ffnn2 = Dense(10)(ffnn2)
        self.ffnn2 = BatchNormalization()(ffnn2)

        preffnnmain3 = Flatten()(tf.expand_dims(preffnnmain3, 0))
        ffnn3 = Dense(10, input_shape=(10,), activation='relu')(preffnnmain3)
        ffnn3 = Dense(10)(ffnn3)
        self.ffnn3 = BatchNormalization()(ffnn3)

        preffnnmain4 = Flatten()(tf.expand_dims(preffnnmain4, 0))  # shape=(1,10)
        ffnn4 = Dense(10, input_shape=(10,), activation='relu')(preffnnmain4)
        ffnn4 = Dense(10)(ffnn4)
        self.ffnn4 = BatchNormalization()(ffnn4)

        preffnnmain5 = Flatten()(tf.expand_dims(preffnnmain5, 0))
        ffnn5 = Dense(10, activation='relu')(preffnnmain5)
        ffnn5 = Dense(10)(ffnn5)
        self.ffnn5 = BatchNormalization()(ffnn5)

        [self.ffnn11, self.ffnn12, self.ffnn13, self.ffnn14, self.ffnn15, self.ffnn16, self.ffnn17, self.ffnn18,
         self.ffnn19, self.ffnn110] = tf.unstack(self.ffnn1, num=10, axis=1, name='unstack')
        self.ffnn11f = tf.reshape(tf.stack([self.ffnn11, self.ffnn12], axis=1), shape=[2, 1])
        self.ffnn12f = tf.reshape(tf.stack([self.ffnn13, self.ffnn14], axis=1), shape=[2, 1])
        self.ffnn13f = tf.reshape(tf.stack([self.ffnn15, self.ffnn16], axis=1), shape=[2, 1])
        self.ffnn14f = tf.reshape(tf.stack([self.ffnn17, self.ffnn18], axis=1), shape=[2, 1])
        self.ffnn15f = tf.reshape(tf.stack([self.ffnn19, self.ffnn110], axis=1), shape=[2, 1])

        [self.ffnn21, self.ffnn22, self.ffnn23, self.ffnn24, self.ffnn25, self.ffnn26, self.ffnn27, self.ffnn28,
         self.ffnn29, self.ffnn210] = tf.unstack(self.ffnn2, num=10, axis=1, name='unstack')
        self.ffnn21f = tf.reshape(tf.stack([self.ffnn21, self.ffnn22], axis=1), shape=[2, 1])
        self.ffnn22f = tf.reshape(tf.stack([self.ffnn23, self.ffnn24], axis=1), shape=[2, 1])
        self.ffnn23f = tf.reshape(tf.stack([self.ffnn25, self.ffnn26], axis=1), shape=[2, 1])
        self.ffnn24f = tf.reshape(tf.stack([self.ffnn27, self.ffnn28], axis=1), shape=[2, 1])
        self.ffnn25f = tf.reshape(tf.stack([self.ffnn29, self.ffnn210], axis=1), shape=[2, 1])

        [self.ffnn31, self.ffnn32, self.ffnn33, self.ffnn34, self.ffnn35, self.ffnn36, self.ffnn37, self.ffnn38,
         self.ffnn39, self.ffnn310] = tf.unstack(self.ffnn3, num=10, axis=1, name='unstack')
        self.ffnn31f = tf.reshape(tf.stack([self.ffnn31, self.ffnn32], axis=1), shape=[2, 1])
        self.ffnn32f = tf.reshape(tf.stack([self.ffnn33, self.ffnn34], axis=1), shape=[2, 1])
        self.ffnn33f = tf.reshape(tf.stack([self.ffnn35, self.ffnn36], axis=1), shape=[2, 1])
        self.ffnn34f = tf.reshape(tf.stack([self.ffnn37, self.ffnn38], axis=1), shape=[2, 1])
        self.ffnn35f = tf.reshape(tf.stack([self.ffnn39, self.ffnn310], axis=1), shape=[2, 1])

        [self.ffnn41, self.ffnn42, self.ffnn43, self.ffnn44, self.ffnn45, self.ffnn46, self.ffnn47, self.ffnn48,
         self.ffnn49, self.ffnn410] = tf.unstack(self.ffnn4, num=10, axis=1, name='unstack')
        self.ffnn41f = tf.reshape(tf.stack([self.ffnn41, self.ffnn42], axis=1), shape=[2, 1])
        self.ffnn42f = tf.reshape(tf.stack([self.ffnn43, self.ffnn44], axis=1), shape=[2, 1])
        self.ffnn43f = tf.reshape(tf.stack([self.ffnn45, self.ffnn46], axis=1), shape=[2, 1])
        self.ffnn44f = tf.reshape(tf.stack([self.ffnn47, self.ffnn48], axis=1), shape=[2, 1])
        self.ffnn45f = tf.reshape(tf.stack([self.ffnn49, self.ffnn410], axis=1), shape=[2, 1])

        [self.ffnn51, self.ffnn52, self.ffnn53, self.ffnn54, self.ffnn55, self.ffnn56, self.ffnn57, self.ffnn58,
         self.ffnn59, self.ffnn510] = tf.unstack(self.ffnn5, num=10, axis=1, name='unstack')
        self.ffnn51f = tf.reshape(tf.stack([self.ffnn51, self.ffnn52], axis=1), shape=[2, 1])
        self.ffnn52f = tf.reshape(tf.stack([self.ffnn53, self.ffnn54], axis=1), shape=[2, 1])
        self.ffnn53f = tf.reshape(tf.stack([self.ffnn55, self.ffnn56], axis=1), shape=[2, 1])
        self.ffnn54f = tf.reshape(tf.stack([self.ffnn57, self.ffnn58], axis=1), shape=[2, 1])
        self.ffnn55f = tf.reshape(tf.stack([self.ffnn59, self.ffnn510], axis=1), shape=[2, 1])

    def decoder(self):
        self.decoded = tf.compat.v1.placeholder(shape=[2, 40], dtype=tf.dtypes.float32)

        self.first = tf.Variable([[10], [10]], shape=(2, 1), dtype=tf.float32)

        # shape=(2,5)
        encoder_output1 = tf.concat([self.ffnn11f, self.ffnn12f, self.ffnn13f, self.ffnn14f, self.ffnn15f], 1)
        encoder_output1 = tf.reshape(encoder_output1, shape=(5, 2))

        encoder_output2 = tf.concat([self.ffnn21f, self.ffnn22f, self.ffnn23f, self.ffnn24f, self.ffnn25f], 1)
        encoder_output2 = tf.reshape(encoder_output2, shape=(5, 2))

        encoder_output3 = tf.concat([self.ffnn31f, self.ffnn32f, self.ffnn33f, self.ffnn34f, self.ffnn35f], 1)
        encoder_output3 = tf.reshape(encoder_output3, shape=(5, 2))

        encoder_output4 = tf.concat([self.ffnn41f, self.ffnn42f, self.ffnn43f, self.ffnn44f, self.ffnn45f], 1)
        encoder_output4 = tf.reshape(encoder_output4, shape=(5, 2))

        encoder_output5 = tf.concat([self.ffnn51f, self.ffnn52f, self.ffnn53f, self.ffnn54f, self.ffnn55f], 1)
        encoder_output5 = tf.reshape(encoder_output5, shape=(5, 2))

        # first
        output_list = [self.first]

        prenn1 = self.first

        # print(prenn.get_shape()) #shape=(2,1)

        z1 = tf.compat.v1.matmul(encoder_output1, prenn1)
        # print(z.get_shape()) #shape=(5,1)

        softmaxed21 = Softmax(name='edsoft')(z1)  # shape=(5,1)
        softmaxed21 = tf.reshape(softmaxed21, shape=(1, 5))
        prenn1 = tf.reshape(prenn1, shape=(1, 2))
        prenn21 = tf.compat.v1.matmul(softmaxed21, encoder_output1) + prenn1  # shape=(1,2)

        ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l0')(prenn21)  # shape=(1,2)
        ffnn1 = Dense(2, name='m0')(ffnn1)  # shape=(1,2)

        output_list.append(tf.reshape(ffnn1, shape=(2, 1)))
        # print(np.shape(output_list))

        for j in range(1, 24):
            x = []
            y = []

            for k in range(0, 24):

                if k < j:
                    x.append(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(output_list[k], output_list[j])))
                    y.append(output_list[k])
                else:
                    x.append(0)
                    y.append([[0], [0]])

            x1 = tf.convert_to_tensor(x)
            y1 = tf.convert_to_tensor(y)
            # print(x.get_shape())
            # print(y.get_shape())

            softmaxed1 = Softmax(name='decatt')(tf.expand_dims(x1, 0))
            # print(softmaxed.get_shape())
            y1 = tf.reshape(y1, shape=(-1, 2))
            # print(y.get_shape())
            prenn1 = tf.reshape(tf.compat.v1.matmul(softmaxed1, y1), shape=(2, 1)) + output_list[j]
            # print(prenn.get_shape())

            z1 = tf.compat.v1.matmul(encoder_output1, prenn1)
            # print(z.get_shape()) #shape=(5,1)

            softmaxed21 = Softmax(name='edsoft')(z1)  # shape=(5,1)
            softmaxed21 = tf.reshape(softmaxed21, shape=(1, 5))
            prenn1 = tf.reshape(prenn1, shape=(1, 2))
            prenn21 = tf.compat.v1.matmul(softmaxed21, encoder_output1) + prenn1  # shape=(1,2)

            if j == 1:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l1')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m1')(ffnn1)  # shape=(1,2)
            elif j == 2:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l2')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m2')(ffnn1)  # shape=(1,2)
            elif j == 3:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l3')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m3')(ffnn1)  # shape=(1,2)
            elif j == 4:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l4')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m4')(ffnn1)  # shape=(1,2)
            elif j == 5:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l5')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m5')(ffnn1)  # shape=(1,2)
            elif j == 6:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l6')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m6')(ffnn1)  # shape=(1,2)
            elif j == 7:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l7')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m7')(ffnn1)  # shape=(1,2)
            elif j == 8:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l8')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m8')(ffnn1)  # shape=(1,2)
            elif j == 9:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l9')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m9')(ffnn1)  # shape=(1,2)
            elif j == 10:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l10')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m10')(ffnn1)  # shape=(1,2)
            elif j == 11:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l11')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m11')(ffnn1)  # shape=(1,2)
            elif j == 12:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l12')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m12')(ffnn1)  # shape=(1,2)
            elif j == 13:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l13')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m13')(ffnn1)  # shape=(1,2)
            elif j == 14:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l14')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m14')(ffnn1)  # shape=(1,2)
            elif j == 15:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l15')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m15')(ffnn1)  # shape=(1,2)
            elif j == 16:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l16')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m16')(ffnn1)  # shape=(1,2)
            elif j == 17:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l17')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m17')(ffnn1)  # shape=(1,2)
            elif j == 18:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l18')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m18')(ffnn1)  # shape=(1,2)
            elif j == 19:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l19')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m19')(ffnn1)  # shape=(1,2)
            elif j == 20:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l20')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m20')(ffnn1)  # shape=(1,2)
            elif j == 21:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l21')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m21')(ffnn1)  # shape=(1,2)
            elif j == 22:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l22')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m22')(ffnn1)  # shape=(1,2)
            elif j == 23:
                ffnn1 = Dense(2, input_shape=(2,), activation='relu', name='l23')(prenn21)  # shape=(1,2)
                ffnn1 = Dense(2, name='m23')(ffnn1)  # shape=(1,2)

            output_list.append(tf.reshape(ffnn1, shape=(2, 1)))
            # print(np.shape(output_list))

        output_list.pop(0)
        self.decoded1 = tf.stack(output_list)
        self.decoded1 = tf.reshape(self.decoded1, shape=(2, -1))

        # fourth
        output_list = [self.first]

        prenn4 = self.first

        # print(prenn.get_shape()) #shape=(2,1)

        z4 = tf.compat.v1.matmul(encoder_output4, prenn4)
        # print(z.get_shape()) #shape=(5,1)

        softmaxed24 = Softmax(name='edsoft')(z4)  # shape=(5,1)
        softmaxed24 = tf.reshape(softmaxed24, shape=(1, 5))
        prenn4 = tf.reshape(prenn4, shape=(1, 2))
        prenn24 = tf.compat.v1.matmul(softmaxed24, encoder_output4) + prenn4  # shape=(1,2)

        ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l0')(prenn24)  # shape=(1,2)
        ffnn4 = Dense(2, name='m0')(ffnn4)  # shape=(1,2)

        output_list.append(tf.reshape(ffnn4, shape=(2, 1)))
        # print(np.shape(output_list))

        for j in range(1, 24):
            x = []
            y = []

            for k in range(0, 24):

                if k < j:
                    x.append(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(output_list[k], output_list[j])))
                    y.append(output_list[k])
                else:
                    x.append(0)
                    y.append([[0], [0]])

            x4 = tf.convert_to_tensor(x)
            y4 = tf.convert_to_tensor(y)
            # print(x.get_shape())
            # print(y.get_shape())

            softmaxed4 = Softmax(name='decatt')(tf.expand_dims(x4, 0))
            # print(softmaxed.get_shape())
            y4 = tf.reshape(y4, shape=(-1, 2))
            # print(y.get_shape())
            prenn4 = tf.reshape(tf.compat.v1.matmul(softmaxed4, y4), shape=(2, 1)) + output_list[j]
            # print(prenn.get_shape())

            z4 = tf.compat.v1.matmul(encoder_output4, prenn4)
            # print(z.get_shape()) #shape=(5,1)

            softmaxed24 = Softmax(name='edsoft')(z4)  # shape=(5,1)
            softmaxed24 = tf.reshape(softmaxed24, shape=(1, 5))
            prenn4 = tf.reshape(prenn4, shape=(1, 2))
            prenn24 = tf.compat.v1.matmul(softmaxed24, encoder_output4) + prenn4  # shape=(1,2)

            if j == 1:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l1')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m1')(ffnn4)  # shape=(1,2)
            elif j == 2:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l2')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m2')(ffnn4)  # shape=(1,2)
            elif j == 3:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l3')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m3')(ffnn4)  # shape=(1,2)
            elif j == 4:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l4')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m4')(ffnn4)  # shape=(1,2)
            elif j == 5:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l5')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m5')(ffnn4)  # shape=(1,2)
            elif j == 6:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l6')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m6')(ffnn4)  # shape=(1,2)
            elif j == 7:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l7')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m7')(ffnn4)  # shape=(1,2)
            elif j == 8:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l8')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m8')(ffnn4)  # shape=(1,2)
            elif j == 9:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l9')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m9')(ffnn4)  # shape=(1,2)
            elif j == 10:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l10')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m10')(ffnn4)  # shape=(1,2)
            elif j == 11:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l11')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m11')(ffnn4)  # shape=(1,2)
            elif j == 12:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l12')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m12')(ffnn4)  # shape=(1,2)
            elif j == 13:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l13')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m13')(ffnn4)  # shape=(1,2)
            elif j == 14:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l14')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m14')(ffnn4)  # shape=(1,2)
            elif j == 15:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l15')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m15')(ffnn4)  # shape=(1,2)
            elif j == 16:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l16')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m16')(ffnn4)  # shape=(1,2)
            elif j == 17:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l17')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m17')(ffnn4)  # shape=(1,2)
            elif j == 18:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l18')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m18')(ffnn4)  # shape=(1,2)
            elif j == 19:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l19')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m19')(ffnn4)  # shape=(1,2)
            elif j == 20:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l20')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m20')(ffnn4)  # shape=(1,2)
            elif j == 21:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l21')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m21')(ffnn4)  # shape=(1,2)
            elif j == 22:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l22')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m22')(ffnn4)  # shape=(1,2)
            elif j == 23:
                ffnn4 = Dense(2, input_shape=(2,), activation='relu', name='l23')(prenn24)  # shape=(1,2)
                ffnn4 = Dense(2, name='m23')(ffnn4)  # shape=(1,2)

            output_list.append(tf.reshape(ffnn4, shape=(2, 1)))
            # print(np.shape(output_list))

        output_list.pop(0)
        self.decoded4 = tf.stack(output_list)
        self.decoded4 = tf.reshape(self.decoded4, shape=(2, -1))

        # fifth
        output_list = [self.first]

        prenn5 = self.first

        # print(prenn.get_shape()) #shape=(2,1)

        z5 = tf.compat.v1.matmul(encoder_output5, prenn5)
        # print(z.get_shape()) #shape=(5,1)

        softmaxed25 = Softmax(name='edsoft')(z5)  # shape=(5,1)
        softmaxed25 = tf.reshape(softmaxed25, shape=(1, 5))
        prenn5 = tf.reshape(prenn5, shape=(1, 2))
        prenn25 = tf.compat.v1.matmul(softmaxed25, encoder_output5) + prenn5  # shape=(1,2)

        ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l0')(prenn25)  # shape=(1,2)
        ffnn5 = Dense(2, name='m0')(ffnn5)  # shape=(1,2)

        output_list.append(tf.reshape(ffnn5, shape=(2, 1)))
        # print(np.shape(output_list))

        for j in range(1, 24):
            x = []
            y = []

            for k in range(0, 24):

                if k < j:
                    x.append(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(output_list[k], output_list[j])))
                    y.append(output_list[k])
                else:
                    x.append(0)
                    y.append([[0], [0]])

            x5 = tf.convert_to_tensor(x)
            y5 = tf.convert_to_tensor(y)
            # print(x.get_shape())
            # print(y.get_shape())

            softmaxed5 = Softmax(name='decatt')(tf.expand_dims(x5, 0))
            # print(softmaxed.get_shape())
            y5 = tf.reshape(y5, shape=(-1, 2))
            # print(y.get_shape())
            prenn5 = tf.reshape(tf.compat.v1.matmul(softmaxed5, y5), shape=(2, 1)) + output_list[j]
            # print(prenn.get_shape())

            z5 = tf.compat.v1.matmul(encoder_output5, prenn5)
            # print(z.get_shape()) #shape=(5,1)

            softmaxed25 = Softmax(name='edsoft')(z5)  # shape=(5,1)
            softmaxed25 = tf.reshape(softmaxed25, shape=(1, 5))
            prenn5 = tf.reshape(prenn5, shape=(1, 2))
            prenn25 = tf.compat.v1.matmul(softmaxed25, encoder_output5) + prenn5  # shape=(1,2)

            if j == 1:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l1')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m1')(ffnn5)  # shape=(1,2)
            elif j == 2:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l2')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m2')(ffnn5)  # shape=(1,2)
            elif j == 3:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l3')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m3')(ffnn5)  # shape=(1,2)
            elif j == 4:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l4')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m4')(ffnn5)  # shape=(1,2)
            elif j == 5:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l5')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m5')(ffnn5)  # shape=(1,2)
            elif j == 6:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l6')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m6')(ffnn5)  # shape=(1,2)
            elif j == 7:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l7')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m7')(ffnn5)  # shape=(1,2)
            elif j == 8:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l8')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m8')(ffnn5)  # shape=(1,2)
            elif j == 9:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l9')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m9')(ffnn5)  # shape=(1,2)
            elif j == 10:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l10')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m10')(ffnn5)  # shape=(1,2)
            elif j == 11:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l11')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m11')(ffnn5)  # shape=(1,2)
            elif j == 12:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l12')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m12')(ffnn5)  # shape=(1,2)
            elif j == 13:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l13')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m13')(ffnn5)  # shape=(1,2)
            elif j == 14:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l14')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m14')(ffnn5)  # shape=(1,2)
            elif j == 15:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l15')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m15')(ffnn5)  # shape=(1,2)
            elif j == 16:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l16')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m16')(ffnn5)  # shape=(1,2)
            elif j == 17:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l17')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m17')(ffnn5)  # shape=(1,2)
            elif j == 18:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l18')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m18')(ffnn5)  # shape=(1,2)
            elif j == 19:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l19')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m19')(ffnn5)  # shape=(1,2)
            elif j == 20:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l20')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m20')(ffnn5)  # shape=(1,2)
            elif j == 21:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l21')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m21')(ffnn5)  # shape=(1,2)
            elif j == 22:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l22')(prenn25)  # shape=(1,2)
                ffnn5 = Dense(2, name='m22')(ffnn5)  # shape=(1,2)
            elif j == 23:
                ffnn5 = Dense(2, input_shape=(2,), activation='relu', name='l23')(prenn21)  # shape=(1,2)
                ffnn5 = Dense(2, name='m23')(ffnn5)  # shape=(1,2)

            output_list.append(tf.reshape(ffnn5, shape=(2, 1)))
            # print(np.shape(output_list))

        output_list.pop(0)
        self.decoded5 = tf.stack(output_list)
        self.decoded5 = tf.reshape(self.decoded5, shape=(2, -1))

        # second
        output_list = [self.first]

        prenn2 = self.first

        # print(prenn.get_shape()) #shape=(2,1)

        z2 = tf.compat.v1.matmul(encoder_output2, prenn2)
        # print(z.get_shape()) #shape=(5,1)

        softmaxed22 = Softmax(name='edsoft')(z2)  # shape=(5,1)
        softmaxed22 = tf.reshape(softmaxed22, shape=(1, 5))
        prenn2 = tf.reshape(prenn2, shape=(1, 2))
        prenn22 = tf.compat.v1.matmul(softmaxed21, encoder_output1) + prenn2  # shape=(1,2)

        ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l0')(prenn22)  # shape=(1,2)
        ffnn2 = Dense(2, name='m0')(ffnn2)  # shape=(1,2)

        output_list.append(tf.reshape(ffnn2, shape=(2, 1)))
        # print(np.shape(output_list))

        for j in range(1, 24):
            x = []
            y = []

            for k in range(0, 24):

                if k < j:
                    x.append(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(output_list[k], output_list[j])))
                    y.append(output_list[k])
                else:
                    x.append(0)
                    y.append([[0], [0]])

            x2 = tf.convert_to_tensor(x)
            y2 = tf.convert_to_tensor(y)
            # print(x.get_shape())
            # print(y.get_shape())

            softmaxed2 = Softmax(name='decatt')(tf.expand_dims(x2, 0))
            # print(softmaxed.get_shape())
            y2 = tf.reshape(y2, shape=(-1, 2))
            # print(y.get_shape())
            prenn2 = tf.reshape(tf.compat.v1.matmul(softmaxed2, y2), shape=(2, 1)) + output_list[j]
            # print(prenn.get_shape())

            z2 = tf.compat.v1.matmul(encoder_output2, prenn2)
            # print(z.get_shape()) #shape=(5,1)

            softmaxed22 = Softmax(name='edsoft')(z2)  # shape=(5,1)
            softmaxed22 = tf.reshape(softmaxed22, shape=(1, 5))
            prenn2 = tf.reshape(prenn2, shape=(1, 2))
            prenn22 = tf.compat.v1.matmul(softmaxed22, encoder_output2) + prenn2  # shape=(1,2)

            if j == 1:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l1')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m1')(ffnn2)  # shape=(1,2)
            elif j == 2:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l2')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m2')(ffnn2)  # shape=(1,2)
            elif j == 3:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l3')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m3')(ffnn2)  # shape=(1,2)
            elif j == 4:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l4')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m4')(ffnn2)  # shape=(1,2)
            elif j == 5:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l5')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m5')(ffnn2)  # shape=(1,2)
            elif j == 6:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l6')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m6')(ffnn2)  # shape=(1,2)
            elif j == 7:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l7')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m7')(ffnn2)  # shape=(1,2)
            elif j == 8:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l8')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m8')(ffnn2)  # shape=(1,2)
            elif j == 9:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l9')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m9')(ffnn2)  # shape=(1,2)
            elif j == 10:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l10')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m10')(ffnn2)  # shape=(1,2)
            elif j == 11:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l11')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m11')(ffnn2)  # shape=(1,2)
            elif j == 12:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l12')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m12')(ffnn2)  # shape=(1,2)
            elif j == 13:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l13')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m13')(ffnn2)  # shape=(1,2)
            elif j == 14:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l14')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m14')(ffnn2)  # shape=(1,2)
            elif j == 15:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l15')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m15')(ffnn2)  # shape=(1,2)
            elif j == 16:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l16')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m16')(ffnn2)  # shape=(1,2)
            elif j == 17:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l17')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m17')(ffnn2)  # shape=(1,2)
            elif j == 18:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l18')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m18')(ffnn2)  # shape=(1,2)
            elif j == 19:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l19')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m19')(ffnn2)  # shape=(1,2)
            elif j == 20:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l20')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m20')(ffnn2)  # shape=(1,2)
            elif j == 21:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l21')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m21')(ffnn2)  # shape=(1,2)
            elif j == 22:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l22')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m22')(ffnn2)  # shape=(1,2)
            elif j == 23:
                ffnn2 = Dense(2, input_shape=(2,), activation='relu', name='l23')(prenn22)  # shape=(1,2)
                ffnn2 = Dense(2, name='m23')(ffnn2)  # shape=(1,2)

            output_list.append(tf.reshape(ffnn2, shape=(2, 1)))
            # print(np.shape(output_list))

        output_list.pop(0)
        self.decoded2 = tf.stack(output_list)
        self.decoded2 = tf.reshape(self.decoded2, shape=(2, -1))

        # third
        output_list = [self.first]

        prenn3 = self.first

        # print(prenn.get_shape()) #shape=(2,1)

        z3 = tf.compat.v1.matmul(encoder_output3, prenn3)
        # print(z.get_shape()) #shape=(5,1)

        softmaxed23 = Softmax(name='edsoft')(z3)  # shape=(5,1)
        softmaxed23 = tf.reshape(softmaxed23, shape=(1, 5))
        prenn3 = tf.reshape(prenn3, shape=(1, 2))
        prenn23 = tf.compat.v1.matmul(softmaxed23, encoder_output3) + prenn3  # shape=(1,2)

        ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l0')(prenn23)  # shape=(1,2)
        ffnn3 = Dense(2, name='m0')(ffnn3)  # shape=(1,2)

        output_list.append(tf.reshape(ffnn3, shape=(2, 1)))
        # print(np.shape(output_list))

        for j in range(1, 24):
            x = []
            y = []

            for k in range(0, 24):

                if k < j:
                    x.append(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(output_list[k], output_list[j])))
                    y.append(output_list[k])
                else:
                    x.append(0)
                    y.append([[0], [0]])

            x3 = tf.convert_to_tensor(x)
            y3 = tf.convert_to_tensor(y)
            # print(x.get_shape())
            # print(y.get_shape())

            softmaxed3 = Softmax(name='decatt')(tf.expand_dims(x3, 0))
            # print(softmaxed.get_shape())
            y3 = tf.reshape(y3, shape=(-1, 2))
            # print(y.get_shape())
            prenn3 = tf.reshape(tf.compat.v1.matmul(softmaxed3, y3), shape=(2, 1)) + output_list[j]
            # print(prenn.get_shape())

            z3 = tf.compat.v1.matmul(encoder_output3, prenn3)
            # print(z.get_shape()) #shape=(5,1)

            softmaxed23 = Softmax(name='edsoft')(z3)  # shape=(5,1)
            softmaxed23 = tf.reshape(softmaxed23, shape=(1, 5))
            prenn3 = tf.reshape(prenn3, shape=(1, 2))
            prenn23 = tf.compat.v1.matmul(softmaxed23, encoder_output3) + prenn3  # shape=(1,2)

            if j == 1:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l1')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m1')(ffnn3)  # shape=(1,2)
            elif j == 2:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l2')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m2')(ffnn3)  # shape=(1,2)
            elif j == 3:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l3')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m3')(ffnn3)  # shape=(1,2)
            elif j == 4:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l4')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m4')(ffnn3)  # shape=(1,2)
            elif j == 5:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l5')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m5')(ffnn3)  # shape=(1,2)
            elif j == 6:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l6')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m6')(ffnn3)  # shape=(1,2)
            elif j == 7:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l7')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m7')(ffnn3)  # shape=(1,2)
            elif j == 8:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l8')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m8')(ffnn3)  # shape=(1,2)
            elif j == 9:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l9')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m9')(ffnn3)  # shape=(1,2)
            elif j == 10:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l10')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m10')(ffnn3)  # shape=(1,2)
            elif j == 11:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l11')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m11')(ffnn3)  # shape=(1,2)
            elif j == 12:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l12')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m12')(ffnn3)  # shape=(1,2)
            elif j == 13:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l13')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m13')(ffnn3)  # shape=(1,2)
            elif j == 14:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l14')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m14')(ffnn3)  # shape=(1,2)
            elif j == 15:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l15')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m15')(ffnn3)  # shape=(1,2)
            elif j == 16:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l16')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m16')(ffnn3)  # shape=(1,2)
            elif j == 17:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l17')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m17')(ffnn3)  # shape=(1,2)
            elif j == 18:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l18')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m18')(ffnn3)  # shape=(1,2)
            elif j == 19:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l19')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m19')(ffnn3)  # shape=(1,2)
            elif j == 20:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l20')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m20')(ffnn3)  # shape=(1,2)
            elif j == 21:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l21')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m21')(ffnn3)  # shape=(1,2)
            elif j == 22:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l22')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m22')(ffnn3)  # shape=(1,2)
            elif j == 23:
                ffnn3 = Dense(2, input_shape=(2,), activation='relu', name='l23')(prenn23)  # shape=(1,2)
                ffnn3 = Dense(2, name='m23')(ffnn3)  # shape=(1,2)

            output_list.append(tf.reshape(ffnn3, shape=(2, 1)))
            # print(np.shape(output_list))

        output_list.pop(0)
        self.decoded3 = tf.stack(output_list)
        self.decoded3 = tf.reshape(self.decoded3, shape=(2, -1))

    def train(self):

        self.encoder()
        self.decoder()

        train_x = pd.read_csv('train_x.csv')
        train_y = pd.read_csv('train_y.csv')
        train_chars = pd.read_csv('train_chars.csv')

        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        train_chars = train_chars.to_numpy()

        train_x = np.reshape(train_x, (-1, 1))
        train_y = np.reshape(train_y, (-1, 1))
        train_chars = np.reshape(train_chars, (-1,))

        # print(np.shape(train_x))
        # print(np.shape(train_y))
        # print(np.shape(train_chars))

        y_tensor = tf.compat.v1.placeholder(shape=[2, 24], dtype=tf.float32)
        learning_rate = 0.00000001

        loss1 = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.decoded1 - y_tensor))
        # self.optimizer1 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
        self.optimizer1 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss1)
        loss2 = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.decoded2 - y_tensor))
        # self.optimizer2 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss2)
        self.optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss2)
        loss3 = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.decoded3 - y_tensor))
        # self.optimizer3 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss3)
        self.optimizer3 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss3)
        loss4 = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.decoded4 - y_tensor))
        # self.optimizer4 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss4)
        self.optimizer4 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss4)
        loss5 = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.decoded5 - y_tensor))
        # self.optimizer5 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss5)
        self.optimizer5 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss5)

        num_epochs = 1

        with tf.compat.v1.Session() as tfs:
            tfs.run(tf.compat.v1.global_variables_initializer())
            for num in range(num_epochs):
                # print(num)
                for i in range(len(train_chars)):

                    if i==int(self.train_length):
                        print("Model trained on", self.train_length, "characters")
                        break

                    print(i)

                    if i == 0:
                        hehe = np.concatenate([train_x[i * 24:(i + 1) * 24], train_y[i * 24:(i + 1) * 24]], axis=1)
                        tfs.run(self.optimizer1,
                                feed_dict={self.a1: [[ord(train_chars[i])], [int(train_chars[i].isupper())]],
                                           self.a2: [[ord(train_chars[i + 1])], [int(train_chars[i + 1].isupper())]],
                                           self.a3: [[ord(train_chars[i + 2])], [int(train_chars[i + 2].isupper())]],
                                           self.a4: [[ord(train_chars[i + 3])], [int(train_chars[i + 3].isupper())]],
                                           self.a5: [[ord(train_chars[i + 4])], [int(train_chars[i + 4].isupper())]],
                                           y_tensor: np.transpose(hehe)})
                    elif i == 1:
                        hehe = np.concatenate([train_x[i * 24:(i + 1) * 24], train_y[i * 24:(i + 1) * 24]], axis=1)
                        tfs.run(self.optimizer2,
                                feed_dict={self.a1: [[ord(train_chars[i - 1])], [int(train_chars[i - 1].isupper())]],
                                           self.a2: [[ord(train_chars[i])], [int(train_chars[i].isupper())]],
                                           self.a3: [[ord(train_chars[i + 1])], [int(train_chars[i + 1].isupper())]],
                                           self.a4: [[ord(train_chars[i + 2])], [int(train_chars[i + 2].isupper())]],
                                           self.a5: [[ord(train_chars[i + 3])], [int(train_chars[i + 3].isupper())]],
                                           y_tensor: np.transpose(hehe)})
                    elif i == len(train_chars) - 2:
                        hehe = np.concatenate([train_x[i * 24:(i + 1) * 24], train_y[i * 24:(i + 1) * 24]], axis=1)
                        tfs.run(self.optimizer4,
                                feed_dict={self.a1: [[ord(train_chars[i - 3])], [int(train_chars[i - 3].isupper())]],
                                           self.a2: [[ord(train_chars[i - 2])], [int(train_chars[i - 2].isupper())]],
                                           self.a3: [[ord(train_chars[i - 1])], [int(train_chars[i - 1].isupper())]],
                                           self.a4: [[ord(train_chars[i])], [int(train_chars[i].isupper())]],
                                           self.a5: [[ord(train_chars[i + 1])], [int(train_chars[i + 1].isupper())]],
                                           y_tensor: np.transpose(hehe)})
                    elif i == len(train_chars) - 1:
                        hehe = np.concatenate([train_x[i * 24:(i + 1) * 24], train_y[i * 24:(i + 1) * 24]], axis=1)
                        tfs.run(self.optimizer5,
                                feed_dict={self.a1: [[ord(train_chars[i - 4])], [int(train_chars[i - 4].isupper())]],
                                           self.a2: [[ord(train_chars[i - 3])], [int(train_chars[i - 3].isupper())]],
                                           self.a3: [[ord(train_chars[i - 2])], [int(train_chars[i - 2].isupper())]],
                                           self.a4: [[ord(train_chars[i - 1])], [int(train_chars[i - 1].isupper())]],
                                           self.a5: [[ord(train_chars[i])], [int(train_chars[i].isupper())]],
                                           y_tensor: np.transpose(hehe)})
                    else:
                        hehe = np.concatenate([train_x[i * 24:(i + 1) * 24], train_y[i * 24:(i + 1) * 24]], axis=1)
                        tfs.run(self.optimizer3,
                                          feed_dict={
                                              self.a1: [[ord(train_chars[i - 2])], [int(train_chars[i - 2].isupper())]],
                                              self.a2: [[ord(train_chars[i - 1])], [int(train_chars[i - 1].isupper())]],
                                              self.a3: [[ord(train_chars[i])], [int(train_chars[i].isupper())]],
                                              self.a4: [[ord(train_chars[i + 1])], [int(train_chars[i + 1].isupper())]],
                                              self.a5: [[ord(train_chars[i + 2])], [int(train_chars[i + 2].isupper())]],
                                              y_tensor: np.transpose(hehe)})

    def test(self, string):

        predictions = []

        self.train()

        with tf.compat.v1.Session() as tfs:
            tfs.run(tf.compat.v1.global_variables_initializer())
            for i in range(len(string)):
                if i == 0:
                    predictions.append(
                        tfs.run(self.decoded1, feed_dict={self.a1: [[ord(string[i])], [int(string[i].isupper())]],
                                                          self.a2: [[ord(string[i + 1])],
                                                                    [int(string[i + 1].isupper())]],
                                                          self.a3: [[ord(string[i + 2])],
                                                                    [int(string[i + 2].isupper())]],
                                                          self.a4: [[ord(string[i + 3])],
                                                                    [int(string[i + 3].isupper())]],
                                                          self.a5: [[ord(string[i + 4])],
                                                                    [int(string[i + 4].isupper())]],
                                                          }))
                elif i == 1:
                    predictions.append(
                        tfs.run(self.decoded2, feed_dict={self.a1: [[ord(string[i])], [int(string[i].isupper())]],
                                                          self.a2: [[ord(string[i + 1])],
                                                                    [int(string[i + 1].isupper())]],
                                                          self.a3: [[ord(string[i + 2])],
                                                                    [int(string[i + 2].isupper())]],
                                                          self.a4: [[ord(string[i + 3])],
                                                                    [int(string[i + 3].isupper())]],
                                                          self.a5: [[ord(string[i + 4])],
                                                                    [int(string[i + 4].isupper())]],
                                                          }))
                elif i == len(string) - 2:
                    predictions.append(
                        tfs.run(self.decoded4,
                                feed_dict={self.a1: [[ord(string[i - 3])], [int(string[i - 3].isupper())]],
                                           self.a2: [[ord(string[i - 2])],
                                                     [int(string[i - 2].isupper())]],
                                           self.a3: [[ord(string[i - 1])],
                                                     [int(string[i - 1].isupper())]],
                                           self.a4: [[ord(string[i])],
                                                     [int(string[i].isupper())]],
                                           self.a5: [[ord(string[i + 1])],
                                                     [int(string[i + 1].isupper())]],
                                           }))
                elif i == len(string) - 1:
                    predictions.append(
                        tfs.run(self.decoded5,
                                feed_dict={self.a1: [[ord(string[i - 4])], [int(string[i - 4].isupper())]],
                                           self.a2: [[ord(string[i - 3])],
                                                     [int(string[i - 3].isupper())]],
                                           self.a3: [[ord(string[i - 2])],
                                                     [int(string[i - 2].isupper())]],
                                           self.a4: [[ord(string[i - 1])],
                                                     [int(string[i - 1].isupper())]],
                                           self.a5: [[ord(string[i])],
                                                     [int(string[i].isupper())]],
                                           }))
                else:
                    predictions.append(
                        tfs.run(self.decoded3,
                                feed_dict={self.a1: [[ord(string[i - 2])], [int(string[i - 2].isupper())]],
                                           self.a2: [[ord(string[i - 1])],
                                                     [int(string[i - 1].isupper())]],
                                           self.a3: [[ord(string[i])],
                                                     [int(string[i].isupper())]],
                                           self.a4: [[ord(string[i + 1])],
                                                     [int(string[i + 1].isupper())]],
                                           self.a5: [[ord(string[i + 2])],
                                                     [int(string[i + 2].isupper())]],
                                           }))

        predictions = np.reshape(predictions, (2, -1))
        predictions = pd.DataFrame(predictions)
        predictions.to_csv('points15.csv', index=False)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.experimental.output_all_intermediates(True)

    print(sys.argv[2])

    transformer = Transformer(sys.argv[2])

    transformer.test(sys.argv[1])