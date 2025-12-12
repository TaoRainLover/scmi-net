#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/2/27 19:57
# @Author : Tao Zhang
# @Email: 2637050370@qq.com

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 使用模型提取某一层的特征
layer_output = model.get_layer('某一层的名称').output
feature_extractor = tf.keras.Model(inputs=model.input, outputs=layer_output)
features = feature_extractor.predict(X)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

# 可视化 t-SNE 降维结果
plt.figure(figsize=(10, 8))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=y)  # 假设 y 是你的标签
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Visualization of Model Features')
plt.colorbar()
plt.show()
