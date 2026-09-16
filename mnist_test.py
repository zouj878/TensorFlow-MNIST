import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文，解决字体警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===================== 1. 加载并预处理 MNIST 数据集 =====================
# 自动下载数据集（第一次运行会下载，约10MB，后续直接读取）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理：归一化（把像素值从 0-255 缩放到 0-1，提升模型训练效率）
x_train = x_train / 255.0
x_test = x_test / 255.0

# 打印数据集基本信息，直观了解数据
print("✅ 数据集加载完成：")
print(f"训练集：{x_train.shape}（60000张28×28的手写数字图片）")
print(f"测试集：{x_test.shape}（10000张测试图片）")

# ===================== 2. 构建简单的神经网络模型 =====================
model = tf.keras.Sequential([
    # 展平层：把28×28的二维图片转换成784维的一维向量（输入层）
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 隐藏层：128个神经元，激活函数ReLU（增加非线性）
    tf.keras.layers.Dense(128, activation='relu'),
    # 输出层：10个神经元（对应0-9共10个数字），激活函数softmax（输出概率）
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型：指定优化器、损失函数、评估指标
model.compile(
    optimizer='adam',  # 常用优化器，自动调整学习率
    loss='sparse_categorical_crossentropy',  # 适合分类任务的损失函数
    metrics=['accuracy']  # 用准确率评估模型效果
)

# ===================== 3. 训练模型 =====================
print("\n🚀 开始训练模型（仅训练5轮，新手快速体验）：")
history = model.fit(
    x_train, y_train,
    epochs=5,  # 训练轮数（轮数越多越准，这里选5轮兼顾速度和效果）
    validation_split=0.1  # 用10%的训练数据做验证，监控过拟合
)

# ===================== 4. 评估模型效果 =====================
print("\n📊 模型测试集评估：")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"测试集准确率：{test_acc:.4f}（一般能到97%以上）")

# ===================== 5. 可视化：随机选5张测试图片并预测 =====================
# 取消随机种子，保证每次选的图不同
np.random.seed(None)
# 从测试集随机选5个不重复的索引（每次运行都不一样）
random_idx = np.random.choice(len(x_test), 5, replace=False)

# 用模型预测随机选中的5张图片
predictions = model.predict(x_test[random_idx])
# 取每个预测结果中概率最大的数字（模型认为的答案）
predicted_labels = np.argmax(predictions, axis=1)

# 绘制图片+真实标签+预测标签
plt.figure(figsize=(10, 4))
for i, idx in enumerate(random_idx):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')  # 显示随机选中的手写数字图片
    plt.title(f"真实：{y_test[idx]}\n预测：{predicted_labels[i]}")
    plt.axis('off')  # 隐藏坐标轴
plt.tight_layout()
plt.show()

print("\n🎉 体验完成！你可以看到模型能准确识别不同的手写数字～")
