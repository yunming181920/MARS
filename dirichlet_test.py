import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 数据集每类的样本数量为 5000
num_samples_per_class = 5000
num_classes = 10
num_users = 100  # 用户数目修改为 100

# 模拟 CIFAR-10 数据集中每类的标签
labels = np.concatenate([np.full(num_samples_per_class, i) for i in range(num_classes)])

# 通过狄利克雷分布划分数据
def dirichlet_distribution(label_data, num_users, alpha):
    user_data = [[] for _ in range(num_users)]
    label_distribution = np.random.dirichlet([alpha] * num_users, num_classes)

    for c in range(num_classes):
        idx = np.where(label_data == c)[0]
        np.random.shuffle(idx)
        proportions = label_distribution[c]
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split_indices = np.split(idx, proportions)

        for user_idx, user_indices in enumerate(split_indices):
            user_data[user_idx] += list(user_indices)

    return user_data

# 函数：根据alpha值绘制图表
def plot_distribution(ax, alpha, labels):
    # 使用不同的alpha值划分数据
    user_data_indices = dirichlet_distribution(labels, num_users, alpha)

    # 统计每个用户每类标签的数据量
    user_class_counts = np.zeros((num_users, num_classes))

    for user_idx, indices in enumerate(user_data_indices):
        user_labels = labels[indices]
        for c in range(num_classes):
            user_class_counts[user_idx, c] = np.sum(user_labels == c)

    # 在 x 轴上表示用户
    x = np.arange(num_users)

    # 逐个类别绘制条形图，不同颜色区分标签类别
    for c in range(num_classes):
        ax.bar(x, user_class_counts[:, c], label=f'Class {c}' if alpha == 0.5 else "",
               bottom=np.sum(user_class_counts[:, :c], axis=1))

    # 设置标签和标题
    ax.set_xlabel('Client Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title(r'$\alpha$' + f'={alpha}', fontsize=16, fontweight='bold')

    # 设置 x 轴刻度
    ax.set_xticks(np.arange(0, num_users, 10))  # 每 10 个用户显示一个标签
    ax.set_xticklabels(np.arange(0, num_users, 10), fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)

# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 依次绘制 alpha=0.2, 0.5, 0.9, 10 四种场景
alphas = [0.2, 0.5, 0.9, 10]
for i, alpha in enumerate(alphas):
    plot_distribution(axs[i // 2, i % 2], alpha, labels)

# 为中间的子图加图例
axs[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# 调整子图布局
plt.tight_layout()

# 保存为 PDF
plt.savefig("dirichlet_data_distribution_comparison.pdf", format='pdf')

# 显示图表
plt.show()
