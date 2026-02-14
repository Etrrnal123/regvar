import numpy as np
import matplotlib.pyplot as plt
import umap

def dnaumap():
    # ======================
    # 1. 读取数据
    # ======================
    X = np.load(r"C:\Users\fs201\Downloads\RegVAR\data\fea\DNA\train_info\train_info_features.npy")  # shape: (N, 1024)
    y = np.load(r"C:\Users\fs201\Downloads\RegVAR\data\fea\DNA\train_info\train_info_labels.npy")  # shape: (N,), 0 / 1

    # ======================
    # 2. 计算 UMAP
    # ======================
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    X_umap = reducer.fit_transform(X)

    # ======================
    # 3. 画图（系统级风格）
    # ======================
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # 背景色
    ax.set_facecolor("#F7F7F7")

    # 颜色（柔和、专业）
    color_0 = "#4C72B0"  # 蓝
    color_1 = "#DD8452"  # 橙

    # Label 0
    plt.scatter(
        X_umap[y == 0, 0],
        X_umap[y == 0, 1],
        c=color_0,
        s=8,
        alpha=0.6,
        label="Label 0",
        linewidths=0
    )

    # Label 1
    plt.scatter(
        X_umap[y == 1, 0],
        X_umap[y == 1, 1],
        c=color_1,
        s=8,
        alpha=0.6,
        label="Label 1",
        linewidths=0
    )

    # 标题
    plt.title(
        "UMAP visualization",
        fontsize=14,
        weight="bold"
    )

    # 去掉坐标刻度（系统里很重要）
    plt.xticks([])
    plt.yticks([])

    # 图例
    plt.legend(
        frameon=False,
        markerscale=1.5,
        fontsize=10
    )

    plt.tight_layout()
    ax = plt.gca()

    # 去掉四周黑色边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ======================
    # 4. 保存图片（推荐）
    # ======================
    plt.savefig("umap_labels.png", dpi=300)
    plt.savefig("umap_labels.svg")

    plt.show()
