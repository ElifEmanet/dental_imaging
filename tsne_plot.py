import numpy as np
import matplotlib.pyplot as plt
import wandb

from sklearn.manifold import TSNE


def tsne_plot(x, y, name):
    wandb.init(project="dental_imaging",
               name=name,
               settings=wandb.Settings(start_method='fork'))

    tsne = TSNE(perplexity=3, n_components=2, init='pca', n_iter=3500, random_state=32, metric='cosine')
    X_t = tsne.fit_transform(x)

    plt.figure(figsize=(12, 8))

    plt.scatter(X_t[np.where(y == 0), 0],
                X_t[np.where(y == 0), 1],
                marker='o', c='g', linewidth='1', alpha=0.8,
                label='0')

    plt.scatter(X_t[np.where(y == 1), 0],
                X_t[np.where(y == 1), 1],
                marker='o', c='r', linewidth='1', alpha=0.8,
                label='1')

    plt.scatter(X_t[np.where(y == 2), 0],
                X_t[np.where(y == 2), 1],
                marker='o', c='b', linewidth='1', alpha=0.8,
                label='2')

    plt.scatter(X_t[np.where(y == 8), 0],
                X_t[np.where(y == 8), 1],
                marker='o', c='y', linewidth='1', alpha=0.8,
                label='8')

    plt.scatter(X_t[np.where(y == 10), 0],
                X_t[np.where(y == 10), 1],
                marker='o', c='c', linewidth='1', alpha=0.8,
                label='10')

    plt.scatter(X_t[np.where(y == 11), 0],
                X_t[np.where(y == 11), 1],
                marker='o', c='m', linewidth='1', alpha=0.8,
                label='11')

    plt.legend(loc='best')
    wandb.log({"plot": plt})

