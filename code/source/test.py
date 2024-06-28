import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Пути к файлам
experiment_results_pca_path = 'experiment_results_pca_2.csv'
model_files_path = 'experiment_results_pca/'

# Загрузка параметров PCA
pca_params_df = pd.read_csv(experiment_results_pca_path, sep='\t')

# Названия моделей для построения визуализаций
model_names = ['model_13681', 'model_13682', 'model_14230', 'model_14231', 'model_14247', 'model_14253', 'model_14263', 'model_13698']  # Укажите нужные модели
# model_names = ['model_2862', 'model_2879', 'model_2899', 'model_2912', 'model_2928', 'model_2947']  # Укажите нужные модели

# Количество строк и столбцов в таблице графиков
num_rows = 2
num_cols = 4

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

# Перемещение оси для правильного отображения графиков
axes = axes.flatten()

# Обработка каждой модели
for idx, model_name in enumerate(model_names):
    row = pca_params_df[pca_params_df['model_name'] == model_name].iloc[0]
    method = row['method']
    data_paths = row['data']
    parameters = row['parameters']
    n_clusters = row['n_clusters']
    pca_params = eval(row['pca_params'])[0]

    # Путь к model_X.csv файлу
    model_path = os.path.join(model_files_path, f"{model_name}.csv")

    try:
        # Чтение данных из model_X.csv
        data_df = pd.read_csv(model_path)
    except pd.errors.ParserError as e:
        print(f"Ошибка чтения файла {model_name}: {e}")
        continue
    except FileNotFoundError as e:
        print(f"Файл {model_name} не найден: {e}")
        continue

    # Извлечение меток и признаков
    labels_pred = data_df['labels_pred']
    features = data_df.drop(columns=['Id', 'labels_pred', 'labels_true'])

    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    X_tsne = tsne.fit_transform(features)
    ax = axes[idx]
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_pred, palette="viridis", ax=ax, legend=None)
    ax.set_title(
        f"{model_name}\nMethod={method}\nPCA Components={pca_params}\nClusters={n_clusters}")

plt.tight_layout()
plt.savefig('visualizations_pca.png')

# import pandas as pd
# import numpy as np
# from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
#
# # Загрузка файла с результатами
# results = pd.read_csv('experiment_results_pca_2.csv', sep='\t')
#
# # Итерация по строкам DataFrame
# for i, row in results.iterrows():
#     try:
#         model_name = row['model_name']
#         df = pd.read_csv(f'experiment_results_pca/{model_name}.csv')
#
#         # Извлечение признаков и меток
#         labels_pred = df['labels_pred']
#         features = df.drop(columns=['Id', 'labels_pred', 'labels_true'])
#
#         # Исключение шумовых меток (-1), если есть
#         valid_indices = labels_pred >= 0
#         features_valid = features[valid_indices]
#         labels_pred_valid = labels_pred[valid_indices]
#
#         # Вычисление уникальных кластеров
#         unique_clusters = np.unique(labels_pred_valid)
#         n_clusters = len(unique_clusters)
#         results.loc[i, 'n_clusters'] = n_clusters
#
#         if n_clusters > 1:  # Метрики применимы только если больше одного кластера
#             # Расчёт метрик
#             dbi = davies_bouldin_score(features_valid, labels_pred_valid)
#             chi = calinski_harabasz_score(features_valid, labels_pred_valid)
#             silhouette = silhouette_score(features_valid, labels_pred_valid)
#
#             # Запись метрик в DataFrame
#             results.loc[i, 'DBI'] = dbi
#             results.loc[i, 'CHI'] = chi
#             results.loc[i, 'Silhouette'] = silhouette
#         else:
#             # Если кластеров меньше двух, метрики бессмысленны
#             results.loc[i, 'DBI'] = np.nan
#             results.loc[i, 'CHI'] = np.nan
#             results.loc[i, 'Silhouette'] = np.nan
#     except:
#         continue
#
# # Сохранение измененного DataFrame
# results.to_csv('experiment_results_pca_2.csv', sep='\t', index=False)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.manifold import TSNE
# import umap
# import numpy as np
#
# # Загрузка данных
# X = pd.read_csv('preprocessed/X4_num.csv').drop('Id', axis=1)
# y = pd.read_csv('preprocessed/y2.csv').iloc[:, 1]
#
# # Значения perplexity для t-SNE и n_neighbors для UMAP
# perplexity_values = np.arange(5, 130, 5)
# umap_n_neighbors_values = np.arange(5, 130, 5)
#
# # Общее количество графиков
# total_plots = len(perplexity_values) + len(umap_n_neighbors_values)
#
# # Количество строк и столбцов в таблице графиков
# num_cols = 4
# num_rows = (total_plots + num_cols - 1) // num_cols  # Округление вверх для получения целого числа строк
#
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
# axes = axes.flatten()  # Преобразование в одномерный массив для удобного доступа
#
# # Визуализация данных с помощью t-SNE
# for i, perplexity in enumerate(perplexity_values):
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#     X_tsne = tsne.fit_transform(X)
#     ax = axes[i]
#     sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, ax=ax, palette='viridis', legend=None)
#     ax.set_title(f"t-SNE with perplexity={perplexity}")
#
# # Визуализация данных с помощью UMAP
# for i, n_neighbors in enumerate(umap_n_neighbors_values):
#     reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
#     X_umap = reducer.fit_transform(X)
#     ax = axes[len(perplexity_values) + i]
#     sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, ax=ax, palette='viridis', legend=None)
#     ax.set_title(f"UMAP with n_neighbors={n_neighbors}")
#
# # Удаление пустых субплотов, если таковые имеются
# for j in range(total_plots, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.savefig('clusters.png')
# # plt.show()


