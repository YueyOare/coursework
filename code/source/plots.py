import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# df = pd.read_csv('preprocessed/X1_num.csv')
# imputed_df = pd.read_csv('preprocessed/X3_num.csv')
#
# n_cols = 3
# columns = df.drop('Id', axis=1).columns.tolist()
# n_rows = len(columns) // n_cols + (len(columns) % n_cols > 0)
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
#
# for col, ax in zip(columns, axs.ravel()):
#     sns.histplot(df[col], kde=True, ax=ax, color='blue', label='Original')
#     sns.histplot(imputed_df[col], kde=True, ax=ax, color='red', label='Imputed')
#     ax.set_title(f'Распределение {col}')
#     ax.legend()
#
# plt.tight_layout()
# plt.savefig('source/distribution_comparison.png')  # Сохраняем график в файл

df = pd.read_excel('raw/RDBA_BEKHTEREV2.xlsx')
df = df.sort_values(by="Id")
target_counts = df['КОД3 основной'].value_counts()
colors = sns.color_palette('pastel')[0:7]
plt.pie(x=target_counts, labels=target_counts.index, colors = colors, autopct='%.2f%%')
# plt.show()
plt.savefig('classes_distribution.png')
