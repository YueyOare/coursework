import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def impute_by_mean_variate(col):
    mean = col.mean()
    std = col.std()
    random_values = np.random.uniform(mean - std, mean + std, size=len(col))
    return col.combine_first(pd.Series(random_values, index=col.index))


def impute_by_value(data_, method='constant', value=0):
    if isinstance(data_, pd.Series):
        # Если входной объект - Series
        imputer = SimpleImputer(missing_values=np.nan, strategy=method, fill_value=value)
        imputed_data = imputer.fit_transform(data_.values.reshape(-1, 1))
        return pd.Series(imputed_data.flatten(), index=data_.index)
    elif isinstance(data_, pd.DataFrame):
        # Если входной объект - DataFrame
        imputer = SimpleImputer(missing_values=np.nan, strategy=method, fill_value=value)
        imputed_data = imputer.fit_transform(data_)
        return pd.DataFrame(imputed_data, columns=data_.columns)
    else:
        raise ValueError("Unsupported input type. Supported types are pd.Series and pd.DataFrame.")


def impute_by_norm_dist(col, norms=None):
    # col.astype('Float64')
    # print(col.dtype)
    size = len(col)

    if norms is not None:
        # Учитываем референсные значения (нормы)
        mean = (norms[1] + norms[0]) / 2
        std = (norms[1] - norms[0]) / 6
    else:
        mean = col.mean()
        std = col.std()

    random_values = np.random.normal(loc=mean, scale=std, size=size)

    return col.combine_first(pd.Series(random_values, index=col.index))


def impute_by_imputer(data_, categories_):
    imputer = IterativeImputer(random_state=0, estimator=RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=5,
        max_depth=4,
        bootstrap=True,
        n_jobs=-1,
        random_state=0,
    ), max_iter=10, verbose=1)
    data_for_fit = pd.merge(data_, categories_, on='Id').drop('Id', axis=1)
    imputed_data = imputer.fit_transform(
        data_for_fit)
    data_temp = pd.DataFrame(imputed_data[:, :116], columns=data_.drop('Id', axis=1).columns, index=data_.index)
    # data_temp = pd.concat([data_['Id'], ])
    data_temp.insert(0, 'Id', data_['Id'])
    return data_temp


def remove_infrequent_classes(y, threshold=0.01):
    class_counts = y.iloc[:, 1].value_counts()
    classes_to_remove = class_counts[class_counts < len(y) * threshold].index
    return y[~y.iloc[:, 1].isin(classes_to_remove)]


def fill_categories(df):
    temp_df = df.copy()
    temp_df['операции'] = impute_by_value(temp_df['операции'], 'constant', 0)
    temp_df = impute_by_value(temp_df, 'median')
    return temp_df


def fill_numeric(df, categories):
    temp_df = impute_by_imputer(df, categories)
    return temp_df


def scale_features(df):
    scaler = StandardScaler()
    scaler.fit_transform(df)
    return df


def encode_categories(df):
    encoder = OneHotEncoder()
    encoder.fit_transform(df)
    return df


def remove_categories(df):
    return df.iloc[:, 0]


variable_actions = {}
df = pd.read_excel('source/raw/RDBA_BEKHTEREV2.xlsx')
df = df.sort_values(by="Id")
# 1
X = df.drop('КОД3 основной', axis=1)
# X = df
y = df[['Id', 'КОД3 основной']]
num_dfs = []
cat_dfs = []
# 2, 3
X1 = X.fillna(np.nan)
X1.dropna(thresh=2, inplace=True)
y1 = y.loc[X1.index]  # удаление строк с <2 заполненных значений
X1.dropna(thresh=9, axis=1, inplace=True)  # удаление столбцов с <9 заполненных значений
X1 = X1.select_dtypes(exclude='object')  # удаление столбцов с текстом
# 4
y2 = remove_infrequent_classes(y1)
X2 = X1.loc[y2.index]
# 5
X1_num = X1.iloc[:, [0] + list(range(22, X1.shape[1]))]
X2_num = X2.iloc[:, [0] + list(range(22, X2.shape[1]))]
X1_cat = X1.iloc[:, :22]
X2_cat = X2.iloc[:, :22]
num_dfs.append(X1_num)
num_dfs.append(X2_num)
cat_dfs.append(X1_cat)
cat_dfs.append(X2_cat)
variable_actions['X1_num'] = ['with class 5', 'numerical']
variable_actions['X1_cat'] = ['with class 5', 'categorical']
variable_actions['X2_num'] = ['without class 5', 'numerical']
variable_actions['X2_cat'] = ['without class 5', 'categorical']
# 6
temp1 = []
temp2 = []
for df in cat_dfs:
    temp_df = fill_categories(df)
    temp1.append(temp_df)
variable_actions['X3_cat'] = variable_actions['X1_cat'] + ['fill_categories']
variable_actions['X4_cat'] = variable_actions['X2_cat'] + ['fill_categories']

for i, df in enumerate(num_dfs):
    temp_df = fill_numeric(df, cat_dfs[i])
    temp2.append(temp_df)
variable_actions['X3_num'] = variable_actions['X1_num'] + ['fill_numeric']
variable_actions['X4_num'] = variable_actions['X2_num'] + ['fill_numeric']

cat_dfs += temp1
num_dfs += temp2
temp1.clear()
temp2.clear()
# 7, 8
for df in num_dfs:
    temp_df = scale_features(df)
    temp2.append(temp_df)
variable_actions['X5_num'] = variable_actions['X1_num'] + ['scale_features']
variable_actions['X6_num'] = variable_actions['X2_num'] + ['scale_features']
variable_actions['X7_num'] = variable_actions['X3_num'] + ['scale_features']
variable_actions['X8_num'] = variable_actions['X4_num'] + ['scale_features']

for df in cat_dfs:
    temp_df = encode_categories(df)
    temp1.append(temp_df)
    temp_df = remove_categories(df)
    temp1.append(temp_df)
variable_actions['X5_cat'] = variable_actions['X1_cat'] + ['encode_categories']
variable_actions['X6_cat'] = variable_actions['X1_cat'] + ['remove_categories']
variable_actions['X7_cat'] = variable_actions['X2_cat'] + ['encode_categories']
variable_actions['X8_cat'] = variable_actions['X2_cat'] + ['remove_categories']
variable_actions['X9_cat'] = variable_actions['X3_cat'] + ['encode_categories']
variable_actions['X10_cat'] = variable_actions['X3_cat'] + ['remove_categories']
variable_actions['X11_cat'] = variable_actions['X4_cat'] + ['encode_categories']
variable_actions['X12_cat'] = variable_actions['X4_cat'] + ['remove_categories']

cat_dfs += temp1
num_dfs += temp2
temp1.clear()
temp2.clear()

for i, df in enumerate(cat_dfs):
    X_name = f'X{i + 1}_cat'  # Формируем имя переменной
    filename = f'source/preprocessed/{X_name}.csv'  # Формируем имя файла
    df.to_csv(filename, index=False)  # Сохраняем переменную в файл CSV

for i, df in enumerate(num_dfs):
    X_name = f'X{i + 1}_num'  # Формируем имя переменной
    filename = f'source/preprocessed/{X_name}.csv'  # Формируем имя файла
    df.to_csv(filename, index=False)  # Сохраняем переменную в файл CSV

y1.to_csv('source/preprocessed/y1.csv', index=False)
y2.to_csv('source/preprocessed/y2.csv', index=False)
pd.Series(variable_actions).to_csv('data_description.csv')
