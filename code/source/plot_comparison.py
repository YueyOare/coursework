# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
#
# df = pd.read_csv('preprocessed/X1_num.csv')
# imputed_df = pd.read_csv('preprocessed/X2_num.csv')
#
# n_cols = 4
# columns = df.drop('Id', axis=1).columns.tolist()
# n_rows = len(columns) // n_cols + (len(columns) % n_cols > 0)
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
#
# for col, ax in zip(columns, axs.ravel()):
#     sns.histplot(df[col], kde=True, ax=ax, color='blue', label='With class 5')
#     sns.histplot(imputed_df[col], kde=True, ax=ax, color='red', label='Without class 5')
#     ax.set_title(f'Распределение {col}')
#     ax.legend()
#
# plt.tight_layout()
# plt.savefig('distribution_class_comparison.png')  # Сохраняем график в файл

import dash
import pandas as pd
import plotly.figure_factory as ff
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# Загружаем данные
df = pd.read_csv('preprocessed/X1_num.csv')
imputed_df_1 = pd.read_csv('preprocessed/Filled_Zero.csv')
imputed_df_2 = pd.read_csv('preprocessed/Filled_Mean.csv')
imputed_df_3 = pd.read_csv('preprocessed/Filled_Bayesian.csv')
imputed_df_4 = pd.read_csv('preprocessed/Filled_RF.csv')

columns = df.drop('Id', axis=1).columns.tolist()
n_cols = 3
n_rows = len(columns) // n_cols + (len(columns) % n_cols > 0)

# Создаем графики с помощью plotly.figure_factory и plotly.graph_objects
figs = []
total_fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=columns)

for idx, col in enumerate(columns):
    row = idx // n_cols + 1
    col_pos = idx % n_cols + 1

    original_data = df[col].dropna()
    imputed_data_1 = imputed_df_1[col].dropna()
    imputed_data_2 = imputed_df_2[col].dropna()
    imputed_data_3 = imputed_df_3[col].dropna()
    imputed_data_4 = imputed_df_4[col].dropna()

    # Объединяем оба ряда в одном ff.create_distplot
    hist_data = [original_data, imputed_data_1, imputed_data_2, imputed_data_3, imputed_data_4]
    # group_labels = ['With class 5', 'Without class 5']
    group_labels = ['Original', 'Impute Zero', 'Impute Mean', 'Impute BayesianRidge', 'Impute RandomForest']
    colors = ['blue', 'red', 'orange', 'green', 'purple']

    distplot = ff.create_distplot(hist_data, group_labels, colors=colors, show_rug=False, show_hist=False,
                                  histnorm='probability density')
    distplot.update_layout(dict(title_text=col))
    if idx != 3:
        distplot.update_layout(dict(showlegend=False))
    # Добавляем гистограммы как трассы в total_fig
    for trace, label in zip(distplot['data'], group_labels*2):
        trace.update(dict(legendgroup=label, opacity=0.75))
        if idx != 3:
            trace.update(dict(showlegend=False))
        total_fig.add_trace(trace, row=row, col=col_pos)

    figs.append(distplot)

total_fig.update_layout(
    height=n_rows * 250,
    width=n_cols * 350,
    barmode='overlay',
    title_text="Comparison of Original and Imputed Data"
)

total_fig.update_traces(opacity=0.75)

# Создаем Dash-приложение
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='graph', figure=total_fig),
    ], style={'flex': '70%', 'display': 'inline-block'}),

    html.Div([
        dcc.Checklist(
            id='checklist',
            options=[{'label': col, 'value': col} for col in columns],
            value=columns,
            inline=False,
            style={'width': '100%', 'textAlign': 'left'}
        )
    ], style={'flex': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'left'})
], style={'display': 'flex', 'justifyContent': 'center'})


@app.callback(
    Output('graph', 'figure'),
    [Input('checklist', 'value')]
)
def update_graph(selected_columns):
    selected_count = len(selected_columns)
    n_cols_new = min(selected_count, n_cols)
    n_rows_new = (selected_count - 1) // n_cols + 1

    new_fig = make_subplots(rows=n_rows_new, cols=n_cols_new, subplot_titles=selected_columns)

    idx = 0
    for col in selected_columns:
        idx += 1
        row = (idx - 1) // n_cols_new + 1
        col_pos = (idx - 1) % n_cols_new + 1

        for distplot in figs:
            # Ищем distplot с нужным именем и добавляем его трассы в новый график
            if distplot.layout.title.text == col:
                for trace in distplot['data']:
                    new_fig.add_trace(trace, row=row, col=col_pos)

    new_fig.update_layout(
        height=n_rows_new * 250,
        width=n_cols_new * 350,
        barmode='overlay',
        title_text="Comparison of Original and Imputed Data"
    )

    new_fig.update_traces(opacity=0.75)

    return new_fig


if __name__ == '__main__':
    app.run_server(debug=True)

# df = pd.read_excel('raw/RDBA_BEKHTEREV2.xlsx')
# df = df.sort_values(by="Id")
# target_counts = df['КОД3 основной'].value_counts()
# colors = sns.color_palette('pastel')[0:7]
# plt.pie(x=target_counts, labels=['БА', 'СД', 'Депрессия', 'Контроль', 'б.Паркинсона', 'Деменция'], colors=colors, autopct='%.2f%%')
# plt.title('Распределение классов в датасете')
# # plt.show()
# plt.savefig('classes_distribution.png')
