# Импортируем необходимые библиотеки
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd

data = pd.read_csv('preprocessed/X3_num.csv').drop('Id', axis=1)
target = pd.read_csv('preprocessed/y1.csv').drop('Id', axis=1)

min_value = 5
max_value = 50
step = 5

# Применяем TSNE для визуализации данных
def compute_tsne(perplexity):
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=0)
    tsne_result = tsne.fit_transform(data)
    df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df['Target'] = target
    return df


data_df = {}
for p in range(min_value, max_value + 1, step):
    df = compute_tsne(p)
    data_df[p] = df

# Инициализируем Dash приложение
app = dash.Dash(__name__)

# Определяем макет приложения
app.layout = html.Div([
    dcc.Slider(
        id='perplexity-slider',
        min=min_value,
        max=max_value,
        step=step,
        value=min_value,
        marks={i: str(i) for i in range(min_value, max_value + 1, step)}
    ),
    dcc.Graph(id='tsne-graph', style={'width': '100%', 'height': '100vh'})
])


# Определяем обратную связь для обновления графика при изменении ползунка
@app.callback(
    Output('tsne-graph', 'figure'),
    [Input('perplexity-slider', 'value')]
)
def update_graph(perplexity):
    df = data_df[perplexity]
    fig = go.Figure(data=[go.Scatter3d(
        x=df['TSNE1'],
        y=df['TSNE2'],
        z=df['TSNE3'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Target'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(scene=dict(
                          xaxis_title='TSNE1',
                          yaxis_title='TSNE2',
                          zaxis_title='TSNE3'
                      ))
    return fig


# Запускаем приложение
if __name__ == '__main__':
    app.run_server(debug=True)
