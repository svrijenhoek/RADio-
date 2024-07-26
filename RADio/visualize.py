import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pandas as pd


def distplot_per_user(df, algorithms, metric):
    data = []
    df = df[df[metric].notna()]
    for algorithm in algorithms:
        df1 = df[df['algorithm'] == algorithm]
        user_df =df1.groupby('userid')[metric].mean().reset_index()
        data.append(user_df[metric].tolist())

    group_labels = algorithms
    fig = ff.create_distplot(data, group_labels, bin_size=0.01, show_hist=False)
    fig.show()


def distplot_all(df, algorithms, metric):
    data = []
    df1 = df[df[metric].notna()]
    for algorithm in algorithms:
        data.append(df1[df1['algorithm'] == algorithm][metric])

    fig = ff.create_distplot(data, algorithms, bin_size=.01, show_hist=False)
    fig.show()


def lineplot_over_time(df, metric):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['per_hour'] = df.date.dt.round(freq='1h')

    hourly_avg = df.groupby(['per_hour', 'algorithm'])[metric].mean().reset_index()
    fig = px.line(hourly_avg, x='per_hour', y=metric, color="algorithm")
    fig.show()


def seaborn_per_hour(df, metric):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['per_hour'] = df.date.dt.round(freq='1h')

    sns.set_theme(rc={'figure.figsize': (15, 6)})
    sns.lineplot(x="per_hour", y=metric,
                 hue="algorithm",
                 data=df)
