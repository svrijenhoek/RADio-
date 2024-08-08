import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pandas as pd


def distplot_per_user(df, algorithms, metric):
    data = []
    for algorithm in algorithms:
        user_df =df.groupby('userid')[algorithm].mean().reset_index()
        user_df = user_df[user_df[algorithm].notna()]
        data.append(user_df[algorithm].tolist())

    group_labels = algorithms
    fig = ff.create_distplot(data, group_labels, bin_size=0.01, show_hist=False)
    fig.update_layout(title='Dist Plot of average values for ' + metric + ' per user')
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


def lineplot(df, metric):
    df['date'] = pd.to_datetime(df['date'])
    df['per_hour'] = df.date.dt.round(freq='1h')
    result = df.drop(columns=['userid', 'date'])
    hourly_avg = result.groupby('per_hour').mean().reset_index()
    df_melted = hourly_avg.melt(id_vars='per_hour', var_name='algorithm', value_name='average_value')
    fig = px.line(df_melted, x='per_hour', y='average_value', color='algorithm', title='Average Values for ' + metric +' Over Time')
    fig.show()

def seaborn_per_hour(df, metric):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['per_hour'] = df.date.dt.round(freq='1h')

    sns.set_theme(rc={'figure.figsize': (15, 6)})
    sns.lineplot(x="per_hour", y=metric,
                 hue="algorithm",
                 data=df)


def visualize(df, metric, algorithms):
    result = df[['userid', 'date', metric]]
    for i, name in enumerate(algorithms):
        result[name] = result[metric].apply(lambda x: x[i])
    result = result.drop(columns=[metric])
    print(result.describe())

    distplot_per_user(result, algorithms, metric)
    lineplot(result, metric)
