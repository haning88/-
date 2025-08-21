import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from pylab import mpl    
mpl.rcParams["font.sans-serif"] = ["SimHei"] # 设置显示中文字体 宋体    
mpl.rcParams["axes.unicode_minus"] = False #字体更改后，会导致坐标轴中的部分字符无法正常显示，此时需要设置正常显示负号

data = pd.read_csv('D:\数据挖掘\data\LondonBikeJourneyAug2023.csv')

# 查看数据维度
data.shape

#查看数据信息
data.info()

#查看各列缺失值
data.isna().sum()

#查看重复值
data.duplicated().sum()

# 将起始日期和结束日期转换为日期时间格式
data['Start date'] = pd.to_datetime(data['Start date'])
data['End date'] = pd.to_datetime(data['End date'])

# 获取数据集中最早和最晚的日期
start_date_min = data['Start date'].min()
start_date_max = data['Start date'].max()
end_date_min = data['End date'].min()
end_date_max = data['End date'].max()

print(f"数据集中最早的起始日期: {start_date_min}")
print(f"数据集中最晚的起始日期: {start_date_max}")
print(f"数据集中最早的结束日期: {end_date_min}")
print(f"数据集中最晚的结束日期: {end_date_max}")

data.drop(['End date'],axis=1,inplace=True)

# 获取起始站和终点站
start_stations = data['Start station'].value_counts()
end_stations = data['End station'].value_counts()
print(f'起始站数量:{len(start_stations)},终点站数量:{len(end_stations)}')

# 查看起点站点和终点站点的差异
unique_start_stations = set(start_stations.index)
unique_end_stations = set(end_stations.index)
# 终点站点但不是起点站点
only_end_stations = unique_end_stations - unique_start_stations
for station in only_end_stations:
    print(f"{station}: {end_stations[station]} 次")
    
    # 删除这些站点的数据
stations_to_remove = list(only_end_stations)
data = data[~data['Start station'].isin(stations_to_remove)]
data = data[~data['End station'].isin(stations_to_remove)]

# 删除起始点和终点站是同一个地方的数据
data = data[data['Start station'] != data['End station']]

plt.figure(figsize=(16,8))
sns.boxplot(y=data['Total duration (ms)'])
plt.title('骑行时长分布情况')
plt.ylabel('骑行时长（毫秒）')
plt.show()

data.drop(['Total duration (ms)'],axis=1,inplace=True)

def convert_duration_to_seconds(duration_str):
    time_parts = duration_str.split()
    total_seconds = 0

    for part in time_parts:
        if 'h' in part:
            total_seconds += int(part.replace('h', '')) * 3600
        elif 'm' in part:
            total_seconds += int(part.replace('m', '')) * 60
        elif 's' in part:
            total_seconds += int(part.replace('s', ''))

    return total_seconds

data['Total duration (s)'] = data['Total duration'].apply(convert_duration_to_seconds)

plt.figure(figsize=(16,8))
sns.boxplot(y=data['Total duration (s)'])
plt.title('骑行时长分布情况')
plt.ylabel('骑行时长（秒）')
plt.show()

max_duration = 8 * 3600  # 8小时（28800秒）
min_duration = 60
# 标记并删除异常值
data = data[(data['Total duration (s)'] >= min_duration) & (data['Total duration (s)'] <= max_duration)].copy()

# 构建新特征
data['date'] = data['Start date'].dt.date # 骑行日期
data['hour'] = data['Start date'].dt.hour
data['weekday'] = data['Start date'].dt.day_name()

data.head()

data.describe(include='all')

# 计算每天的骑行总时长和骑行次数
daily_stats = data.groupby('date').agg({'Total duration (s)': 'sum', 'Number': 'count'}).reset_index()
daily_stats.rename(columns={'Number': 'Ride Count'}, inplace=True)
# 创建双 y 轴图表
fig, ax1 = plt.subplots(figsize=(12, 6))
# 绘制总时长
color = 'tab:blue'
ax1.set_xlabel('日期')
ax1.set_ylabel('骑行总时长（秒）', color=color)
ax1.plot(daily_stats['date'], daily_stats['Total duration (s)'], color=color, marker='o', label='总时长')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个 y 轴，共享 x 轴
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('骑行次数', color=color)
ax2.plot(daily_stats['date'], daily_stats['Ride Count'], color=color, marker='x', label='骑行次数')
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.94,0.95))

plt.title('每天的骑行总时长和骑行次数')
plt.show()

plt.figure(figsize=(20,6))
weekday_rides = data.groupby('weekday').size()
plt.subplot(1,2,1)
sns.barplot(x=weekday_rides.index, y=weekday_rides.values, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('每周每天的骑行次数')
plt.xlabel('星期')
plt.ylabel('骑行次数')

weekday_duration = data.groupby('weekday')['Total duration (s)'].sum().reset_index()
weekday_duration['weekday'] = pd.Categorical(weekday_duration['weekday'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
weekday_duration = weekday_duration.sort_values('weekday')
plt.subplot(1,2,2)
sns.barplot(data=weekday_duration, x='weekday', y='Total duration (s)')
plt.title('每周每天的骑行总时长')
plt.xlabel('星期')
plt.ylabel('骑行总时长（秒）')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
hourly_rides = data.groupby('hour').size()
hourly_rides.plot(kind='bar')
plt.title('每小时的骑行次数')
plt.xlabel('小时')
plt.ylabel('骑行次数')
plt.xticks(rotation=0)
plt.show()

# 统计站点间的骑行次数
station_flows = data.groupby(['Start station', 'End station']).size().reset_index(name='ride_counts')

# 创建一个字典来存储站点对的权重
flow_dict = {}
for index, row in station_flows.iterrows():
    start, end, count = row['Start station'], row['End station'], row['ride_counts']
    if (start, end) in flow_dict:
        flow_dict[(start, end)] += count
    else:
        flow_dict[(start, end)] = count

# 将字典转换回DataFrame
station_flows['weight'] = station_flows.apply(lambda row: flow_dict[(row['Start station'], row['End station'])], axis=1)

# 选择前10条常用的站点路线
top_station_flows = station_flows.sort_values(by='weight', ascending=False).head(10)

# 提取前10条路线中的所有站点
top_stations = set(top_station_flows['Start station']).union(set(top_station_flows['End station']))

# 过滤数据，只保留前10条路线中的站点
filtered_flows = station_flows[(station_flows['Start station'].isin(top_stations)) & (station_flows['End station'].isin(top_stations))]

top_station_flows

top_stations

# 创建一个有向图
G = nx.DiGraph()

# 添加边和权重（骑行次数）
for index, row in filtered_flows.iterrows():
    start_station = row['Start station']
    end_station = row['End station']
    weight = row['weight']
    G.add_edge(start_station, end_station, weight=weight)

plt.figure(figsize=(15,10))

# 使用circular_layout布局
pos = nx.circular_layout(G)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# 获取所有颜色
colors = list(mcolors.CSS4_COLORS.values())
np.random.shuffle(colors)

# 绘制边，双向边不显示箭头，自环边显示箭头
edges = G.edges(data=True)
for i, (u, v, d) in enumerate(edges):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrowstyle='-', arrowsize=10, edge_color=colors[i % len(colors)], width=2)

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# 绘制边标签
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('前10条常用站点路线的流量网络图')
plt.show()

# 统计每个站点作为起始站和终点站每天的骑行次数
daily_start_rides = data.groupby(['Start station', 'date']).size().reset_index(name='start_ride_counts')
daily_end_rides = data.groupby(['End station', 'date']).size().reset_index(name='end_ride_counts')

# 计算每个站点的平均每天骑行次数
avg_daily_start_rides = daily_start_rides.groupby('Start station')['start_ride_counts'].mean().reset_index()
avg_daily_start_rides.columns = ['Station', 'Avg Daily Start Rides']

# 找到排名前10的起始站，如果有并列第十名的情况，全部绘制
top_10_avg_rides = avg_daily_start_rides[avg_daily_start_rides['Avg Daily Start Rides'] >= avg_daily_start_rides['Avg Daily Start Rides'].nlargest(10).min()]

# 对结果进行排序
top_10_avg_rides = top_10_avg_rides.sort_values(by='Avg Daily Start Rides', ascending=False)

# 绘制条形图
plt.figure(figsize=(12,8))
plt.barh(top_10_avg_rides['Station'], top_10_avg_rides['Avg Daily Start Rides'], color='skyblue')
plt.xlabel('平均每天骑行次数')
plt.ylabel('起始站')
plt.title('排名前10的起始站（按平均每天骑行次数）')
plt.gca().invert_yaxis()  # 反转y轴，使得排名靠前的在上方
plt.show()


# 找到排名倒数10的起始站，如果有并列第八十名的情况，全部绘制
bottom_10_avg_rides = avg_daily_start_rides[avg_daily_start_rides['Avg Daily Start Rides'] <= avg_daily_start_rides['Avg Daily Start Rides'].nsmallest(10).max()]

# 对结果进行排序
bottom_10_avg_rides = bottom_10_avg_rides.sort_values(by='Avg Daily Start Rides')

# 绘制条形图
plt.figure(figsize=(12,8))
plt.barh(bottom_10_avg_rides['Station'], bottom_10_avg_rides['Avg Daily Start Rides'], color='lightcoral')
plt.xlabel('平均每天骑行次数')
plt.ylabel('起始站')
plt.title('排名倒数10的起始站（按平均每天骑行次数）')
plt.gca().invert_yaxis()  # 反转y轴，使得排名靠前的在上方
plt.show()


avg_daily_end_rides = daily_end_rides.groupby('End station')['end_ride_counts'].mean().reset_index()
avg_daily_end_rides.columns = ['Station', 'Avg Daily End Rides']

# 合并数据，计算差值
avg_daily_rides = pd.merge(avg_daily_start_rides, avg_daily_end_rides, on='Station', how='outer').fillna(0)
avg_daily_rides['avg_ride_diff'] = avg_daily_rides['Avg Daily Start Rides'] - avg_daily_rides['Avg Daily End Rides']

# 找到前十和倒数十的站点
top_10_diff = avg_daily_rides.nlargest(10, 'avg_ride_diff')
bottom_10_diff = avg_daily_rides.nsmallest(10, 'avg_ride_diff')
# 对前十和倒数十的站点分别进行排序
top_10_diff = top_10_diff.sort_values(by='avg_ride_diff', ascending=False)
bottom_10_diff = bottom_10_diff.sort_values(by='avg_ride_diff', ascending=False)

# 合并前十和倒数十的站点数据
combined_diff = pd.concat([top_10_diff, bottom_10_diff])

# 设置颜色
colors = ['skyblue' if x >= 0 else 'lightcoral' for x in combined_diff['avg_ride_diff']]

# 绘制条形图
plt.figure(figsize=(14, 10))
plt.barh(combined_diff['Station'], combined_diff['avg_ride_diff'], color=colors)
plt.xlabel('站点净流量')
plt.ylabel('站点')
plt.title('前十和倒数十站点（站点净流量）')
plt.axvline(0, color='black', linewidth=0.8)
plt.gca().invert_yaxis()  # 反转y轴，使得正x轴的在上方
plt.show()

station_stats = pd.merge(avg_daily_start_rides, avg_daily_end_rides, on='Station', how='outer').fillna(0)# 计算每天平均骑行次数差值
station_stats['Avg Daily Ride Diff'] = station_stats['Avg Daily Start Rides'] - station_stats['Avg Daily End Rides']
# 计算每个站点骑行的平均时长
start_duration = data.groupby('Start station')['Total duration (s)'].mean().reset_index()
start_duration.columns = ['Station', 'Avg Ride Duration']
station_stats = pd.merge(station_stats, start_duration, on='Station', how='outer').fillna(0)

station_stats.head()

features = station_stats.drop(['Station'],axis=1)
# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 使用肘部法则来确定最佳聚类数
inertia = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=15,n_init=10).fit(scaled_features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
    
    plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.xlabel('聚类中心数目')
plt.ylabel('惯性')
plt.title('肘部法则图')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('聚类中心数目')
plt.ylabel('轮廓系数')
plt.title('轮廓系数图')

plt.tight_layout()
plt.show()

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=5, random_state=15,n_init=10)
kmeans.fit(scaled_features)
labels = kmeans.labels_
station_stats['Cluster'] = kmeans.labels_

# 使用 PCA 将数据降维到 2 维以便可视化
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# 可视化聚类结果
plt.figure(figsize=(12, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o')
centers = kmeans.cluster_centers_
reduced_centers = pca.transform(centers)
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('站点聚类结果')
plt.show()


# 可视化对比
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.boxplot(x='Cluster', y='Avg Daily Start Rides', data=station_stats, palette='viridis')
plt.xlabel('聚类')
plt.ylabel('每天平均起始次数')
plt.title('不同聚类中的每天平均起始次数分布')

plt.subplot(2,2,2)
sns.boxplot(x='Cluster', y='Avg Daily End Rides', data=station_stats, palette='viridis')
plt.xlabel('聚类')
plt.ylabel('每天平均终点次数')
plt.title('不同聚类中的每天平均终点次数分布')

plt.subplot(2,2,3)
sns.boxplot(x='Cluster', y='Avg Daily Ride Diff', data=station_stats, palette='viridis')
plt.xlabel('聚类')
plt.ylabel('每天平均骑行次数差值（净流量）')
plt.title('不同聚类中的每天平均骑行次数差值（净流量）分布')

plt.subplot(2,2,4)
sns.boxplot(x='Cluster', y='Avg Ride Duration', data=station_stats, palette='viridis')
plt.xlabel('聚类')
plt.ylabel('平均骑行时长（秒）')
plt.title('不同聚类中的平均骑行时长分布')

plt.tight_layout()
plt.show()

def anova_test(groups):
    anova_result = stats.f_oneway(*groups)
    return anova_result.statistic, anova_result.pvalue

results = []
hourly_groups = [data[data['hour'] == hour].groupby('Start date').size() for hour in range(24)]
stat, pval = anova_test(hourly_groups)
results.append(['Hour', stat, pval])

weekday_groups = [data[data['weekday'] == day].groupby('Start date').size() for day in data['weekday'].unique()]
stat, pval = anova_test(weekday_groups)
results.append(['Weekday', stat, pval])

station_groups = [data[data['Start station'] == station].groupby('Start date').size() for station in station_stats['Station']]
stat, pval = anova_test(station_groups)
results.append(['Station', stat, pval])

bike_model_groups = [data[data['Bike model'] == model].groupby('Start date').size() for model in data['Bike model'].unique()]
stat, pval = anova_test(bike_model_groups)
results.append(['Bike Model', stat, pval])

results_df = pd.DataFrame(results, columns=['特征', '统计值', 'p值'])

print(results_df)