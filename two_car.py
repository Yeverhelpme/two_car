# 导入必要的库
import pandas as pd  # 数据处理库
import xgboost  # XGBoost建模库
# 统计分析库：皮尔逊相关系数、斯皮尔曼相关系数、方差分析
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.model_selection import train_test_split  # 数据集拆分
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 标签编码和标准化
# 模型评估指标：均方误差、决定系数、平均绝对误差
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 绘图相关库
import plotly.express as px  # 交互式可视化
import matplotlib.pyplot as plt  # 基础可视化
import seaborn as sns  # 高级可视化
import matplotlib
matplotlib.use('TkAgg')  # 强制启用兼容Windows的绘图后端
# 设置中文显示，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ---------------------- 数据读取与基础查看 ----------------------
# 读取训练集数据（二手车交易数据）
data = pd.read_csv('used_car_train_20200313.csv', sep=' ')

# 完善数据查看部分
print("===== 训练集前5行数据 =====")
print(data.head())  # 查看前5行，了解数据结构
print("\n===== 训练集后5行数据 =====")
print(data.tail())  # 查看后5行，检查数据完整性
print("\n===== 训练集数据形状（行数, 列数） =====")
print(data.shape)  # 查看数据规模
print("\n===== 训练集数据类型及缺失值情况 =====")
data.info()  # 查看各列数据类型和缺失值数量
print("\n===== 训练集列名列表 =====")
print(data.columns.tolist())  # 显示所有列名
print("\n===== 训练集数值型特征的统计描述 =====")
print(data.describe())  # 查看均值、标准差、分位数等统计量
print("\n===== 训练集缺失值统计 =====")
print(data.isnull().sum())  # 统计每列缺失值数量

# 读取测试集数据
test_A = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
test_B = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 合并两个测试集（按行合并）
test_set = pd.concat([test_A, test_B], axis=0)
print("\n===== 合并后的测试集缺失值统计 =====")
print(test_set.isnull().sum())  # 查看测试集缺失情况

# 读取提交模板（用于存放预测结果）
submit = pd.read_csv('used_car_sample_submit.csv', sep=',')
print("\n===== 提交模板前5行 =====")
print(submit.head())

# ---------------------- 数据预处理：时间特征处理 ----------------------
# 检查日期格式是否存在异常（将regDate和creatDate转为字符串后尝试解析为日期）
# 若存在无法解析的日期，isna()会返回True
reg_date_error = pd.to_datetime(data['regDate'].astype(str), errors='coerce').isna().any()
creat_date_error = pd.to_datetime(data['creatDate'].astype(str), errors='coerce').isna().any()
print(f"\n注册日期是否存在异常格式：{reg_date_error}")
print(f"创建日期是否存在异常格式：{creat_date_error}")

# 查看异常日期示例（发现regDate存在20070009这类异常值，无法解析为完整日期）
print("\n异常注册日期示例：", data['regDate'][14])

# 时间格式转换：提取年份（避免完整日期解析错误）
data['creatDate'] = pd.to_datetime(data['creatDate'].astype(str))  # creatDate格式正常，直接转日期
data['regDate_year'] = data['regDate'].apply(lambda x: int(str(x)[:4]))  # 提取regDate的年份
data['creatDate_year'] = data['creatDate'].dt.year  # 提取creatDate的年份
print("\n===== 处理后的时间特征前5行 =====")
print(data[['regDate', 'regDate_year', 'creatDate', 'creatDate_year']].head())

# ---------------------- 数据预处理：重复值检查 ----------------------
# 检查重复值：全局重复、按SaleID重复、按name重复
full_dup = data.duplicated().sum()  # 所有列都相同的重复行
saleid_dup = data.duplicated('SaleID').sum()  # SaleID（唯一标识）重复
name_dup = data.duplicated('name').sum()  # 车辆名称重复
print(f"\n全局重复行数：{full_dup}")
print(f"SaleID重复行数：{saleid_dup}")
print(f"name重复行数：{name_dup}")
# 结论：SaleID无重复（符合唯一标识特性），name重复可能是同一客户多辆车，无需处理

# ---------------------- 数据预处理：缺失值分析 ----------------------
# 找出存在缺失值的列
hiatus_feature = data.columns[data.isnull().any(axis=0)]
print("\n存在缺失值的特征：", hiatus_feature.tolist())

# 统计各缺失特征的缺失数量
print("\n各特征缺失值数量：")
print(data[hiatus_feature].isnull().sum())

# 绘制缺失值热力图（直观展示缺失位置）
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap='coolwarm', cbar=False)  # 白色表示缺失
plt.title('缺失值热力图')
plt.show()

# 查看缺失特征的取值分布（为缺失值处理提供依据）
print("\n===== 缺失特征的取值分布 =====")
for col in hiatus_feature:
    print(f'\n{col}的取值 unique值：{data[col].unique()[:5]}...')  # 展示部分取值
    print(f'{col}的取值计数：\n{data[col].value_counts(dropna=False).head()}')  # 包含缺失值的计数

# 缺失值处理思路：
# 1. model缺失值极少，可考虑删除样本或后续模型自动处理
# 2. bodyType, fuelType, gearbox为有序离散特征，XGBoost可自动处理缺失值，暂不填充

# ---------------------- 特征筛选与分类 ----------------------
# 无意义特征剔除：SaleID（唯一标识）、name（重复率高且无明确规律）
# 特征分类：连续特征、类别特征、日期特征
continuous_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3',
                       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
                       'v_13', 'v_14']  # 数值型连续特征

categorical_features = ['model', 'brand', 'bodyType', 'fuelType',
                        'gearbox', 'kilometer', 'notRepairedDamage', 'regionCode',
                        'seller', 'offerType']  # 类别型离散特征

datetime_feature = ['regDate', 'creatDate', 'regDate_year', 'creatDate_year']  # 日期相关特征

# 查看离散特征的取值情况（检查是否有异常值）
print("\n===== 离散特征取值分布 =====")
for col in categorical_features:
    print(f'\n{col}的unique值：{data[col].unique()[:10]}...')  # 展示前10个取值

# ---------------------- 特征工程：特征转换与衍生 ----------------------
# 1. 处理notRepairedDamage特征（将'-'转换为'0.0'，统一格式）
data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', '0.0')

# 2. 衍生特征：汽车使用年限（创建日期年份 - 注册日期年份）
data['car_age'] = data['creatDate_year'] - data['regDate_year']
print("\n===== 衍生特征car_age前5行 =====")
print(data[['regDate_year', 'creatDate_year', 'car_age']].head())

# 3. 高基数特征处理：regionCode（区域代码）基数大，按价格均值分箱
# 计算各区域的平均价格
region_price_mean = data.groupby('regionCode')['price'].mean()
# 按平均价格分5箱，转换为0-4的等级
region_price_bins = pd.qcut(region_price_mean, q=5, labels=False)
# 构建区域到等级的映射字典
region_to_level = region_price_bins.to_dict()
# 新增区域等级特征
data['regionCode_level'] = data['regionCode'].map(region_to_level)
print("\n===== 区域等级特征regionCode_level前5行 =====")
print(data[['regionCode', 'regionCode_level']].head())

# ---------------------- 特征相关性分析 ----------------------
# 1. 多维交叉分析：燃油类型与车龄对价格的影响（热力图）
pivot_ll = data.pivot_table(index='fuelType', columns='car_age', values='price', aggfunc='mean')
fig = px.imshow(
    pivot_ll,
    text_auto='.2f',
    labels=dict(x="车龄", y="燃油类型", color="价格均值"),
    color_continuous_scale="Blues"
)
fig.update_layout(
    title="不同燃油类型——车龄的价格热力图",
    title_x=0.5,
    title_font=dict(size=20)
)
fig.show()  # 展示热力图，观察价格分布规律

# 2. 调整特征列表（加入新衍生特征，删除原始高基数特征）
continuous_features.append('car_age')  # 新增车龄到连续特征
categorical_features.append('regionCode_level')  # 新增区域等级到类别特征
categorical_features.remove('regionCode')  # 删除原始区域代码

# 3. 连续特征与目标变量的线性相关性（皮尔逊相关系数）
print("\n===== 连续特征与价格的皮尔逊相关系数 =====")
for i in continuous_features:
    corr_coef, p_value = pearsonr(data[i], data['price'])
    print(f'{i}: 相关系数={corr_coef:.4f}, p值={p_value:.4f}, '
          f'{"显著" if p_value < 0.05 else "不显著"}')

# 4. 连续特征与目标变量的非线性相关性（斯皮尔曼相关系数）
print("\n===== 连续特征与价格的斯皮尔曼相关系数 =====")
for i in continuous_features:
    corr, p = spearmanr(data[i], data['price'])
    print(f'{i}: 相关系数={corr:.4f}, p值={p:.4f}, '
          f'{"显著" if p < 0.05 else "不显著"}')

# 5. 类别特征与目标变量的相关性（方差分析）
print("\n===== 类别特征与价格的方差分析 =====")
for col in categorical_features:
    if data[col].nunique() <= 1:
        print(f'{col}为常数列，无分析意义，跳过')
        continue
    # 按类别分组提取价格数据
    groups = [data['price'][data[col] == category].dropna()
              for category in data[col].dropna().unique()]
    groups = [g for g in groups if len(g) > 0]  # 过滤空组
    if len(groups) < 2:
        print(f'{col}有效分组不足，无法分析，跳过')
        continue
    try:
        f_stat, p_val = f_oneway(*groups)
        print(f'{col}: F值={f_stat:.4f}, p值={p_val:.4f}, '
              f'{"显著" if p_val < 0.05 else "不显著"}')
    except Exception as e:
        print(f'{col}计算出错：{e}')

# 特征筛选结论：剔除无显著影响的offerType
features = continuous_features + categorical_features
features.remove('offerType')
print("\n最终用于建模的特征：", features)

# ---------------------- 建模准备：数据编码与划分 ----------------------
# 目标变量（预测价格）
target = data['price']

# 标签编码：将notRepairedDamage转换为数值（0/1）
scaler = LabelEncoder()
data['notRepairedDamage'] = scaler.fit_transform(data['notRepairedDamage'])

# 划分训练集和测试集（7:3）
x_train, x_test, y_train, y_test = train_test_split(
    data[features], target, random_state=42, test_size=0.3
)

# 标准化目标变量（消除量纲影响，便于模型评估）
std = StandardScaler()
y_train_scaler = std.fit_transform(y_train.values.reshape(-1, 1))  # 训练集拟合并转换
y_test_scaler = std.transform(y_test.values.reshape(-1, 1))  # 测试集用训练集的规则转换

# ---------------------- 模型训练与评估 ----------------------
# 构建XGBoost回归模型
model = xgboost.XGBRegressor(
    enable_categorical=True,  # 支持类别特征
    n_estimators=300,  # 树的数量
    max_depth=9,  # 树的最大深度
    learning_rate=0.1,  # 学习率
    subsample=0.8,  # 样本采样比例
    objective='reg:squarederror',  # 回归目标函数（平方误差）
    random_state=42  # 随机种子，保证结果可复现
)

# 训练模型
model.fit(x_train.values, y_train_scaler)

# 模型预测（测试集）
y_pred = model.predict(x_test.values)

# 评估模型性能
print("\n===== 模型评估指标 =====")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test_scaler, y_pred):.2f}")
print(f"均方误差 (MSE): {mean_squared_error(y_test_scaler, y_pred):.2f}")
print(f"决定系数 (R²): {r2_score(y_test_scaler, y_pred):.4f}")  # 越接近1越好

# 绘制预测值vs实际值散点图（评估整体拟合效果）
plt.figure(figsize=(10, 5))
plt.scatter(y_test_scaler, y_pred, alpha=0.5, label='预测值', color='royalblue')
plt.plot([y_test_scaler.min(), y_test_scaler.max()],
         [y_test_scaler.min(), y_test_scaler.max()],
         color='red', linestyle='--', label='理想预测线')  # 对角线表示完美预测
plt.xlabel('实际值（标准化后）')
plt.ylabel('预测值（标准化后）')
plt.title('模型预测值 vs 实际值')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制残差图（评估误差分布是否随机）
residuals = y_test_scaler.flatten() - y_pred  # 残差=实际值-预测值
plt.figure(figsize=(10, 5))
plt.scatter(y_pred[:1000], residuals[:1000], alpha=0.5, color='orange')  # 取前1000个样本
plt.hlines(0, y_pred[:1000].min(), y_pred[:1000].max(), colors='red', linestyles='dashed')  # 残差为0的参考线
plt.xlabel('预测值（标准化后）')
plt.ylabel('残差（实际值 - 预测值）')
plt.title('残差图')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------- 测试集预测与结果提交 ----------------------
# 对测试集执行与训练集相同的预处理
print("\n===== 测试集预处理 =====")
# 1. 处理notRepairedDamage特征
test_set['notRepairedDamage'] = test_set['notRepairedDamage'].replace('-', '0.0')

# 2. 时间特征处理
test_set['creatDate'] = pd.to_datetime(test_set['creatDate'].astype(str))
test_set['regDate_year'] = test_set['regDate'].apply(lambda x: int(str(x)[:4]))
test_set['creatDate_year'] = test_set['creatDate'].dt.year

# 3. 衍生汽车使用年限
test_set['car_age'] = test_set['creatDate_year'] - test_set['regDate_year']

# 4. 区域等级映射（使用训练集的映射规则）
test_set['regionCode_level'] = test_set['regionCode'].map(region_to_level)

# 5. 标签编码（使用训练集的编码器）
test_set['notRepairedDamage'] = scaler.transform(test_set['notRepairedDamage'].astype(str))

# 对测试集进行预测
predict = model.predict(test_set[features].values)

# 生成提交文件（取前50000条结果，转为整数）
submit['price'] = predict[:50000].astype(int)
submit.to_csv('./used_car_sample_submit_predict.csv', index=False)  # 保存时去掉索引
print("\n===== 提交文件前5行 =====")
print(submit.head())