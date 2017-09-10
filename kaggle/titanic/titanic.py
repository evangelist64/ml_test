# -*- coding: utf8 -*-
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing as preprocessing
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import BaggingRegressor
def print_basic_info():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
     
    plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
    plt.title(u"获救情况 (1为获救)") # 标题
    plt.ylabel(u"人数")  

    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"船舱等级分布")

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Age, data_train.Survived)
    plt.xlabel(u"年龄")                         # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='x') 
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")# plots an axis lable
    plt.ylabel(u"密度") 
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.

    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")  
    plt.show()

def print_level_info():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各船舱等级的获救情况")
    plt.xlabel(u"船舱等级") 
    plt.ylabel(u"人数") 
    plt.show()

def print_sex_info():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title(u"按性别看获救情况")
    plt.xlabel(u"性别") 
    plt.ylabel(u"人数")
    plt.show()

def print_cabin_info():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按Cabin有无看获救情况")
    plt.xlabel(u"Cabin有无") 
    plt.ylabel(u"人数")
    plt.show()

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

#初步分析
data_train = pd.read_csv("./train.csv");
#data_train.info()
#print data_train.describe();
#print_basic_info()
#print_level_info()
#print_sex_info()
#print_cabin_info()

#特征工程
# 把已有的数值型特征取出来丢进Random Forest Regressor中
age_df = data_train[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# 乘客分成已知年龄和未知年龄两部分
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
# y即目标年龄
y = known_age[:, 0]
# X即特征属性值
X = known_age[:, 1:]
# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)
# 用得到的模型进行未知年龄结果预测
predictedAges = rfr.predict(unknown_age[:, 1::])
# 用得到的预测结果填补原缺失数据
data_train.loc[ (data_train.Age.isnull()), 'Age' ] = predictedAges

#补足Cabin
set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
data_train_c = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

#计算训练集的平均值和标准差，以便测试数据集使用相同的变换StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = preprocessing.StandardScaler()
reshape_age = data_train_c['Age'].reshape(-1,1)
reshape_fare = data_train_c['Fare'].reshape(-1,1)
data_train_c['Age_scaled'] = scaler.fit_transform(reshape_age, scaler.fit(reshape_age))
data_train_c['Fare_scaled'] = scaler.fit_transform(reshape_fare, scaler.fit(reshape_fare))

print data_train_c

# 用正则取出我们要的属性值
data_train_f = data_train_c.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = data_train_f.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
# fit到LogisticRegression之中
lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
lr.fit(X, y)

#同样处理test数据
data_test = pd.read_csv("./test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

#计算训练集的平均值和标准差，以便测试数据集使用相同的变换StandardScaler(copy=True, with_mean=True, with_std=True)
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
reshape_age = df_test['Age'].reshape(-1,1)
reshape_fare = df_test['Fare'].reshape(-1,1)
df_test['Age_scaled'] = scaler.fit_transform(reshape_age, scaler.fit(reshape_age))
df_test['Fare_scaled'] = scaler.fit_transform(reshape_fare, scaler.fit(reshape_fare))

df_test_f = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#预测
predictions = lr.predict(df_test_f)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("./result.csv", index=False)

#分析优化
#print pd.DataFrame({"columns":list(data_train_f.columns)[1:], "coef":list(lr.coef_.T)})

#交叉验证
X = data_train_f.as_matrix()[:,1:]
Y = data_train_f.as_matrix()[:,0]
#print cross_validation.cross_val_score(lr, X, Y, cv=5)

# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = cross_validation.train_test_split(data_train, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
lr.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
# 对cross validation数据进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = lr.predict(cv_df.as_matrix()[:,1:])
bad_cases = data_train.loc[data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
print bad_cases
'''
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
# fit到BaggingRegressor之中
lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_lr = BaggingRegressor(lr, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
bagging_lr.fit(X, y)

predictions = bagging_lr.predict(df_test_f)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("./result2.csv", index=False)
'''
