# -*- coding: utf8 -*-
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series,DataFrame

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

data_train = pd.read_csv("./train.csv");
#data_train.info()
#print data_train.describe();
#print_basic_info()
#print_level_info()
#print_sex_info()
print_cabin_info()





