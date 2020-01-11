
# coding: utf-8

# In[1]:


# 构建随机森林模型


# In[2]:



# directory of packages
import sys
sys.path += ['/opt/pycharm-community-2018.2.4/helpers/pydev',
 '/opt/spark-hadoop/python',
 '/opt/pycharm-community-2018.2.4/helpers/pydev',
 '/usr/lib/python36.zip',
 '/usr/lib/python3.6',
 '/usr/lib/python3.6/lib-dynload',
 '/usr/local/lib/python3.6/dist-packages',
 '/usr/local/lib/python3.6/dist-packages/setuptools-39.1.0-py3.6.egg',
 '/usr/lib/python3/dist-packages',
 '/usr/local/lib/python3.6/dist-packages/IPython/extensions',
 '/home/zhangshuang/PycharmProjects/untitled']

# directory of working files
import os
os.chdir('/home/zhangshuang/PycharmProjects/untitled')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# data load
df = pd.read_csv('lendingclub2017/LoanStats3a.csv',skiprows=[0])


# In[4]:


# exploring data


# In[5]:


#　定义好样本为：fully paid，标记为０； 坏样本为：charged off，标记为１
df = df[df.loan_status.isin(['Fully Paid', 'Charged Off'])]


# In[6]:


# preprocessing
import sklearn


# In[7]:


# split the data
x_col = df.columns.tolist()
x_col.remove('loan_status')
X = df[x_col]
y = df['loan_status']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

df_train = pd.concat([x_train, y_train], axis=1)
df_vali = pd.concat([x_vali, y_vali], axis=1)
df_test = pd.concat([x_test, y_test], axis=1)


# In[8]:


# default values
# select features which 60% of them is missing
mis_feas = [i for i in df.columns if ((df[i].isnull().sum()) * 1.0 / df.shape[0]) > 0.6]
# output missing rate to get a further understanding of missing info
for i in mis_feas:
    mis_rate = (df[i].isnull().sum()) * 1.0 / df.shape[0]
    print(i, mis_rate)


# In[9]:


mis_feas.remove('mths_since_last_delinq')
mis_feas.remove('mths_since_last_record')
df.drop(mis_feas, axis=1, inplace=True)
df_list = [df_train, df_vali, df_test]
for i in df_list:
    i.drop(mis_feas, axis=1, inplace=True)


# In[10]:


# define missing value features to be filled
mis_feas_to_fil = [i for i in df.columns if df[i].isnull().sum() != 0]
for i in mis_feas_to_fil:
    mis_rate = (df[i].isnull().sum()) * 1.0 / df.shape[0]
    print(i,mis_rate)


# In[11]:


# drop after issued features
for i in df_list:
    i.drop(['last_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med'],
        axis=1,inplace=True)
# fill with mode number
to_fil_with_mode = ['emp_title', 'emp_length', 'title', 
                    'revol_util','pub_rec_bankruptcies', 'tax_liens']

from scipy.stats import mode
for i in to_fil_with_mode:
    for j in df_list:
        j[i][j[i].isnull()] = mode(j[i][j[i].notnull()])[0][0]
    


# In[12]:


# check the result
for i in df_list:
    for j in to_fil_with_mode:
        mis_rate = (i[j].isnull().sum()) * 1.0 / i.shape[0]
        print(j,mis_rate)


# In[13]:


# 同值性特征识别处理,将阈值设定为９０％
# 识别同一性数据
equi_fea = []
for i in df_list[0].columns:
    try:
        mode_value = mode(df[i])[0][0]
        mode_rate = mode(df[i])[1][0]*1.0 / df.shape[0]
        if mode_rate > 0.9:
            equi_fea.append([i, mode_value, mode_rate])
    except:
        pass
e = pd.DataFrame(equi_fea, columns=['col_name', 'mode_value', 'mode_rate'])    
e.sort_values(by='mode_rate')


# In[14]:


# 处理同一性数据
same_val_fea_to_drop = list(e.col_name.values)
for i in ['pub_rec', 'pub_rec_bankruptcies']:
    same_val_fea_to_drop.remove(i)
for i in df_list:
    i.drop(same_val_fea_to_drop, axis=1, inplace=True)


# In[15]:


# features to be regularized
for i in df_list:
    i.term = i.term.str.replace(' months', '').astype('float')
    i.int_rate = i.int_rate.str.replace('%', '').astype('float')
    i.earliest_cr_line = [pd.datetime.strptime(i, '%b-%Y') for i in i.earliest_cr_line]
    i.issue_d = [pd.datetime.strptime(i, '%b-%Y') for i in i.issue_d]
    i.revol_util = i.revol_util.str.replace('%', '').astype('float')


# In[56]:


df_train.term.value_counts()


# In[57]:


# encoding features:grade, emp_length, home_ownership, verification_status,purpose
for i in df_list:
    i.grade.replace({"A": 0,"B": 1,"C": 2, "D": 3, "E": 4,"F": 5,"G": 6}, inplace=True)
    i.emp_length.replace({"10+ years": 11,"9 years": 10,"8 years": 9,
                    "7 years": 8,"6 years": 7,"5 years": 6,"4 years":5,
                    "3 years": 4,"2 years": 3,"1 year": 2,"< 1 year": 1,
                    np.nan: 0}, inplace=True)
    i.home_ownership.replace({"MORTGAGE":0,"OTHER":1,"NONE":2,"OWN":3,"RENT":4}, inplace=True)
    i.verification_status.replace({"Not Verified":0,"Source Verified":1,"Verified":2}, inplace=True)
    i.purpose.replace({"credit_card":0,"home_improvement":1,"debt_consolidation":2,       
                    "other":3,"major_purchase":4,"medical":5,"small_business":6,
                    "car":7,"vacation":8,"moving":9, "house":10, 
                    "renewable_energy":11,"wedding":12, 'educational':13}, inplace=True)
    i.loan_status.replace({'Fully Paid':0, 'Charged Off':1}, inplace=True)
    i.term.replace({36.0:0, 60.0:1}, inplace=True)


# In[17]:


# preprocessing text features
# title 和 desc　两个变量与purpose的信息相关，且分类太多,将两者删除
text_feas = ['emp_title','purpose' ,'addr_state']


# In[18]:


# delete leaking features & not meaningful features
leak_feas = ['sub_grade','title','zip_code','recoveries','last_pymnt_amnt',
            'funded_amnt','funded_amnt_inv','total_pymnt','total_pymnt_inv',
             'total_rec_prncp','total_rec_int', 'desc']
for i in df_list:
        i.drop(leak_feas, axis=1, inplace=True)


# In[19]:


# feature engineering


# In[20]:


# deriving features, cre_hist = issue_d - earliest_cr_line
for i in df_list:
    i['cre_hist'] = [j.days for j in (i.issue_d - i.earliest_cr_line)/30]
    i.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)


# In[21]:


# vif


# In[23]:



# select number features
vif_feas = [i for i in df_train.columns if i not in 
           ['term', 'mths_since_last_delinq', 'mths_since_last_record','addr_state','emp_title','loan_status']]

# calculate vif
from statsmodels.stats.outliers_influence import  variance_inflation_factor as vif
vif_ls = []
for i in range(len(vif_feas)):
    vif_ls.append([vif_feas[i], vif(df_train[vif_feas].values, i)])   
vif_df = pd.DataFrame(vif_ls, columns=['col_name', 'vif'])
vif_df.sort_values(by='vif', ascending=False)


# In[24]:


# calculate correlation
cor = df_train[vif_feas].corr()
# get lower triangular matrix of cor
cor.iloc[:, :] = np.tril(cor.values, k=-1)
# stack columns of cor
cor = cor.stack()
cor[np.abs(cor)>0.7]


# In[26]:


#　从上述关于协方差和VIF的分析中我们可以猜想，installment&loan_amnt,int_rate&grade
# total_acc&open_acc,pub_rec_bankruptcies&pub_rec之间存在线性关系，因为pub_rec_bankruptcies&pub_rec
# 中大多数都是０，所以其有相关可以解释。剩下的几对，我们可以尝试先删除每对中的一个,删除installment & grade,再去检测相关系数
for i in ['installment', 'grade']:
    vif_feas.remove(i)
    
# 有几个处于灰色地带的特征，暂时留下，特征工程中删除特征时要谨慎，因为删除特征，意味着弃用一些信息。
for i in df_list:
    i.drop(['installment', 'grade'], axis=1, inplace=True)


# In[27]:


# feature boxing
feas_to_box = [i for i in df_train.columns if (len(df_train[i].value_counts()))>15]


# In[28]:


# features need special skills to box
special_feas_to_box = ['emp_title', 'addr_state', 'mths_since_last_delinq', 'mths_since_last_record']


# In[29]:


# numerical features to box
num_feas_to_box = [i for i in feas_to_box if i not in special_feas_to_box]


# In[30]:


# 卡方分箱试验

# add target column to num_feas_to_box
# num_feas_to_box.append('loan_status')
# df_chi = df_train[num_feas_to_box]


def Chi2(df, total_col, bad_col,overallRate):
    '''
     #此函数计算卡方值
     :df dataFrame
     :total_col 每个值得总数量
     :bad_col 每个值的坏数据数量
     :overallRate 坏数据的占比
     : return 卡方值
    '''
    df2=df.copy()
    df2['expected']=df[total_col].apply(lambda x: x*overallRate)
    combined=zip(df2['expected'], df2[bad_col])
    chi=[(i[0]-i[1])**2/i[0] for i in combined]
    chi2=sum(chi)
    return chi2
def ChiMerge_MaxInterval_Original(df, col, target,max_interval=5):
    '''
    : df dataframe
    : col 要被分项的特征
    ： target 目标值 0,1 值
    : max_interval 最大箱数
    ：return 箱体
    '''
    colLevels=set(df[col])
    colLevels=sorted(list(colLevels))
    N_distinct=len(colLevels)
    if N_distinct <= max_interval:
        print("the row is cann't be less than interval numbers")
        return colLevels[:-1]
    else:
        total=df.groupby([col])[target].count()
        total=pd.DataFrame({'total':total})
        bad=df.groupby([col])[target].sum()
        bad=pd.DataFrame({'bad':bad})
        regroup=total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N=sum(regroup['total'])
        B=sum(regroup['bad'])
        overallRate=B*1.0/N
        groupIntervals=[[i] for i in colLevels]
        groupNum=len(groupIntervals)
        while(len(groupIntervals)>max_interval):
            chisqList=[]
            for interval in groupIntervals:
                df2=regroup.loc[regroup[col].isin(interval)]
                chisq=Chi2(df2,'total','bad',overallRate)
                chisqList.append(chisq)
            min_position=chisqList.index(min(chisqList))
            if min_position==0:
                combinedPosition=1
            elif min_position==groupNum-1:
                combinedPosition=min_position-1
            else:
                if chisqList[min_position-1]<=chisqList[min_position + 1]:
                    combinedPosition=min_position-1
                else:
                    combinedPosition=min_position+1
            #合并箱体
            groupIntervals[min_position]=groupIntervals[min_position]+groupIntervals[combinedPosition]
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum=len(groupIntervals)
        groupIntervals=[sorted(i) for i in groupIntervals]
        print(groupIntervals)
        cutOffPoints=[i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints


# In[57]:


# encoding features:grade, emp_length, home_ownership, verification_status,purpose
for i in df_list:
    i.grade.replace({"A": 0,"B": 1,"C": 2, "D": 3, "E": 4,"F": 5,"G": 6}, inplace=True)
    i.emp_length.replace({"10+ years": 11,"9 years": 10,"8 years": 9,
                    "7 years": 8,"6 years": 7,"5 years": 6,"4 years":5,
                    "3 years": 4,"2 years": 3,"1 year": 2,"< 1 year": 1,
                    np.nan: 0}, inplace=True)
    i.home_ownership.replace({"MORTGAGE":0,"OTHER":1,"NONE":2,"OWN":3,"RENT":4}, inplace=True)
    i.verification_status.replace({"Not Verified":0,"Source Verified":1,"Verified":2}, inplace=True)
    i.purpose.replace({"credit_card":0,"home_improvement":1,"debt_consolidation":2,       
                    "other":3,"major_purchase":4,"medical":5,"small_business":6,
                    "car":7,"vacation":8,"moving":9, "house":10, 
                    "renewable_energy":11,"wedding":12, 'educational':13}, inplace=True)
    i.loan_status.replace({'Fully Paid':0, 'Charged Off':1}, inplace=True)
    i.term.replace({36.0:0, 60.0:1}, inplace=True)


# In[32]:


# by the sequence of num_feas_to_box:
cut_points_list = [
   [4450.0, 6000.0, 7725.0, 9750.0, 12200.0, 14275.0, 17800.0, 22875.0, 29900.0],
    [6.03, 7.75, 9.99, 13.99, 16.01, 17.49, 18.99, 20.3, 21.67],
    [26004.0, 37225.0, 40560.0, 48156.0, 51669.0, 62851.0, 74143.68, 89092.0, 115275.0],
    [2.53, 4.18, 6.45, 9.65, 12.47, 15.62, 19.31, 21.03, 23.87],
    [2.0, 3.0, 5.0, 6.0, 15.0, 20.0, 26.0, 30.0, 32.0],
    [1098.1, 2838.2, 4632.6,6577.0,8841.0,11465.0,14833.0,19816.2,29139.5],
    [10.17, 21.4, 29.7, 42.0, 55.3, 68.0, 80.5, 92.7, 96.5],
    [3.0, 4.0, 14.0, 26.0, 36.0, 43.0, 50.0, 60.0, 69.0],
    [55, 104, 186, 220, 314, 376, 425, 473, 525]]

cut_points_dict = {}
for i in range(len(num_feas_to_box)):
    cut_points_dict[num_feas_to_box[i]] = cut_points_list[i]


# In[33]:


# 工作机构分类，依据A政府机构类，B银行类，F医院类，E学校类，C自职业类，D公司和其它类，G退休类分类
# 如果df中的emp_title与某个上述A-G有交集，则将它划为该类，用字母字符表示
# 缺省值为’H'

A = ['board', 'general','american','u.s.' ,'army', 'force', 'us', 'states', 'corp', 'navy', 'united', 'department', 'government']
B = ['bank', 'morgan']
C = ['self']
D = 'OTHER'
E = ['college', 'school', 'university']
F = ['hospital', 'clinic', 'health', 'healthcare']
G = ['retired']
ls_letter = [0,1,2,4,5,6]
ls = [A,B,C,E,F,G]

def emp_classify(df1):
    for i in df1.emp_title.index:
        emp_list = []
        for j in range(len(ls)):
            emp_list.append((set(str(df1.emp_title[i]).lower().split()) & set(ls[j])))
        if emp_list.count(set()) != 6:
            sr_emp = pd.Series(emp_list)
            idx = sr_emp[sr_emp!=set()].index
            df1.emp_title[i] = ls_letter[idx[0]]
        else:
            df1.emp_title[i] = 3
    df1.emp_title[df1.emp_title.isnull()] = 7


# In[34]:


# 尝试用频数进行分类，使每一个分组内的样本数尽量相近,假设分8个样本，每个分组大约３３００个样本
S = [['CA'], ['NY'], ['FL', 'TX'], ['NJ', 'PA', 'IL'],['VA','GA','MA','OH'],['MD','AZ','WA','CT','CO','NC'],
 ['MI','MO','MN','NV','OR','WI','LA','SC','AL','OK']]

addr_list = df_train.addr_state.value_counts().index.tolist()

SS = []
for i in S:
    SS += i
    
    
S.insert(7, list(set(addr_list)-set(SS)))

addr_set = []
for i in S:
    addr_set.append(set(i))

addr_dict = {}
for i, j in zip(range(8), addr_set):
    addr_dict[i] = j

    
# 将addr_state进行分类转换
def trans_addr_func(df_to_trans):
    for i in df_to_trans.addr_state:
        for j in range(len(addr_dict)):
            if i in addr_dict[j]:
                df_to_trans.addr_state[df_to_trans.addr_state==i] = list(addr_dict.keys())[j]


# In[36]:


# 对mths_since_last_record & mths_since_last_delinq分箱编写函数
def box_mth_col(df_to_box, mth_col, bins):
    bins = [0.0]+[1.0] + bins + [150.0]
    df_to_box[mth_col][df_to_box[mth_col].notnull()] = pd.cut(
        df_to_box[mth_col][df_to_box[mth_col].notnull()], bins=bins, include_lowest=True,
        labels=range(len(bins)-1))
    df_to_box[mth_col][df_to_box[mth_col].isnull()] = -1


# In[37]:


# 用卡方分箱得到的分割点进行特征分箱，要注意，所有数据集中同样的特征区间都必须是相同的
def box_col_to_df(df_to_box, col, cut_points):
    bins = [-10.0] + cut_points + [100000000.0]
    # 如果有重复切割点,duplicates='drop'去重
    df_to_box[col] = pd.cut(df_to_box[col], bins=bins,include_lowest=True,duplicates='drop', 
                            labels=range(len(bins)-1))


# In[39]:


# 对num_feas_to_box中的特征实施分箱
for df_i in df_list:
    for i,j in zip(num_feas_to_box, cut_points_list):
            box_col_to_df(df_i, i,j)


# In[42]:


# 对mths_since_last_record & mths_since_last_delinq实施分箱
mth_feas = ['mths_since_last_delinq', 'mths_since_last_record']
# mth_cut_points_list by the sequence of mth_feas
mth_cut_points_list = [[19.0, 33.0, 38.0, 63.0],[46.0, 68.0, 79.0, 82.0]]

for df_x in df_list:
    for i,j in zip(mth_feas, mth_cut_points_list):
        box_mth_col(df_x, i,j)


# In[52]:


# 对职务标签进行分箱,计算时间长，结果记得备份！
for df_x in df_list:
    emp_classify(df_x)


# In[53]:


# 对借款人位置分箱，计算时间长，结果记得备份！
for df_x in df_list:
    trans_addr_func(df_x)


# In[72]:


#计算WOE和IV值
def CalcWOE(df,col, target):
    '''
    : df dataframe
    : col 注意这列已经分过箱了，现在计算每箱的WOE和总的IV
    ：target 目标列 0-1值
    ：return 返回每箱的WOE和总的IV
    '''
    total=df.groupby([col])[target].count()
    total=pd.DataFrame({'total':total})
    bad=df.groupby([col])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N=sum(regroup['total'])
    B=sum(regroup['bad'])
    regroup['good']=regroup['total']-regroup['bad']
    G=N-B
    regroup['bad_pcnt']=regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt']=regroup['good'].map(lambda x: x*1.0/G)
    # woe在这里显示不同特征不同属性对好样本的预测能力，这符合评分卡计分标准，分数越高，越可信
    regroup['WOE']=regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    WOE_dict=regroup[[col,'WOE']].set_index(col).to_dict()
    IV=regroup.apply(lambda x:(x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    IV_SUM=sum(IV)
    return {'WOE':WOE_dict,'IV_sum':IV_SUM,'IV':IV}


# In[76]:


# 输出IV
iv_list = []
for i in df_train.columns:
    iv_dict = CalcWOE(df_train,i, 'loan_status')
    iv_list.append(iv_dict['IV_sum'])
    
# 形成IV表
iv_df = pd.DataFrame({'iv_name':df_train.columns.values, 'iv':iv_list})

iv_df.sort_values('iv',ascending=False)


# In[75]:


# 如果输出IV值是无穷，说明该特征中某些属性中缺失某类样本，这需要重新分箱，将这类样本添加到相邻类（对于连续型数值样本）或样本数量较少的那一类（对于分类样本）中去！
#for i in df_train_boxed.columns[[11,21,5,16]]:
#    print(i, '\n',CalcWOE(df_train_boxed,i, 'loan_status'),'\n')
# deling_2yrs:把7.0,8.0,9.0,11.0划归到7.0那一类,全算作6.0
# home_ownership:把2添加到1 这一类
# pub_rec：把3.0，4.0，添加到2.0这一类

# 实施上述区间合并
for df_i in df_list:
    df_i.delinq_2yrs[df_i.delinq_2yrs.isin([7.0,8.0,9.0,11.0])] = 6.0
    df_i.home_ownership[df_i.home_ownership==2] = 1
    df_i.pub_rec[df_i.pub_rec.isin([3.0,4.0])] = 2.0


# In[77]:


# 保留IV值大于0.015的变量
valid_feas = iv_df[iv_df.iv > 0.015].iv_name.tolist()
valid_feas.remove('loan_status')


# In[225]:


# 用随机森林方法进行建模
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=80)

#RF.fit(df_train.drop('loan_status', axis=1).values, df_train['loan_status'].values)


# In[199]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[210]:


# 网格搜索优化模型，对随机森林中的决策树个数(n_estimators)进行网格遍历
from sklearn.model_selection import GridSearchCV
RF = RandomForestClassifier()
parameters = {'n_estimators':[5,10, 20,35,50,100]}
gs = GridSearchCV(estimator=RF, param_grid=parameters, scoring='roc_auc',cv=5,n_jobs=-1)
grid_result = gs.fit(df_train.drop('loan_status', axis=1).values, df_train.loan_status.values)


# In[227]:


lr = LogisticRegression()


# In[213]:


# 学习曲线


# In[228]:


from sklearn.model_selection import learning_curve
 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
   
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
 
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.gca().invert_yaxis()
 
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
 
plot_learning_curve(RF, 'learning curve',df_test.drop('loan_status',axis=1).values, df_test.loan_status.values,train_sizes=np.linspace(.05, 1., 50))


# In[96]:


#计算WOE和IV值
def CalcWOE(df,col, target):
    total=df.groupby([col])[target].count()
    total=pd.DataFrame({'total':total})
    bad=df.groupby([col])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N=sum(regroup['total'])
    B=sum(regroup['bad'])
    regroup['good']=regroup['total']-regroup['bad']
    G=N-B
    regroup['bad_pcnt']=regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt']=regroup['good'].map(lambda x: x*1.0/G)
    # woe在这里显示不同特征不同属性对好样本的预测能力，这符合评分卡计分标准，分数越高，越可信
    regroup['WOE']=regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    WOE_dict=regroup[[col,'WOE']].set_index(col).to_dict()
    IV=regroup.apply(lambda x:(x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    IV_SUM=sum(IV)
    return {'WOE':WOE_dict,'IV_sum':IV_SUM,'IV':IV}


# In[98]:


# 输出IV
iv_list = []
for i in df_train_boxed.columns:
    iv_dict = CalcWOE(df_train,i, 'loan_status')
    iv_list.append(iv_dict['IV_sum'])
# 形成IV表
iv_df = pd.DataFrame({'iv_name':df_train.columns.values, 'iv':iv_list})
iv_df.sort_values('iv',ascending=False)


# In[398]:


# 如果输出IV值是无穷，说明该特征中某些属性中缺失某类样本，这需要重新分箱，将这类样本添加到相邻类（对于连续型数值样本）或样本数量较少的那一类（对于分类样本）中去！
for i in df_train_boxed.columns[[11,21,5,16]]:
    print(i, '\n',CalcWOE(df_train_boxed,i, 'loan_status'),'\n')
# deling_2yrs:把7.0,8.0,9.0,11.0划归到7.0那一类,全算作6.0
# home_ownership:把2添加到1 这一类
# pub_rec：把3.0，4.0，添加到2.0这一类


# In[320]:


# 实施上述区间合并
df_train_boxed.delinq_2yrs[df_train_boxed.delinq_2yrs.isin([7.0,8.0,9.0,11.0])] = 6.0
df_train_boxed.home_ownership[df_train_boxed.home_ownership==2] = 1
df_train_boxed.pub_rec[df_train_boxed.pub_rec.isin([3.0,4.0])] = 2.0


# In[99]:


# 保留IV值大于0.015的变量
valid_feas = iv_df[iv_df.iv > 0.015].iv_name.tolist()
valid_feas.remove('loan_status')


# In[101]:


# 输出IV
iv_dict = {}
for i in valid_feas:
    iv_dict[i] = CalcWOE(df_train_boxed,i, 'loan_status')


# In[114]:


# 用woe替换相应位置的属性值：
df_card_train = df_train[valid_feas+ ['loan_status']]
df_card_test = df_test[valid_feas+ ['loan_status']]

for i in range(len(valid_feas)):
    df_card_train[valid_feas[i]].replace(iv_dict[valid_feas[i]]['WOE']['WOE'],inplace=True)
    df_card_test[valid_feas[i]].replace(iv_dict[valid_feas[i]]['WOE']['WOE'],inplace=True)

# 构建逻辑回归模型：
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(df_card_train.drop('loan_status', axis=1), df_card_train['loan_status'])


# In[130]:


from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(LR.predict(df_card_test.drop('loan_status', axis=1)),df_card_test['loan_status'])
fpr, tpr, thre = roc_curve(LR.predict(df_card_test.drop('loan_status', axis=1)),df_card_test['loan_status'])
ks = max(tpr-fpr)


# In[133]:


print('auc:{}    ks:{}'.format(auc,ks))


# In[159]:


# 输出评分卡
import statsmodels.api as sm

#df_card_train = sm.add_constant(df_card_train)
logit=sm.Logit(df_card_train['loan_status'].values, df_card_train.drop('loan_status',axis=1).values).fit()

B=20/np.log(2)
A=600+20*np.log(1/60)/np.log(2)
basescore=round(A-B*logit.params[0],0)
scorecard=[]
#features.remove('loan_status')
for j, i in enumerate(valid_feas+['loan_amnt']):
    woe = iv_dict[i]['WOE']['WOE']
    interval=[]
    scores=[]
    for key,value in woe.items():
        score=round(-(value*logit.params[j]*B))
        scores.append(score)
        interval.append(key)
    data=pd.DataFrame({'interval':interval,'scores':scores}) 
    scorecard.append(data)

