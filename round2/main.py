import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load data
# train_path = 'tcdata/hy_round2_train_20200225'
# test_path = 'tcdata/hy_round2_testB_20200312'
train_path = 'E:\\比赛\天池\智慧海洋建设\hy_round2_train_20200225'
test_path = 'E:\\比赛\天池\智慧海洋建设\hy_round2_testA_20200225'
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

# a=6378137.0000
# b=6356752.3142
# B0=0
# lon0=0
# def k_val_compute():
    # e_=np.sqrt(a**2-b**2)/b
    # return a**2*np.cos(B0)/(b*np.sqrt((1+(e_*np.cos(B0))**2)))

# def X_unit_trans(lon):
    # lon_rad=lon*np.pi/180 #角度-> 弧度
    # k=k_val_compute()
    # return k*(lon_rad-lon0)

# def Y_unit_trans(lat):
    # lat_rad=lat*np.pi/180 #角度-> 弧度
    # k=k_val_compute()
    # e=np.sqrt(a**2-b**2)/a
    # dot_val=np.tan(np.pi/4+lat_rad/2)*((1-e*np.sin(lat_rad))/(1+e*np.sin(lat_rad)))**(e/2)                        
    # return k*np.log(dot_val)

ret = []
for file in tqdm(train_files):
    df = pd.read_csv(f'{train_path}/{file}')
    ret.append(df)
df_train = pd.concat(ret)
df_train.columns = ['ship','lat','lon','v','d','time','type']

ret = []
for file in tqdm(test_files):
    df = pd.read_csv(f'{test_path}/{file}')
    ret.append(df)
df_test = pd.concat(ret)
df_test.columns = ['ship','lat','lon','v','d','time','type']

# Feature extract
def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def extract_feature(df, train):
    # 基本 统计特征
    # t = group_feature(df, 'ship','x',['max','min','mean','std','skew','sum','count'])
    # train = pd.merge(train, t, on='ship', how='left')
    # t = group_feature(df, 'ship','y',['max','min','mean','skew','sum'])
    # train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','lat',['max','min','mean','std','skew','sum','count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','lon',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    
    # # 上下1/4分位数，x y的协方差，相关系数
    # t=df.groupby('ship')['x'].agg({'x_25':lambda x: np.percentile(x,25),'x_50':lambda x: np.percentile(x,50)
    #                                ,'x_75':lambda x: np.percentile(x,75)})
    # train = pd.merge(train, t, on='ship', how='left')
    # t=df.groupby('ship')['y'].agg({'y_25':lambda x: np.percentile(x,25),'y_50':lambda x: np.percentile(x,50),
    #                                'y_75':lambda x: np.percentile(x,75)})
    # train = pd.merge(train, t, on='ship', how='left') 
    t=df.groupby('ship')['lon'].agg({'lon_25':lambda x: np.percentile(x,25),'lon_50':lambda x: np.percentile(x,50)
                                   ,'lon_75':lambda x: np.percentile(x,75)})
    train = pd.merge(train, t, on='ship', how='left')
    t=df.groupby('ship')['lat'].agg({'lat_25':lambda x: np.percentile(x,25),'lat_50':lambda x: np.percentile(x,50),
                                   'lat_75':lambda x: np.percentile(x,75)})
    train = pd.merge(train, t, on='ship', how='left') 
    t=df.groupby('ship')['v'].agg({'v_50':lambda x: np.percentile(x,50),
                                    'v_75':lambda x: np.percentile(x,75)})
    train = pd.merge(train, t, on='ship', how='left')
    t=df.groupby('ship')['d'].agg({'d_75':lambda x: np.percentile(x,75)})
    train = pd.merge(train, t, on='ship', how='left')
    # train['xy_cov']=df[['ship','x','y']].groupby('ship').cov().values[::2,1]
    # train['xy_corr']=df[['ship','x','y']].groupby('ship').corr().values[::2,1]
    train['lon_lat_cov']=df[['ship','lat','lon']].groupby('ship').cov().values[::2,1]
    train['lon_lat_corr']=df[['ship','lat','lon']].groupby('ship').corr().values[::2,1]
    # x,y 交叉特征
    # train['x_max_x_min'] = train['x_max'] - train['x_min']
    # train['y_max_y_min'] = train['y_max'] - train['y_min']
    # train['y_max_x_min'] = train['y_max'] - train['x_min']
    # train['x_max_y_min'] = train['x_max'] - train['y_min']
    # train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    # train['area'] = train['x_max_x_min'] * train['y_max_y_min']
    # lat,lon 交叉特征
    train['lat_max_lat_min'] = train['lat_max'] - train['lat_min']
    train['lon_max_lat_min'] = train['lon_max'] - train['lat_min']
    train['lon_max_lon_min'] = train['lon_max'] - train['lon_min']
    train['lat_max_lon_min'] = train['lat_max'] - train['lon_min']
    train['lat_lon_slope'] = train['lat_max_lat_min'] / np.where(train['lon_max_lon_min']==0, 0.001, train['lon_max_lon_min'])
    train['lat_lon_area'] = train['lon_max_lon_min'] * train['lat_max_lat_min']
    # 时间特征 
    mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict() # 最频繁的时间
    train['mode_hour'] = train['ship'].map(mode_hour)
    t = group_feature(df, 'ship','day',['nunique']) # day nunique
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['time'].agg({'dif_time':lambda x:np.max(x)-np.min(x)}).reset_index() # 耗时长(以秒计算)
    t['dif_second'] = t['dif_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
    #
    df=df.set_index('ship')
    diff_feat=['lon','lat','v','d']
    diff_cols=['diff_'+id for id in diff_feat]
    diff_df=df[diff_feat].diff(1)
    diff_df.columns=diff_cols
    diff_df['time_seconds'] = df['time'].diff(1).dt.total_seconds()
    diff_df['diff_dis'] = np.sqrt(diff_df['diff_lon']**2 + diff_df['diff_lat']**2)
    diff_df['diff_lon/lat'] = diff_df['diff_lon']/np.where(diff_df['diff_lat']==0, 0.001, diff_df['diff_lat'])
    diff_df['diff_dis_time_seconds']=diff_df['diff_dis']/diff_df['time_seconds']
    
    diff_time_seconds=[]
    diff_zeros=[]
    diff_pos=[]
    for f in diff_cols:
        diff_time_seconds.append(f+'_time_seconds')
        diff_zeros.append(f+'_zero')
        diff_pos.append(f+'_pos')
        diff_df[diff_time_seconds[-1]]=diff_df[f].abs() / diff_df['time_seconds']
        diff_df[diff_zeros[-1]]=diff_df[f].apply(lambda x: 1 if x==0 else 0)
        diff_df[diff_pos[-1]]=diff_df[f].apply(lambda x: 1 if x>0 else 0)

    diff_df=diff_df.reset_index()
    t=group_feature(diff_df,'ship','diff_dis',['mean','max'])
    train = pd.merge(train, t, on='ship', how='left')
    t=group_feature(diff_df,'ship','diff_lon/lat',['mean','max','min'])
    train = pd.merge(train, t, on='ship', how='left')
    t=group_feature(diff_df,'ship','diff_dis_time_seconds',['mean','min'])
    train = pd.merge(train, t, on='ship', how='left')
    for f in diff_time_seconds:
        t=group_feature(diff_df,'ship',f,['mean','min'])
        train = pd.merge(train, t, on='ship', how='left')
    for f in diff_zeros:
        t=group_feature(diff_df,'ship',f,['mean'])
        train = pd.merge(train, t, on='ship', how='left')
    for f in diff_pos:
        t=group_feature(diff_df,'ship',f,['mean'])
        train = pd.merge(train, t, on='ship', how='left')

    return train

def extract_dt(df):
    df['time'] =pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['day']=df.time.dt.day
    df['hour']=df.time.dt.hour
    df['second']=df.time.dt.second
    df['minute']=df.time.dt.minute
    
    return df
# lat,lon -> x,y	
# df_train['x']=df_train.lon.apply(X_unit_trans)
# df_train['y']=df_train.lat.apply(Y_unit_trans)

# df_test['x']=df_test.lon.apply(X_unit_trans)
# df_test['y']=df_test.lat.apply(Y_unit_trans)

# label
train_label = df_train.drop_duplicates('ship')
test_label = df_test.drop_duplicates('ship')
	
type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)

train_label = extract_feature(extract_dt(df_train), train_label)
test_label = extract_feature(extract_dt(df_test), test_label)

# Choose features 
features = [x for x in train_label.columns if x not in ['ship','type','time','dif_time',
    'd_min','v_min','day_nunique','d_max','v_sum','v_mean','d_skew','mode_hour',
    'lon_max_lon_min','diff_lon/lat_min','d_sum','lon_sum','v_max','lat_count','lon_lat_cov','diff_dis_time_seconds_min',
    'diff_dis_max','lat_lon_area','diff_d_pos_mean','diff_dis_time_seconds_mean','diff_v_time_seconds_min','lon_lat_corr',
    'diff_lon/lat_max','diff_lon_time_seconds_min','lat_std','diff_lon_time_seconds_mean','diff_v_pos_mean','diff_lat_time_seconds_mean',
    'diff_d_time_seconds_min','d','lon_mean','diff_lon_pos_mean','v','d_mean','d_75','lat_mean'
]]
target = 'type'

print(len(features),len(train_label.columns), ','.join(features))

# Model  Lightgbm 单模
lgb_params = {
    'learning_rate' : 0.03,
    'n_estimators': 2000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 200,
}
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_label[features].copy()
y = train_label[target]
X_test=test_label[features].copy()
models = []
pred = np.zeros((len(test_label),3))
oof = np.zeros((len(X), 3))

for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

    train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
    val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

    model = lgb.train(lgb_params, train_set, valid_sets=[train_set, val_set],verbose_eval=100)
    models.append(model)
    val_pred = model.predict(X.iloc[val_idx])
    oof[val_idx] = val_pred
    val_y = y[val_idx]
    val_pred = np.argmax(val_pred, axis=1)
    
    print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
    test_pred = model.predict(X_test)
    pred += test_pred/fold.n_splits
# import catboost as cbt
# cab_params={
    # 'learning_rate':0.073683,
    # 'iterations':3000, 
    # 'eval_metric':'MultiClass',
    # 'use_best_model':True, 
    # 'random_seed':2020, 
    # 'logging_level':'Verbose', 
    # 'early_stopping_rounds':200, 
    # 'loss_function':'MultiClass',
# }
# X = train_label[features].copy()
# y = train_label[target]
# X_test=test_label[features].copy()
# pred = np.zeros((len(test_label),3))
# oof = np.zeros((len(X), 3))
# models = []

# kfold = StratifiedKFold(n_splits=30, random_state=12306, shuffle=True)
# for index, (trn_idx, val_idx) in enumerate(kfold.split(X=X, y=y)):

    # print('-' * 88)
    # x_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
    # x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # trn_pool = cbt.Pool(x_trn, y_trn)
    # val_pool = cbt.Pool(x_val, y_val)
    # model = cbt.CatBoostClassifier(**cab_params)
    # model.fit(trn_pool, eval_set=val_pool, verbose=100)
    # models.append(model)
    # pred += (model.predict_proba(X_test) / kfold.n_splits)
    # val_pred= model.predict_proba(x_val)
    # oof[val_idx] = val_pred
    # val_pred = np.argmax(val_pred, axis=1)
    # print(index, 'val f1', metrics.f1_score(y_val, val_pred, average='macro'))
    
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))
# oof f1 0.8848700426084485 5folds  oof f1 0.8998521348929372 10folds 39-features

# Write submission
pred = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)
sub.to_csv('result.csv', index=None, header=None)   

# Feature importance
# ret = []
# for index, model in enumerate(models):
    # df = pd.DataFrame()
    # # df['name'] = model.feature_name()
    # # df['score'] = model.feature_importance()
    # df['name'] = model.feature_name()
    # df['score'] = model.feature_importance()
    # df['fold'] = index
    # ret.append(df)
    
# df = pd.concat(ret)

# df = df.groupby('name', as_index=False)['score'].mean()
# df = df.sort_values(['score'], ascending=False)
# print(df)