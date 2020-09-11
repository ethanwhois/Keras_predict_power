import pandas as pd
import os
BASEPATH=os.path.dirname(__file__)
file_dir=os.path.join(BASEPATH,'data','2019')


def read_files():
    df_list=[]
    errorlist=[]
    for doc in os.listdir(file_dir):
        doc_name=doc.replace('.csv','')
        doc_dir=os.path.join(file_dir, doc)
        #判断文件是否为空
        if os.path.getsize(doc_dir):
            df=pd.read_csv(
                doc_dir,
                skiprows=4,
                sep=';',
                error_bad_lines=False,
                engine='python',
                encoding="ISO-8859-1",
            )

            # 数据清洗
            # 1.合并并且去掉单位行
            df.columns = df.columns + '[' + df.iloc[0].map(str) + ']'
            df.columns = df.columns.str.replace('\[nan\]','')
            df.drop([0],axis=0,inplace=True)
            # 2.添加日期信息到第一列
            col_name = df.columns.tolist()
            col_name.insert(0, 'date')

            # df['datetime'] = pd.to_datetime(df['TimeStamp[hh:mm]'].map(lambda x:doc_name+' '+x))
            df['date']=doc_name
            # col_name.pop(1)
            df=df[col_name]
            df_list.append(df)
            print('Completed:'+doc_name)
        else:
            errorlist.append(doc_name)
    return df_list,errorlist

df_list,errorlist=read_files()
df_merged=pd.concat(df_list)
df_merged.sort_values(by=['date','TimeStamp[hh:mm]'],inplace=True)
df=df_merged.set_index(['date','TimeStamp[hh:mm]'])
df_eTotal=df.dropna(subset=['E-Total.1[kWh]'])
df_eTotal.to_csv(os.path.join(BASEPATH,'data','E_total.csv'),index=False)
print(['Empty_files:  '+i for i in errorlist])
