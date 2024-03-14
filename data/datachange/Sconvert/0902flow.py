import pandas as pd
import numpy as np
mm = 60
df2 = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//0902data//20190902 Flow Data 2.csv', sep=";"))
df3 = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//0902data//20190902 Flow Data 3.csv'))
speed = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//0902data//20190902 Speed (Volume Weighted) Data.csv'))
#6:00-6:59
df_inner2 = df2.iloc[367:427]
df_inner3 = df3.iloc[367:427]
print(df_inner2)
#提取810-SB和820SB的数据
df_inner22 = df_inner2.iloc[:, 3:6]
df_inner32 = df_inner3.iloc[:, 1:9]
print("820-818ooooooooooooooooooooooooooo")
print(df_inner22)
data22 = df_inner22.values.tolist()
print(data22)
print("817-810ppppppppppppppppppppppppppp")
print(df_inner32)
data32 = df_inner32.values.tolist()
#重新写入csv文件
def get_date_list(begin_date,end_date):
    date_list = [x.strftime('%H:%M') for x in list(pd.date_range(start=begin_date, end=end_date, freq='min'))]
    return date_list
times = get_date_list('6:00', '6:59')
col_names = [
             '820 SB',
             '819 SB',
             '818 SB',
             '817 SB',
             '816 SB',
             '815 SB',
             '814 SB',
             '813 SB',
             '812 SB',
             '811 SB',
             '810 SB']
dff = pd.DataFrame(np.zeros((mm, 11)), index=times, columns=col_names)
dff.iloc[:, 0:3] = data22
dff.iloc[:, 3:] = data32
print(dff)
dff.to_csv('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59flow0902.csv', index=times, columns=col_names, sep=";")
