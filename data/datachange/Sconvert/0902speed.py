import pandas as pd
import numpy as np
speed = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//0902data//20190902 Speed (Volume Weighted) Data.csv'))
#提取实验时间段的数据8：00-9.59的数据,489-608#第二次修改为5.30-9：29#3:00-11.59#5:00-9.59#6:00-6:59
speed1 = speed.iloc[364:424]
print(speed1)
#提取810-SB和820SB的数据
speed2 = speed1.iloc[:, 21:32]
print(speed2)
datas = speed2.values.tolist()
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
dff = pd.DataFrame(np.zeros((60, 11)), index=times, columns=col_names)
dff.iloc[:, :] = datas
print(dff)
dff.to_csv('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59speed0902.csv', index=times, columns=col_names, sep=";")
