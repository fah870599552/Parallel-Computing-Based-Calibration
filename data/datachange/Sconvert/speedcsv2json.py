import sys, json
import pandas as pd
tip = """
请确保：
1. CSV格式是UTF-8
2. CSV第一行是键值

用法：
python csv2json.py foobar.csv
其中foobar.csv是需要转换的源数据文件

运行环境：
Python 3.4.3

日期：
2015年12月29日
"""
print(tip)
# 获取输入数据
input_file = "D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59speed0902.csv"
#input_file = sys.argv[1]
lines = open(input_file, "r", encoding="utf_8_sig").readlines()
lines = [line.strip() for line in lines]
data = pd.DataFrame(pd.read_csv('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59speed0902.csv', sep=";"))
data = data.to_dict()
mm = 60
# 获取键值
keys = lines[0].split(';')
l = len(keys)
line_num = 1
total_lines = len(lines)

dic = {}
for j in range(l):
    dic[keys[j]] = list()
for i in range(mm):
    values = lines[i+1].split(";")
    dic[keys[0]].append(values[0])
    for k in range(l-1):
        dic[keys[k+1]].append(float(values[k+1]))
print(dic)
json_str = json.dumps(dic, ensure_ascii=False, indent=4)
output_file = input_file.replace("csv", "json")

# write to the file
f = open(output_file, "w", encoding="utf-8")
f.write(json_str)
f.close()

print("解析结束！")
with open('D://graduate1//SUMMER//Calibration//data//datachange//Newdata//6.00-6.59speed0902.json', 'r') as load_f:
    tflow = json.load(load_f)
print(tflow)