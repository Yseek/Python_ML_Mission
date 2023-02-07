import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

df=pd.read_csv("국립암센터_24개종 암발생률_20221228.CSV",encoding='ANSI')
x=[]
y=[]
for (k1,k2,k3,k4), group in df.groupby(['발생연도','성별','발생자수','암종']):
    if k2=='남녀전체':
        if k4=='모든암':
            x.append(k1)
            y.append(k3)

#차팅
'''
plt.scatter(x,y)
plt.show()
'''

a=np.polyfit(x,y,3)
model=np.poly1d(a)
line = np.linspace(1999, 2020, 1000)
plt.rcParams['font.family'] = 'gulim'
plt.scatter(x,y)
plt.plot(line,model(line))
plt.xlabel('연도')
plt.ylabel('발생자수')

plt.show()

r2 = r2_score(y, model(x)) # 0(전혀관계X) <= r2 <= 1(100%관계O)
print("r2:", r2) #0.9719987096554549

# 예측
predict = model(1950)
print(predict)