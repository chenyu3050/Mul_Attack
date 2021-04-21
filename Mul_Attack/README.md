### base on invert mul_attack  of dlg

#### dir as following:
- MUL_ATTACK:
  - inversefed
    - data
    - nn
    - traing
    - *.py
  - scripts
    - clean.sh
  - mul_attack.py
  - README.md
#### 代码执行说明：
 ./scripts/clean.sh 即可删除output文件夹
 
#### 重要参数说明：
- init:randn,rand,zeros
- optim:adam,sgd,LBFGS
 


#### 攻击模式：
- 单张：iDLG
- 多张：DLG


