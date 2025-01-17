# Module模块
---
## 1.构造函数与析构函数
### 1.1 BWAPI\::AIModule::AIModule()
---
## 2.成员函数
### 2.1 BWAPI\::AIModule::onStart()
只在游戏开始时被调用一次，目的是让AI模块在游戏开始时进行数据初始化
> 警告：在调用此函数之前使用Broodwar接口可能会产生未定义的行为并导致bot崩溃。
（例如，在类的静态初始化期间）


### 2.2 BWAPI\::AIModule::onEnd(bool isWinner)
只在游戏结束时被调用一次

isWinnner: 布尔值变量，决定当前玩家是否赢得本场比赛，在获胜时值为==true==，在失败或游戏是录像时值为==false==

### 2.3 BWAPI\::AIModule::onFrame()
在每个逻辑帧中被调用一次

> 用户将几乎所有代码放在这个函数中