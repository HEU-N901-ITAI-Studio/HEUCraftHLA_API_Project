

---

author:MesonoxianYao

title:install.md

datetime:2024.12.12 0:15

---

# 安装快速入门

---

### 安装
- 安装Visual Studio 2013/2017/2019/2022
- 安装星际争霸：母巢之战
- 更新《星际争霸：母巢之战》至1.16.1
- 安装BWAPI

### 编译
- ExampleProjects.sln在打开BWAPI安装目录
- 在 RELEASE 模式下构建 ExampleAIModule 项目(不要忘记手动配置本机kits与编译SDK)
- 复制ExampleAIModule.dll到星际争霸安装文件夹内的bwapi-data/AI

### 通过 Chaoslauncher 运行星际争霸
- Chaoslauncher.exe以管理员身份运行
    - Chaoslauncher 位于 Chaoslauncher 目录中BWAPI安装目录
- 检查BWAPI喷油器 xxx [释放]
- 单击“开始”
    - 确保版本设置为星际争霸 1.16.1，而不是 ICCup 1.16.1

### 与暴雪的 AI 进行游戏
- 进入单人游戏->扩展
- 选择任意用户并单击“确定”
- 点击“自定义游戏”，选择地图，然后开始游戏

### 与自己进行游戏
- Chaoslauncher - MultiInstance.exe以管理员身份运行
- 开始
    - 前往多人游戏->扩展->本地电脑
    - 选择任意用户并单击“确定”
    - 点击创建游戏，选择地图，点击确定
- 开始-取消选中BWAPI注射器 xxx [发布]让人类玩游戏，更不用说让人工智能自己玩了
    - 前往多人游戏->扩展->本地电脑
    - 选择任意用户并单击“确定”
    - 加入其他客户端创建的现有游戏