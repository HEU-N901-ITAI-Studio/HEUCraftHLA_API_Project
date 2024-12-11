---

author:MesonoxianYao
title:example.md
datetime:2024.12.11 22:14

---

### BWAPI基础框架

下面提供了一个BWAPI的机器人的框架,其通过编译.dll来作用.


ExampleAIModule.cpp:
```cpp
#include "ExampleAIModule.h"
#include <iostream>

using namespace BWAPI;
using namespace Filter;

void ExampleAIModule::onStart()
{
  // Hello World!
  Broodwar->sendText("Hello world!");

  // 输出地图名字
  Broodwar << "The map is " << Broodwar->mapName() << "!" << std::endl;

  // 设置用户标志,使得用户能够进行操作
  Broodwar->enableFlag(Flag::UserInput);

  // [不推荐]:下面的这一条指令会令AI知道地图所有信息
  //Broodwar->enableFlag(Flag::CompleteMapInformation);

  // 设置指令优化级别,以降低AI的APM以及对常见指令分组
  Broodwar->setCommandOptimizationLevel(2);

  // 如果是回放
  if ( Broodwar->isReplay() ){
    // 提醒玩家正在回放
    Broodwar << "The following players are in this replay:" << std::endl;
    
    // 遍历所有玩家
    Playerset players = Broodwar->getPlayers();
    for(auto p : players){
      // 仅展示非Observer玩家
      if ( !p->isObserver() )
        Broodwar << p->getName() << ", playing as " << p->getRace() << std::endl;
    }
  }
  // 如果不是一个回放
  else {
    // 检索你和你敌人的比赛,enemy()返回第一个敌人
    // 如果你想处理多个敌人,使用enemies()返回所有
    if ( Broodwar->enemy() ) // 先确保存在敌人
      Broodwar << "The matchup is " << Broodwar->self()->getRace() 
	      << " vs " << Broodwar->enemy()->getRace() << std::endl;
  }
}

void ExampleAIModule::onEnd(bool isWinner)
{
  // 当比赛结束时调用
  if ( isWinner )
  {
    // 在这里处理你的胜利
  }
}

void ExampleAIModule::onFrame()
{
  // 每一个游戏帧时处理一次
  // 在游戏左上方展示游戏帧率
  Broodwar->drawTextScreen(200, 0,  "FPS: %d", Broodwar->getFPS() );
  Broodwar->drawTextScreen(200, 20, "Average FPS: %f", Broodwar->getAverageFPS() );

  // 路过游戏是录像或者被暂停,返回
  if ( Broodwar->isReplay() || Broodwar->isPaused() || !Broodwar->self() )return;

  // 防止由于延迟带来的多余帧带来延迟
  // 延迟帧是指令被执行前处理的帧
  if ( Broodwar->getFrameCount() % Broodwar->getLatencyFrames() != 0 )return;

  // 遍历我们拥有的所有单位
  for (auto &u : Broodwar->self()->getUnits()){
    // 忽略已经不存在的单位
    // [警告]在处理单位指针时确保单位是否存在
    if ( !u->exists() )continue;

    // 忽略处于无法操作状态的单位
    if ( u->isLockedDown() || u->isMaelstrommed() || u->isStasised() )continue;

    // 忽略处理下面状态的单位
    if ( u->isLoaded() || !u->isPowered() || u->isStuck() )continue;

    // 忽略未完成或者正在建造的单位
    if ( !u->isCompleted() || u->isConstructing() )continue;

    // 现在开始单位具体行为
    // 如果单位是一个工人单位
    if ( u->getType().isWorker() ){
      // 如果该工人单位空闲
      if ( u->isIdle() ){
        // 指令工人单位采集一些资源并且将其返回基地中心
        // 或者找到一条收获资源的路径
        if ( u->isCarryingGas() || u->isCarryingMinerals() ){
          u->returnCargo();
        }
        else if ( !u->getPowerUp()){
	      // 如果工人不是已经采集了其他powerup,比如flag
          // 采集矿物或瓦斯从最近的资源点
          if ( !u->gather( u->getClosestUnit( IsMineralField || IsRefinery ))){
            // 如果调用失败,返回最近的一条错误消息
            Broodwar << Broodwar->getLastError() << std::endl;
          }
        } // closure: has no powerup
      } // closure: if idle
    }
    // 一个资源depot是一个人族或虫族或神族的指挥中心
    else if ( u->getType().isResourceDepot() ) {
      // 指令depot制造更多的工人单位,但是仅在空闲时间
      if ( u->isIdle() && !u->train(u->getType().getRace().getWorker())){
        // 如果失败了返回错误信息以了解更多信息
        // 然而这个帧正常只会显示一帧
        // 因而创建一个事件来使得其显示更多帧
        Position pos = u->getPosition();
        Error lastErr = Broodwar->getLastError();
        Broodwar->registerEvent(
	        [pos,lastErr](Game*){ 
		        Broodwar->drawTextMap(pos, "%c%s", 
				    Text::White, 
					lastErr.c_str()
				); 
			},   // 行为
            nullptr,    // 条件
            Broodwar->getLatencyFrames()    // 将会执行的帧
        );  
        // 在供给用光的时候获取种族的供给提供者类型
        UnitType supplyProviderType = u->getType().getRace().getSupplyProvider();
        static int lastChecked = 0;

        // 如果供给受限且一定时间内并没有建造供给提供者
        if (  lastErr == Errors::Insufficient_Supply &&
              lastChecked + 400 < Broodwar->getFrameCount() &&
              Broodwar->self()->incompleteUnitCount(supplyProviderType) == 0 )
        {
          lastChecked = Broodwar->getFrameCount();

          // 检索一个能够构建供给者的单位
          Unit supplyBuilder = u->getClosestUnit(
	          GetType==supplyProviderType.whatBuilds().first &&
	          (IsIdle || IsGatheringMinerals) &&
              IsOwned);
          // 如果一个单位被找到
          if ( supplyBuilder ){
            if ( supplyProviderType.isBuilding() ){
              TilePosition targetBuildLocation = 
	              Broodwar->getBuildLocation(
		              supplyProviderType, 
		              supplyBuilder->getTilePosition()
		          );
              if ( targetBuildLocation ){
                // 注册一个事件画出建设供给者的区域
                Broodwar->registerEvent(
	                [targetBuildLocation,supplyProviderType](Game*){
	                    Broodwar->drawBoxMap(Position(targetBuildLocation),
                        Position(targetBuildLocation+
	                        supplyProviderType.tileSize()),
                        Colors::Blue);
                    },
                    nullptr,  // 条件
                    supplyProviderType.buildTime() + 100 
                );  // 运行的帧
                // 指令建筑者去建造供给者
                supplyBuilder->build( supplyProviderType, targetBuildLocation );
              }
            }
            else{
              // 训练供给者,如果供给者不是一个建筑物,比如虫族
              supplyBuilder->train( supplyProviderType );
            }
          } // closure: supplyBuilder is valid
        } // closure: insufficient supply
      } // closure: failed to train idle unit
    }
  } // closure: unit iterator
}

void ExampleAIModule::onSendText(std::string text)
{
  // 发送文本给程序,当其没有被进行时.
  Broodwar->sendText("%s", text.c_str());
  // 请确保使用%s与c_str()的组合
  // 否则你将会在使用%(percent)时遇到麻烦
}

void ExampleAIModule::onReceiveText(BWAPI::Player player, std::string text)
{
  // 解析收到的文本
  Broodwar << player->getName() << " said \"" << text << "\"" << std::endl;
}

void ExampleAIModule::onPlayerLeft(BWAPI::Player player)
{
  // 告知玩家离开
  Broodwar->sendText("Goodbye %s!", player->getName().c_str());
}

void ExampleAIModule::onNukeDetect(BWAPI::Position target)
{
  // 确认目标地址合法
  if ( target ){
    // 如果这样,输出提醒
    Broodwar << "Nuclear Launch Detected at " << target << std::endl;
  }
  else {
    // 否则询问玩家们核弹在哪
    Broodwar->sendText("Where's the nuke?");
  }
  // 你也可以通过Broodwar->getNukeDots()获得所有的核弹信息
}

void ExampleAIModule::onUnitDiscover(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitEvade(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitShow(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitHide(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitCreate(BWAPI::Unit unit)
{
  if ( Broodwar->isReplay() ){
    // 如果我们正在回放,那么打印出所有的建造队列
    if ( unit->getType().isBuilding() && !unit->getPlayer()->isNeutral() ){
      int seconds = Broodwar->getFrameCount()/24;
      int minutes = seconds/60;
      seconds %= 60;
      Broodwar->sendText("%.2d:%.2d: %s creates a %s", 
	      minutes, 
	      seconds, 
	      unit->getPlayer()->getName().c_str(), 
	      unit->getType().c_str()
	  );
    }
  }
}

void ExampleAIModule::onUnitDestroy(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitMorph(BWAPI::Unit unit)
{
  if ( Broodwar->isReplay() ){
    // 如果我们正在回放,那么打印出所有的建造队列
    if ( unit->getType().isBuilding() && !unit->getPlayer()->isNeutral()){
      int seconds = Broodwar->getFrameCount()/24;
      int minutes = seconds/60;
      seconds %= 60;
      Broodwar->sendText("%.2d:%.2d: %s morphs a %s", minutes,
	      seconds, 
	      unit->getPlayer()->getName().c_str(), 
	      unit->getType().c_str()
	  );
    }
  }
}

void ExampleAIModule::onUnitRenegade(BWAPI::Unit unit)
{
}

void ExampleAIModule::onSaveGame(std::string gameName)
{
  Broodwar << "The game was saved to \"" << gameName << "\"" << std::endl;
}

void ExampleAIModule::onUnitComplete(BWAPI::Unit unit)
{
}

```
