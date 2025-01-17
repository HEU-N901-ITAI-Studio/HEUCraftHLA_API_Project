# 1. UnitType 单位类型类
## 1.1 构造函数与析构函数
### 1.1.1 constexpr BWAPI::UnitType::UnitType (int id = UnitTypes::Enum::None)
如果类型是无效类型，则变为Types::Unknown。

如果一个类型的值小于0或大于Types::Unknown，则该类型无效。


## 1.2 成员函数
### 1.2.1 Race BWAPI::UnitType::getRace () const 
获取单位对应种族
> "Race" 指代种族类型
> Race::NONE 不指代任何种族（例如小动物）

### 1.2.2 int BWAPI::UnitType::width () const

