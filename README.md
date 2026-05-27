# 50ETF 波动率套利策略复现

基于兴业证券研报《期权波动率交易之二：基于期权复制策略的波动率套利策略》的策略复现。

## 策略概述

当 50ETF 看涨期权的隐含波动率 (IV) 高于波动率锥中对应期限的 85% 分位数时，卖出看涨期权并买入 50ETF 现货进行 Delta 中性对冲，待收益达到预期或到期时平仓获利。

**核心逻辑**：赚取隐含波动率与实际波动率之间的价差。

## 项目结构

```
VolArb_50ETF_Reproduction/
├── Data/                          # 数据存储
│   ├── 50ETF_5min.parquet         # 50ETF 5分钟行情
│   ├── 50ETF_options_5min.parquet # 期权5分钟行情
│   ├── etf_daily_510050.parquet   # 50ETF日线数据
│   ├── option_metadata.parquet    # 期权合约元数据
│   ├── options_daily.parquet      # 期权日线数据(缓存)
│   ├── options_greeks.parquet     # 带Greeks的期权数据(缓存)
│   └── processed/                 # 处理后的图表
├── src/                           # 核心代码
│   ├── data/
│   │   ├── load_data.py           # 数据加载与5分钟→日线重采样
│   │   └── preprocess.py          # 数据预处理、IV/Greeks计算
│   ├── models/
│   │   ├── bsm.py                 # Black-Scholes-Merton 定价
│   │   ├── greeks.py              # Greeks 计算 (Delta, Gamma, Vega, Theta)
│   │   ├── implied_vol.py         # 隐含波动率求解 (二分法/牛顿法)
│   │   ├── volatility.py          # 历史波动率与已实现波动率
│   │   └── vol_cone.py            # 波动率锥构建
│   ├── strategy/
│   │   ├── signal.py              # 开仓信号生成
│   │   ├── hedging.py             # Delta 对冲引擎
│   │   └── backtest.py            # 回测框架
│   └── utils/
│       └── visualize.py           # 可视化工具
├── notebooks/
│   ├── 03_backtest_phase_a.ipynb  # Phase A 回测
│   └── 04_backtest_full.ipynb     # 完整回测
└── pyproject.toml
```

## 模块说明

### `src/models` - 定价与波动率模型

| 模块 | 功能 |
|------|------|
| `bsm.py` | BSM 期权定价公式，支持标量与向量化计算 |
| `greeks.py` | Delta / Gamma / Vega / Theta 计算，含有限差分验证 |
| `implied_vol.py` | 二分法 + 牛顿法隐含波动率求解器 |
| `volatility.py` | 历史波动率 (日线) 与已实现波动率 (5分钟高频) |
| `vol_cone.py` | 滚动窗口波动率锥构建，支持自定义窗口与分位数 |

### `src/data` - 数据处理

| 模块 | 功能 |
|------|------|
| `load_data.py` | Parquet 数据加载，5分钟→日线 OHLCV 重采样 |
| `preprocess.py` | 期权数据清洗、ETF/期权对齐、批量 IV 与 Greeks 计算 |

### `src/strategy` - 策略逻辑

| 模块 | 功能 |
|------|------|
| `signal.py` | 基于 IV 与波动率锥阈值的开仓信号 |
| `hedging.py` | `Trade` 类：逐日 Delta 再平衡、P&L 跟踪、平仓逻辑 |
| `backtest.py` | `VolArbBacktest` 回测引擎，支持滚动阈值、提前平仓、费用建模 |

## 策略参数

| 参数 | 值 |
|------|-----|
| 标的 | 50ETF (510050) |
| 期权类型 | 看涨期权 (Call) / 看跌期权 (Put) |
| 开仓阈值 | IV > 波动率锥 20 日 85% 分位数 |
| 对冲方式 | Delta 中性，逐日再平衡 (使用预测波动率) |
| 提前平仓条件 | 组合收益 >= 预期收益 |
| 现货手续费 | 万分之五 |
| 无风险利率 | 4% |
| 合约乘数 | 10000 |

## 快速开始

```bash
# 安装依赖
uv sync

# 启动 notebook
uv run jupyter notebook notebooks/
```

## 依赖

- Python >= 3.11
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.11
- matplotlib >= 3.7
- pyarrow >= 24.0
