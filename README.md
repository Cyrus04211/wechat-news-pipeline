# 公众号行业调研消息采集 Pipeline

从微信公众号研究源采集行业调研分析消息，使用 LLM 进行摘要、主题分类，并生成日报/周报。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
playwright install chromium   # 仅首次登录需要

# 2. 配置 LLM API
export CLOSEAI_API_KEY="your-key"

# 3. 登录微信公众号后台
python scripts/wechat_mp_login.py        # 扫码登录，保存会话

# 4. 配置公众号源（如果首次使用）
python scripts/wechat_mp_discover_fakeids.py --update-yaml
```

## 运行

### 全流程（采集 → 分类 → 报告）

```bash
# 日报
python src/pipelines/research_workflow.py --mode collect-classify-report --collection-mode daily_brief

# 周报
python src/pipelines/research_workflow.py --mode collect-classify-report --collection-mode weekly_review

# 仅基于已有数据重生成报告
python src/pipelines/research_workflow.py --mode report-only --collection-mode weekly_review
```

### 定时采集

```bash
python -m src.pipelines.news_scheduler --daemon --mode daily_brief
python -m src.pipelines.news_scheduler --daemon --mode weekly_review
```

## 流程概览

1. 微信公众号源采集文章列表 → 2. 去重、时效过滤、噪声过滤 → 3. 正文抓取 → 4. LLM 摘要（事件名 + 摘要 + 噪声判断）→ 5. LLM 主题分类 → 6. 生成日报/周报

## 项目结构

```
├── config/
│   ├── news_sources.yaml       # 公众号源配置（账号、fakeid）
│   ├── news_runtime.yaml       # 采集窗口、LLM 参数
│   └── research_runtime.yaml   # 报告输出配置
├── src/
│   ├── collectors/             # 采集模块（去重、LLM 处理、正文抓取）
│   ├── sources/                # 数据源适配器（微信公众号等）
│   ├── pipelines/              # 采集/报告管线
│   ├── research/               # 研报生成模块
│   ├── agents/                 # LLM Agent 封装
│   ├── domain/                 # 领域模型
│   ├── config/                 # 配置加载
│   └── utils/                  # 工具函数
└── templates/reports/          # 报告模板（Jinja2）
```

## 配置

| 文件 | 说明 |
|------|------|
| `config/news_sources.yaml` | 公众号源列表、账号名、fakeid |
| `config/news_runtime.yaml` | 采集窗口、正文抓取、LLM 并发与调度参数 |
| `config/research_runtime.yaml` | 报告输出路径与默认窗口 |

采集模式：

| 模式 | 采集窗口 | 报告类型 |
|------|----------|----------|
| `daily_brief` | 24h | 日报 |
| `weekly_review` | 168h | 周报 |

## 关键输出

| 路径 | 说明 |
|------|------|
| `data/news/events_classified.csv` | 事件表（经 LLM 处理） |
| `data/news/reviewed_all.csv` | 审计表 |
| `data/news/research/reports/` | 日报/周报 |
