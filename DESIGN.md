# BenchFlow Fast 改造方案

状态：设计阶段，尚未开始修改代码  
日期：2026-08-02

## 1. 项目命名

- 项目名、GitHub 仓库名、Python distribution 名：`benchflow-fast`
- 项目位置：https://github.com/zhangle2088/benchflow-fast.git(已初始化)
- 主命令：`bff`
- 完整命令别名：`benchflow-fast`
- 上游兼容命令 `bench`：默认不注册，避免覆盖官方 BenchFlow

典型命令应保持与 BenchFlow 对应：

```bash
bff eval run \
  --tasks-dir tasks/offer-letter-generator \
  --agent codex-acp \
  --model gpt-5.4-mini \
  --skill-mode with-skill \
  --skills-dir tasks/offer-letter-generator/environment/skills \
  --sandbox apple-container
```


## 2. 背景和问题

BenchFlow 已经较完整地实现了：

- `task.md` 任务定义和校验；
- Agent registry；
- Codex、Claude 等 Agent 的 ACP 启动和通信；
- API Key、Codex `auth.json`、Claude 登录信息等认证注入；
- oracle、Agent、verifier 的执行流程；
- with-skill、no-skill 和其他 skill mode；
- timeout、并发、错误分类；
- trajectory、日志、reward、summary 和 artifacts 保存；
- Docker、Apple Container、Modal、Daytona 等 sandbox provider。

本地高频实验中，现有环境准备方式存在明显的重复成本：

1. 大多数 task 会触发 task Dockerfile build；
2. task 输入文件和 skill 可能进入 build context 和镜像 layer；
3. 新容器中会再次安装 Agent；
4. 容器结束后删除，容器内的 Agent 安装结果不能复用；
5. 即使命中 image layer cache，仍需准备 build context 和调用构建流程；
6. 一次性 task 数据形成镜像内容，不利于存储、清理和敏感数据管理。

skill-inject 的可取之处是预先构建 Agent 镜像，运行时通过 bind mount 注入 workspace。本项目计划在保留 BenchFlow 评测语义的前提下采用这一思路。

## 3. 项目目标

与benchflow主要区别（改进）：

* 增加通用镜像，各task启动容器，在容器中进行必要setup，而不是各自build镜像（需要增加build通用镜像环节）
* 将workspace通过mount进容器，而不是在镜像中build进去（需要增加构建workspce环节）

### 3.1 必须实现

1. Fork BenchFlow 并持续兼容上游；
2. 保留原有 task、Agent ACP、认证、verifier 和结果格式；
3. 新增通用镜像加 workspace 挂载的快速环境准备模式；
4. Agent 预装于通用镜像，不在每个临时容器中重复安装；
5. task 输入、skills、临时认证和输出通过运行时 materialize、mount 或 upload 注入；
6. task 特有准备通过临时容器中的 setup 命令或 `setup.sh` 完成；
7. 不为普通 task 生成持久化 task image；
8. 对不能使用快速模式的 task 自动或显式回退到原始 build 模式；
9. 在结果中记录镜像 digest、环境策略、Agent 版本和 setup 信息；
10. 支持本机安装，并通过 GitHub 供项目组合作者使用。

### 3.2 暂不追求

1. 第一阶段不重写 ACP 通信层；
2. 第一阶段不改变 verifier 语义；
3. 第一阶段不覆盖所有 cloud provider；
4. 第一阶段不保证 mounted 模式的成绩与官方 build 模式完全等价；
5. 第一阶段不自动把任意复杂 Dockerfile 无损转换为 shell setup；
6. 第一阶段不移除原始 BenchFlow build 路径。

## 4. 兼容性定义

“兼容 BenchFlow”分为四个层级。

### 4.1 CLI 兼容

以下命令结构和主要参数保持一致：

```text
bff tasks ...
bff eval run ...
bff skills ...
bff sandbox ...
```

新增参数不能改变未指定参数时的上游默认行为。

### 4.2 Task 兼容

继续读取原生 BenchFlow/SkillsBench task：

```text
tasks/<task-id>/
  task.md
  environment/
    Dockerfile
    skills/
  oracle/
  verifier/
```

原有 task 无需修改即可继续通过 build 模式运行。mounted 模式可使用可选的快速运行描述或本地 overlay。

### 4.3 执行和结果兼容

保留：

- rollout 生命周期；
- ACP session；
- prompt 发送；
- trajectory 和 usage 采集；
- verifier；
- `jobs/` 目录组织；
- `result.json`、`reward.txt`、`summary.json` 等核心 artifacts。

新增字段只能作为扩展，例如：

```json
{
  "provisioning": "mounted",
  "runtime_image": "ghcr.io/example/benchflow-fast-acp@sha256:...",
  "runtime_profile": "documents-v1",
  "setup_hash": "sha256:..."
}
```

### 4.4 上游行为兼容

默认保留严格模式：

```text
provisioning=build
```

需要官方可比性或遇到特殊 task 时，始终可以使用原始 Dockerfile build 路径。

## 5. 总体架构

```text
                   BenchFlow upstream
                           │
                           │ 定期 merge
                           ▼
                    benchflow-fast
                           │
          ┌────────────────┴────────────────┐
          │                                 │
    原始评测控制层                    新增环境策略层
 task / ACP / auth / verifier       build / mounted / auto
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                 Docker / Apple Container
                           │
                           ▼
                 相同 jobs/result 输出层
```

核心原则：

> 镜像保存稳定软件，挂载保存单次实验状态，setup 处理 task 特有准备，BenchFlow 继续负责评测控制。

## 6. Provisioning 策略

计划新增与 sandbox provider 正交的配置：

```text
build
mounted
auto
```

### 6.1 build

- 完全保留上游 BenchFlow 行为；
- 使用 task Dockerfile；
- 适合官方复现、特殊系统环境、多服务和无法安全转换的 task。

### 6.2 mounted

- 使用预构建通用 ACP 镜像；
- 不构建 task image；
- 创建临时 host workspace；
- 将允许暴露的 task 输入 materialize 到 workspace；
- 注入临时 Agent home 和 skills；
- bind mount 到容器；
- 执行 task setup；
- 运行 ACP Agent；
- Agent 结束后再执行 verifier；
- 删除临时容器和临时 workspace；
- 保留 jobs artifacts。

### 6.3 auto

决策顺序：

1. task 明确声明 mounted-compatible：使用 mounted；
2. 本地 overlay 声明 mounted-compatible：使用 mounted；
3. task 所需 runtime profile 可用且通过能力检查：使用 mounted；
4. 其他情况：记录原因并回退 build。

不应在能力未知时静默运行 mounted，以免产生看似成功但语义不一致的结果。

## 7. 通用镜像设计

通用镜像应保存稳定、跨 task 复用的内容：

- Ubuntu 基础环境；
- Python、Node、Git、curl、证书；
- BenchFlow ACP 运行需要的目录、用户和 shell；
- `codex-acp`；
- `claude-agent-acp`；
- `opencode`；
- 常用、版本固定的 Python/npm 工具；
- LiteLLM/ACP 运行所需依赖；
- 非 root sandbox user（如采用）；
- 健康检查和版本清单。

第一阶段可以先维护一个 Apple Silicon 使用的：

```text
linux/arm64
```

之后通过 GitHub Actions 发布：

```text
linux/arm64
linux/amd64
```

镜像应同时提供 tag 和不可变 digest。正式实验优先按 digest 运行。

通用镜像不应包含：

- task 输入数据；
- oracle；
- verifier；
- task 专用 skill；
- 用户真实认证文件；
- 宿主机 `.codex` 或 `.claude` 目录；
- 无法解释来源和版本的大量工具。

未来如果一个镜像过大，可以使用共享 core layer 的 profiles：

```text
benchflow-fast-core
├── benchflow-fast-documents
├── benchflow-fast-data
├── benchflow-fast-browser
└── benchflow-fast-security
```

## 8. Workspace 和挂载设计

每次 rollout 创建独立目录。由于项目位于 SynologyDrive，运行时 workspace 优先放在本机临时磁盘：

```text
/private/tmp/benchflow-fast/<run-id>/
  workspace/
  agent-home/
  input/
  cache/
  logs/
  artifacts/
```

任务结束后将需要保留的内容复制到 `jobs/`，然后清理临时目录。

建议的可见性：

| 内容 | Agent 阶段 | Verifier 阶段 | 挂载方式 |
|---|---:|---:|---|
| workspace | 读写 | 读取 | bind mount |
| task input | 只读或复制后隐藏 | 可选读取 | filtered staging |
| skills | with-skill 可见 | 通常不需要 | 独立挂载/上传 |
| auth | 必需时可见 | 不可见 | 临时文件 |
| oracle | 不可见 | 不可见 | 不挂载 |
| verifier | 不可见 | 可见 | 验证阶段单独注入 |
| jobs/logs | 仅指定目录 | 指定目录 | 独立挂载 |

禁止直接挂载整个 task 根目录，因为其中包含 `oracle/`、`verifier/` 和可能在 no-skill 实验中泄漏的 `environment/skills/`。

## 9. Agent 认证与临时 Home

不直接把宿主机完整的 `~/.codex`、`~/.claude` 挂载到容器。

每次 rollout 创建临时 Agent home：

```text
agent-home/
  .codex/
    auth.json
    config.toml
    skills/
  .claude/
    ...
```

原则：

1. 复用 BenchFlow 现有 credential 解析和上传逻辑；
2. 只复制本次 Agent 必需的认证文件；
3. no-skill 的 skills 目录为空；
4. with-skill 只注入本次指定 skills；
5. 临时 home 可写，但不会写回宿主机真实 home；
6. artifacts 和日志不得保存认证内容；
7. 容器结束后删除临时认证文件。

## 10. ACP Agent 保持不变

Agent 的运行协议仍与 BenchFlow 一致：

```text
resolve agent config
→ prepare model proxy and environment
→ write credentials
→ start ACP process
→ connect ACP session
→ execute prompts
→ collect trajectory and usage
```

需要修改的只是“是否安装 Agent binary”。

现有 `skip_install` 会将 `_agent_cfg` 设为 `None`。这会把“跳过二进制安装”和“跳过 Agent 配置解析”错误地绑定，可能影响 credential files、skill paths 和 Agent policy。

目标语义应为：

```text
skip binary installation = true
resolve AgentConfig = always
```

即使 Agent 已预装，也必须继续解析完整 AgentConfig，并复用 BenchFlow 的认证、skill 分发、Web policy 和 ACP launch 配置。

通用镜像中的 Agent 路径和版本必须与 Agent registry 的预期一致。启动前执行快速 version/which probe，失败时给出明确错误，而不是运行时才发现。

## 11. Task setup 设计

mounted 模式允许 task 在临时容器中执行独特 setup，而不生成持久化 task image。

推荐支持：

1. 现有 `sandbox.setup_commands`；
2. 可选 `environment/setup.sh`；
3. 项目本地 overlay 中的 setup；
4. setup timeout、工作目录、用户、环境变量和日志；
5. setup 失败时停止 Agent 执行。

setup 适合：

- 将输入文件放到任务要求的绝对路径；
- 安装少量 task 特有 Python/npm 包；
- 初始化目录、权限、配置文件；
- 初始化简单数据库；
- 启动简单后台进程并执行 healthcheck。

setup 不适合或需要回退 build：

- 特殊 Ubuntu/Python/Node 基础版本；
- 多容器 Docker Compose；
- GPU/CUDA 基础镜像；
- 特殊 entrypoint 或系统服务；
- 内核能力、设备或复杂网络拓扑；
- 与通用镜像存在依赖冲突；
- setup 时间接近或超过直接构建缓存镜像的成本。

所有 setup 依赖应锁定版本，输出完整日志，并记录 setup 文件 hash。

## 12. 缓存策略

mounted 模式不保存 task image，但 setup 可能仍需下载依赖。可选共享缓存：

- uv/pip cache；
- npm cache；
- Hugging Face 或模型 cache；
- task 明确声明的只读数据 cache。

缓存必须满足：

1. 不存储认证；
2. 不包含上次 task workspace 或输出；
3. 使用 runtime profile、架构、lockfile hash 等生成 cache key；
4. 并发运行安全；
5. 可以通过 `--no-cache` 禁用；
6. 结果中记录是否命中缓存。

## 13. Verifier 隔离

Agent 阶段不得读取 verifier 实现。

推荐流程：

1. Agent 容器只获得 workspace、输入、临时 auth 和允许的 skills；
2. Agent 完成或超时；
3. 停止 Agent ACP 进程并清理认证；
4. verifier 再被上传或挂载到隔离路径；
5. 执行 BenchFlow 原有 hardening；
6. verifier 读取 workspace，写入 reward 和日志；
7. 收集结果。

如果为了兼容而在同一容器执行 verifier，也必须保证 verifier 在 Agent 阶段尚不可访问。

## 14. CLI 设计

### 14.1 保留原命令结构

```bash
bff eval run --tasks-dir ... --agent ... --sandbox ...
```

### 14.2 新增参数建议

```text
--provisioning build|mounted|auto
--runtime-image <image-ref>
--runtime-profile <profile-name>
--setup-mode task|overlay|none
--no-setup-cache
--strict-compat
```

默认建议：

```text
官方兼容默认：build
项目组本地配置默认：auto
明确的快速实验：mounted
```

不要把 mounted 隐式设为所有用户的唯一模式。默认值可由用户配置文件覆盖，但结果必须记录实际选择的策略。

### 14.3 简化高频命令

后续可增加 profile/config preset，而不是不断增加必填参数：

```bash
bff run offer-letter-generator --preset codex-with-skill
bff run offer-letter-generator --preset codex-no-skill
```

它们应只是 `bff eval run` 的薄包装，最终落到相同 EvaluationConfig 和结果格式，避免形成第二套评测逻辑。

第一阶段优先实现兼容的 `bff eval run`，简写命令放在后续阶段。

## 15. 与上游同步和独立维护

建议 Git remotes：

```text
origin    项目组 GitHub 的 benchflow-fast
upstream  github.com/benchflow-ai/benchflow
```

建议策略：

1. 保留完整 BenchFlow Git 历史；
2. 定期 `fetch upstream`；
3. 将 `upstream/main` merge 到项目主分支；
4. 不重写已经由合作者使用的公开主分支历史；
5. 将自定义代码集中在新模块和少数 hook，减少 merge conflict；
6. 通用改进尽量向上游提交；
7. 每个 release 记录对应的 upstream commit；
8. CI 同时运行上游测试和 mounted 模式测试。

许可证和发布要求：

- 保留 Apache-2.0 LICENSE；
- 保留原项目版权和 attribution；
- 修改的文件明确标注；
- README 明确说明是独立派生项目，不是 BenchFlow 官方发行版；
- GitHub Release 同时记录 BenchFlow 上游版本和通用镜像 digest。

安装方式第一阶段可以是：

```bash
uv tool install git+https://github.com/<org>/benchflow-fast.git
```

后续再决定是否发布到 PyPI。Python distribution 使用 `benchflow-fast`，避免与上游 `benchflow` 包名直接混淆。

## 16. 预期代码改动边界

计划阶段不改代码。正式实施时，尽量限制在以下范围：

1. CLI/EvaluationConfig：增加 provisioning 和 runtime image 配置；
2. rollout setup：增加 workspace materialization 和 mounted 生命周期；
3. Docker/Apple Container sandbox：支持受控 bind mounts；
4. Agent install：拆分 AgentConfig resolve 与 binary install；
5. result metadata：记录实际环境策略；
6. capability/fallback：判断 task 是否支持 mounted；
7. 新增测试、文档和通用镜像构建文件。

以下部分原则上不改：

- ACP protocol 和 session 驱动；
- prompt 执行；
- verifier 评分逻辑；
- SkillsBench task 规则；
- oracle 行为；
- trajectory 核心格式。

## 17. 分阶段实施计划

### Phase 0：设计和基线

- 固化本文档；
- 记录 BenchFlow 上游 commit；

### Phase 1：最小可用版本

- Fork BenchFlow；
- 注册 `bff` 和 `benchflow-fast` 命令；
- 构建 arm64 通用 Codex ACP 镜像；
- 支持 Apple Container；
- 支持单容器、文件型 task；
- 支持 workspace materialization；
- 修正预装 Agent 的 AgentConfig 解析；
- 支持 with-skill/no-skill；
- 使用 `offer-letter-generator` 验证。

### Phase 2：setup 和自动回退

- 支持 task/overlay setup；
- 支持 setup cache；
- 实现 mounted capability check；
- 实现 `auto` 和 mounted→build fallback；
- 增加 Docker provider。

### Phase 3：协作发布

- GitHub Actions；
- GHCR multi-arch 镜像；
- GitHub Release；
- 安装和登录文档；
- 上游同步流程；
- 项目组共享 presets。

### Phase 4：扩展

- Claude ACP 镜像；
- runtime profiles；
- 更多 task 类型；
- 简写命令 `bff run`；
- 评估是否向 BenchFlow 上游贡献 plugin/provider hook。

## 18. 第一阶段验收标准

以 `offer-letter-generator` 为第一条测试链路：

1. 原始 `bench` build 模式仍可运行；
2. `bff` build 模式结果与上游一致；
3. `bff` mounted 模式不调用 task image build；
4. mounted 模式不在容器启动后安装 Codex ACP；
5. with-skill 能注入指定 skill；
6. no-skill 看不到 task bundled skills；
7. Agent 看不到 oracle 和 verifier；
8. `auth.json` 由 BenchFlow 逻辑写入临时 home；
9. ACP trajectory 和 verifier reward 正常生成；
10. jobs 结构与 BenchFlow 兼容；
11. 结果记录 provisioning、镜像 digest 和 Agent 版本；
12. 第二次运行的环境准备耗时明显低于原始方式。

## 19. 风险和权衡

### 19.1 公平性

通用镜像可能提供原 task 没有的工具，导致 mounted 成绩与官方成绩不可直接比较。结果必须标记环境 profile；严格比较使用 build 模式。

### 19.2 依赖冲突

一个大镜像无法满足所有 task。优先使用少量 profiles，不追求无边界的“大而全”。

### 19.3 setup 重复时间

不构建 task image 意味着 setup 每次重新运行。通过预装常用依赖和安全缓存平衡启动速度。

### 19.4 挂载泄漏

错误挂载可能暴露 oracle、verifier 或 no-skill 中的 skills。必须使用 filtered staging，而不是挂载 task 根目录。

### 19.5 上游冲突

直接修改现有 provider 大段逻辑会增加 merge 成本。优先增加独立 materializer、provisioning strategy 和小型 hook。

### 19.6 认证安全

临时容器拥有模型认证且可能允许公网访问。只注入最少认证材料，不挂载完整用户目录，不在日志中输出认证内容。

## 20. 尚待实施前确认的问题

1. 第一版只支持 Codex ACP，还是同时包含 Claude ACP；（两个加opencode）
2. 通用镜像是否先只发布 arm64；（是）
3. build 是否作为默认策略，还是本地配置将默认覆盖为 auto；（默认采用mount模式）
4. 是否在第一版加入共享 pip/uv/npm cache；
5. Python distribution 是否立即更名，还是第一版先作为 BenchFlow fork 安装；（先不改名）

当前建议答案：第一版使用 Codex ACP、Apple Container、arm64、外部 overlay、默认 build 但本地 preset 使用 mounted；跑通后再扩展。
