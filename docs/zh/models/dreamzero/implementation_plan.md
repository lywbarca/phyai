---
title: "DreamZero 接入设计与实现计划"
description: "DreamZero 在 PhyAI 中的多卡并行设计、组件拆分和分阶段实现计划"
---

# DreamZero 接入设计与实现计划

本文档汇总 DreamZero 在 PhyAI 中的第一版接入设计。目标不是一次性完成全部端到端功能，而是先把显存瓶颈最大的 DiT 和 KV cache 路径按多卡并行跑通，再逐步接入 T5、CLIP、VAE 和完整输入输出流程。

## 背景与目标

DreamZero 当前参考实现主要由以下组件组成：

| 组件 | 作用 | 第一版策略 |
| --- | --- | --- |
| T5 text encoder | 将文本 prompt 编成 text embedding | 放在 `phyai.models.dreamzero` 下原生实现，后期可做 rank0/broadcast 优化 |
| CLIP image encoder | 生成图像条件特征 | 参考 Pi0.5 的 vision encoder 组织方式，但实现放在 DreamZero 目录 |
| Wan VAE encoder | 将 observation image/video 编成 latent | 参考 Cosmos3 Wan VAE，但不能跨 model import |
| KV update | 用 clean image/history latent 预填历史 KV cache | 必须走同一套 TP 多卡 DiT |
| DiT forward | denoise loop 中预测 video/action flow | 第一优先级，必须 TP=4 切分权重 |
| Scheduler | 控制 timestep、CFG、UniPC step、KV update 与 denoise loop | 参考 Cosmos3 policy scheduler 的职责划分 |

目标部署环境是 8 卡、单卡约 48GB 显存级别的 GPU。DreamZero-DROID 的 DiT 是 14B 级别，完整单卡推理无法作为主线 baseline。因此第一版可运行路径从一开始就按多卡设计：

```text
world_size = 8
tp_size = 4
cfg_size = 2
```

这里的 `tp_size=4` 才是权重切分；不要把它叫成 `DP=4`。传统 data parallel 会在每张卡加载完整模型，不能解决 DiT 权重显存爆炸问题。

## 总体架构

DreamZero 代码放在独立 model 目录，遵循 PhyAI model implementation skill 的三层结构：

```text
phyai/src/phyai/models/dreamzero/
  __init__.py
  configuration_dreamzero.py
  modeling_dreamzero.py
  model_runner_dreamzero.py
  scheduler_wn_dreamzero_policy.py
  main_dreamzero_policy_wn.py
  sampler_flow_unipc.py
  text_encoder_dreamzero.py
  image_encoder_dreamzero.py
  vae_wan.py
```

职责边界如下：

| 文件 | 职责 |
| --- | --- |
| `configuration_dreamzero.py` | 读取 checkpoint/config，定义 DiT、action、encoder、scheduler 配置 |
| `modeling_dreamzero.py` | 无状态 DiT 架构，只做一次 forward，不保存 KV cache |
| `model_runner_dreamzero.py` | 管理 KV cache、cross-attn cache、condition cache 和 forward mode |
| `scheduler_wn_dreamzero_policy.py` | 编排 KV update、denoise loop、CFG 并行、UniPC step 和多卡 collective |
| `main_dreamzero_policy_wn.py` | Engine plugin 入口，加载权重并创建 runner/scheduler |
| `sampler_flow_unipc.py` | DreamZero 使用的 FlowUniPC sampler |
| `text_encoder_dreamzero.py` | DreamZero T5/Wan text encoder |
| `image_encoder_dreamzero.py` | DreamZero CLIP/XLM-R image encoder |
| `vae_wan.py` | DreamZero 自己的 Wan VAE 适配实现 |

不直接从 `phyai.models.cosmos3` 或 `phyai.models.pi05` import 代码。Pi0.5 和 Cosmos3 只能作为参考；如果后续确认有长期共享价值，再单独讨论抽到通用层。

## 多卡并行设计

8 卡划分为两个 CFG branch，每个 branch 内部用 4 卡 TP：

```text
rank 0-3: cfg_rank=0, tp_rank=0-3, cond branch
rank 4-7: cfg_rank=1, tp_rank=0-3, uncond branch
```

每个 denoise step 的流程：

1. cond branch 和 uncond branch 并行运行同一个 TP=4 DiT forward。
2. 每个 TP rank 只保存自己的权重 shard 和 local-head KV cache。
3. DiT 输出后沿 `cfg` 轴做 `P.all_gather(axis="cfg")`。
4. 每个 rank 本地计算 CFG：

```python
guided = uncond + guidance_scale * (cond - uncond)
```

5. scheduler 用 guided flow 推进 video/action latent。

KV update 也必须走同一套 `tp_size=4, cfg_size=2` 路径。它本质上也是一次完整 DiT forward，只是输入是 clean image/history latent，并且会更新 KV cache。不能把 KV update 留在单卡，也不建议尝试把它和后续 denoise step 合并删除，因为二者输入语义不同。

## DiT TP 化原则

DreamZero DiT 的第一版实现不写完整单卡 `nn.Linear` 版本再改 TP，而是直接使用 PhyAI TP Linear：

| 子模块 | TP 方式 |
| --- | --- |
| Q/K/V projection | `ColumnParallelLinear` 或 fused `QKVParallelLinear` |
| attention output projection | `RowParallelLinear` |
| MLP gate/up projection | `ColumnParallelLinear` 或 `MergedColumnParallelLinear` |
| MLP down projection | `RowParallelLinear` |
| action/state projector | 按维度和通信代价决定 column/row/replicated |
| KV cache | 按 local attention heads 保存，不 gather 成全量 heads |

Column parallel 负责切输出维度，Row parallel 负责切输入维度。算子的 collective 由 PhyAI parallel layer 负责，模型代码只需要保证每个位置选择正确的切分方式。

第一版精度使用 bf16，不依赖 TensorRT/NVFP4。已有 ONNX/TRT/NVFP4 优化主要面向支持 FP4 的硬件，不适合作为通用 Ampere GPU 的第一版主路径。

## 分阶段完成计划

## 当前实现状态

截至 2026-07-05，已完成第一版 DreamZero 配置/权重 remap、阶段 2 的 TP DiT 参数骨架、不涉及 attention/KV 的基础 forward primitive，以及 stateless attention 的最小 forward 路径。

已落地内容：

- `configuration_dreamzero.py`：支持解析 DreamZero-DROID `action_head_cfg.config` 下的 DiT、text/image encoder、VAE 和 policy 参数。
- `modeling_dreamzero.py`：新增 DreamZero DiT skeleton，attention/FFN 主干使用 `ColumnParallelLinear` 和 `RowParallelLinear` 声明 TP=4 shard 参数。
- DiT 大矩阵的 loader 已按 TP 维度切分：Q/K/V/FFN up 按输出维切，attention out/FFN down 按输入维切。
- patch embedding、modulation、action/state category-specific 参数、head 等暂时 replicated；其中 head 和 action/state 小模块第一版复制更简单。
- 新增轻量 CPU 测试，使用 tiny config + fake `tp=4` mesh 验证本地 shard shape、`hf_keys` 和 loader 切片行为。
- 已实现并测试 `DreamZeroCategorySpecificLinear`、`DreamZeroCategorySpecificMLP`、`DreamZeroActionEncoder`、`DreamZeroSinusoidalPositionalEncoding` 和 `DreamZeroMLP` 的 forward。
- 已实现并测试 `DreamZeroSelfAttention.forward` 与 `DreamZeroCrossAttention.forward` 的最小 no-cache/stateless 路径：modeling 层接受外部 KV/cache 参数并返回更新后的 cache，但不把 cache 存到模块状态中。
- `DreamZeroSelfAttention` 目前覆盖投影、Q/K RMSNorm、PhyAI `Attention` 调用、输出投影和可选 cache append。
- `DreamZeroCrossAttention` 目前覆盖 Wan I2V 风格的 image/text context split、text/image 两路非 causal attention、输出投影和可选 text cross-attn cache。

尚未完成内容：

- `DreamZeroDiT.forward` 仍未接入真实 forward。
- attention 中的 DreamZero 专用 RoPE、teacher-forcing clean/noisy split、blockwise causal mask、action/state register mask，以及 KV update 的完整语义仍未实现。
- `DreamZeroDiTBlock.forward` 仍未接入真实 block 计算。
- KV cache 的创建、更新、读取还未实现，应放在 runner 层。
- `tp_size=4,cfg_size=2` 的 8 卡 scheduler 编排还未实现。
- T5、CLIP、VAE 还未接入。

当前通过的验证：

```bash
docker exec phyai_dev_luyiwen bash -lc \
  'cd /phyai_workspace/phyai && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest phyai/tests/models/dreamzero -q'
```

结果：`19 passed`。重启 `phyai_dev_luyiwen` 后容器内 CUDA/NVML 已恢复，PyTorch 可见 8 张 A40；但真实 4 卡/8 卡 TP runtime 仍需后续用 `torchrun` 单独验证。

### 阶段 1：配置与权重映射

目标：先不跑完整模型，只确认配置和权重结构可控。

- 新增 DreamZero model 目录和配置 dataclass。
- 解析 DreamZero-DROID config 中的 DiT 参数：`dim=5120`、`num_layers=40`、`num_heads=40`、`ffn_dim=13824`、action/state/frame layout 等。
- 编写 DiT weight remap 草稿，区分 replicated、column-sharded、row-sharded 权重。
- 检查每个 sharded 维度能被 `tp_size=4` 整除。

验收标准：

- 能加载 config 并打印关键结构参数。
- weight remap key 覆盖率清晰，missing/unexpected 可解释。
- 不需要加载完整模型到单张 GPU。

### 阶段 2：TP=4 DiT skeleton

目标：用 4 卡搭出 DiT forward 骨架，验证权重切分和基础 shape。

- 直接实现 TP 版 attention、MLP、patch embedding、unpatchify、action/state embedding。
- KV cache 先按空 cache 或最小 fake cache 验证 shape。
- 使用 reference dump 的 model-ready tensor 输入，不接 T5/CLIP/VAE。

验收标准：

- `torchrun --nproc_per_node=4` 可创建 `tp_size=4,cfg_size=1` engine。
- 每张卡只加载自己的 DiT shard。
- 单步 forward 输出 video/action flow shape 正确。

### 阶段 3：TP=4 KV update

目标：把 clean image/history latent 的 prefill 路径跑通。

- 在 runner 中创建 `DreamZeroKVCache` 和 cross-attn cache。
- runner 提供两个入口：

```text
prefill_clean_context(...)
forward_denoise_branch(...)
```

- `prefill_clean_context` 调用 DiT 并更新每层 local-head KV cache。
- modeling 层只返回 updated cache，不保存任何状态。

验收标准：

- KV cache 层数、seq_len、local heads、head_dim 与配置一致。
- cache 更新后 denoise branch 能读取同一份 cache。

### 阶段 4：TP=4 denoise loop

目标：单 branch 多卡 denoise loop 跑通。

- 移植 DreamZero FlowUniPC scheduler。
- scheduler 编排 timestep、video/action noise、clean latent re-impose、action padding tail 处理。
- 暂时继续使用 reference dump 的 prompt embedding、clip feature、vae latent。

验收标准：

- `tp_size=4,cfg_size=1` 能完成固定步数 denoise loop。
- 固定 seed 下输出稳定。
- 显存峰值低于单卡完整模型路径。

### 阶段 5：TP=4 + CFG=2

目标：扩展到完整 8 卡拓扑。

- 新增 CFG branch 分派逻辑：`cfg_rank=0` 跑 cond，`cfg_rank=1` 跑 uncond。
- KV update 和 denoise forward 都走 CFG/TP mesh。
- 使用 `P.all_gather(axis="cfg")` 收集 cond/uncond 输出。
- 本地计算 guided flow 后推进 scheduler。

验收标准：

- `torchrun --nproc_per_node=8` 能跑 `tp_size=4,cfg_size=2`。
- 输出对齐单 branch 顺序 CFG 逻辑。
- 所有 rank 都执行 step，只有 rank0 保存结果。

### 阶段 6：逐步接入 encoder

目标：把 reference dump 输入替换为 PhyAI 内部组件输出。

推荐顺序：

1. T5 text encoder：直接影响 prompt embedding，优先接入。
2. CLIP image encoder：验证 `clip_feature`。
3. VAE encoder：验证 clean image/video latent。
4. VAE decoder：只在需要输出视频时接入。

encoder 第一版不强制 TP。可先复制或在每个 CFG group 的 rank0 编码后 broadcast；等 DiT 主路径稳定后再优化 encoder 显存和速度。

## 验证计划

第一版验证不能只看 shape，需要和 DreamZero 参考实现对齐：

| 验证项 | 目标 |
| --- | --- |
| config parity | 影响 shape、attention、RoPE、patch、action layout 的字段一致 |
| weight parity | remap 后每类权重切片正确，代表 tensor 数值能 spot-check |
| DiT single step | 同一 model-ready 输入下，flow/action 输出 cosine > 0.99 |
| KV update | cache shape、seq_len、local heads 与 reference 语义一致 |
| TP parity | `tp=4,cfg=1` 对齐未切分逻辑或 reference dump |
| CFG parity | `tp=4,cfg=2` 对齐顺序 cond/uncond CFG |
| end-to-end smoke | 固定 seed、固定输入，8 卡完成 action 输出 |

模型级大权重验证脚本放到 `.cache/`，不要放入主 CI。主仓库测试只保留轻量配置、shape、weight remap 或新增通用层测试。

## 文件清单

第一版建议新增：

```text
phyai/src/phyai/models/dreamzero/
  __init__.py
  configuration_dreamzero.py
  modeling_dreamzero.py
  model_runner_dreamzero.py
  scheduler_wn_dreamzero_policy.py
  main_dreamzero_policy_wn.py
  sampler_flow_unipc.py
  text_encoder_dreamzero.py
  image_encoder_dreamzero.py
  vae_wan.py

phyai-utils-tools/src/phyai_utils_tools/models/dreamzero/
  __init__.py
  processor_dreamzero.py
  steps_dreamzero.py

examples/dreamzero/
  run_dreamzero_policy_wn.py

docs/zh/models/dreamzero/
  implementation_plan.md
```

可选新增轻量测试：

```text
phyai/tests/models/dreamzero/
  test_configuration_dreamzero.py
  test_weight_remap_dreamzero.py
  test_dreamzero_shapes.py
```

GPU/大权重验证建议放在：

```text
.cache/dreamzero_validation/
  compare_ref_single_step.py
  compare_tp4.py
  smoke_tp4_cfg2.py
```

## 当前默认决策

- 第一版只做 bf16，不做 FP4/NVFP4/TensorRT。
- 主线不依赖完整单卡推理 baseline。
- DiT 从第一版开始就是 TP=4。
- CFG 并行使用 `cfg_size=2`，不是 data parallel。
- KV cache 放在 runner，不放在 modeling。
- scheduler 只通过 runner 调用 model，不直接调 `nn.Module.forward`。
- T5、CLIP、VAE 都放在 DreamZero model 目录下实现或适配，不跨 model import。
