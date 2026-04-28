# SpikeNet → btorch 迁移指南

本文档记录了将 SpikeNet（C++/MATLAB 模拟器）的 Chen & Gong 2021 双峰 WTA 模型迁移到
btorch 过程中遇到的所有关键问题、修复方法和参数注意事项，并附一个最小可运行 demo。

---

## 迁移结论

**已成功迁移。** 判断依据：

| 指标 | SpikeNet 预期 | btorch 实现结果 |
|------|--------------|----------------|
| 平均发放率 | ~5–15 Hz | 9.8 Hz（inh_scale=0.5） |
| Winner 切换 | Levy-like 多次切换 | 24 次切换 / 6 s |
| GPU 执行 | RTX 5090 支持 | ✓（spike_gpu 环境） |
| 动力学机制 | ChemSyn model-0 pre-syn 门控 | ✓（`ChemSynModel0Gate` 封装） |

迁移代码位于 `spikenet_btorch_demo/chen_gong_2021_static_double_peak.py`。

---

## Bug 一：突触过度激活（发放率 200+ Hz）

### 现象
使用 `SpikeNetCompositePSC` 默认接口（直接传入 spike 向量 `z`）时，网络在刺激开始后
数十毫秒内发放率飙升至 200–225 Hz，远超生理范围（5–30 Hz）。

### 根本原因
btorch 的 `SpikeNetExponentialPSC` 实现的是**简单指数 PSC**：

```
每个 spike → 立即累加权重全值到 psc
psc[t] = alpha * psc[t-1] + W @ z[t]
```

而 SpikeNet C++ 的 `ChemSyn model-0` 使用**突触前递质门控**：

```
# 每个神经元 i 维护释放门控变量 s[i] ∈ [0,1]
# 发放后，trans_left[i] += steps_trans（维持 Dt_trans 时长的释放窗口）
# 每步：
contribution = K_trans * (1 - s[i]) * weight    # 释放量随 s 增大而减小
s[i] += K_trans * (1 - s[i])                    # s 在释放期积累
s[i] *= exp(-dt / tau_decay)                     # 始终衰减
```

关键：高发放率下 `s[i] → 1`，`(1 - s[i]) → 0`，**自动饱和**，防止失控激发。
btorch 的简单指数 PSC 没有此饱和机制。

### 推荐修复方案（使用 `ChemSynModel0Gate`）

btorch 现已提供封装好的 `ChemSynModel0Gate`（`btorch.models.synapse`），
直接替代手动循环。`forward(z)` 返回 `release` 张量，传入各子通道即可：

```python
from btorch.models.synapse import ChemSynModel0Gate

# --- 初始化（在仿真循环外）---
gate = ChemSynModel0Gate.from_ei_populations(
    n_e=n_e, n_i=n_i,
    Dt_trans_ampa=cfg.Dt_trans_AMPA,   # ms
    Dt_trans_gaba=cfg.Dt_trans_GABA,   # ms
    tau_syn=cfg.tau_ampa,              # 使用 AMPA tau（或合适的代表值）
    dt=cfg.dt,
).to(device)
gate.init_state()

# --- 仿真循环内（每步）---
z = neuron(rec_current + ext_current)

with environ.context(dt=cfg.dt):      # context 已在外层时无需重复
    release = gate(z)                 # (1, n_total) float32，含饱和门控

ampa_psc = synapse.ampa(release)   # 传 release，不传 z
gaba_psc = synapse.gaba(release)
synapse.psc = ampa_psc + gaba_psc
```

**注意**：`synapse.ampa` 权重矩阵只对 E 神经元列有非零权重，`synapse.gaba` 只对
I 神经元列有非零权重。传入完整的 `release` 张量时，零权重列自动屏蔽另一群体，
无需额外分拆。

### 备选：手动实现（仍有效）

若不引入 `ChemSynModel0Gate`，可在循环中手动展开：

```python
fired = (z > 0).to(torch.int32)
trans_left = trans_left + fired * steps_per_neuron
active = trans_left > 0

release = active.float() * k_trans * (1.0 - s_pre)
s_pre = torch.where(active, s_pre + k_trans * (1.0 - s_pre), s_pre) * s_pre_decay
trans_left = torch.clamp(trans_left - active.to(torch.int32), min=0)

ampa_psc = synapse.ampa(release)
gaba_psc = synapse.gaba(release)
synapse.psc = ampa_psc + gaba_psc
```

两种方式数值上完全等价（见一致性测试）。

### 效果
| | 修复前 | 修复后 |
|--|--------|--------|
| 平均发放率 | 200–225 Hz | 4.5 Hz |
| 活跃神经元峰值 | 接近不应期上限 | 15–30 Hz |

---

## Bug 二：参数设置——抑制强度过高导致无 Winner 切换

### 现象
按 Chen & Gong 2021 原文参数运行，发放率正常（4.5 Hz），但 winner_switches = 0，
网络处于约 50/50 的对称静止态，始终无一方占优。

### 根本原因
Chen & Gong 2021 原始的 `g_EI = 13.5e-3`、`g_II = 25e-3` 参数设置使抑制过强，
网络锁定在初始 winner 无法逃出，或两个吸引子无法稳定维持。

### 修复方案
扫参发现 `inh_scale = 0.5`（即 `g_EI = 6.75e-3`、`g_II = 12.5e-3`）时：
- 发放率升至 ~10 Hz（更活跃的竞争态）
- 6 秒刺激窗口内出现 24 次 winner 切换
- 平均驻留时间 ~250 ms，符合 Levy-like 特征

**经验规则**：若复现 Levy-like 切换，先尝试将原文 `g_EI`/`g_II` 降至 50–60%。

---

## Bug 三：`trans_left` 计步器 E/I 不分

### 现象
将 I 神经元的 `trans_left` 也用 `steps_trans_ampa` 更新，当 `Dt_trans_AMPA ≠
Dt_trans_GABA` 时释放窗口宽度错误。

### 推荐修复（使用 `ChemSynModel0Gate.from_ei_populations`）

`from_ei_populations()` 内部自动构造每神经元的 `steps_trans` 张量，
正确区分 E/I 子群体——这正是 Bug 3 的封装所在：

```python
gate = ChemSynModel0Gate.from_ei_populations(
    n_e=n_e, n_i=n_i,
    Dt_trans_ampa=1.0,   # ms，E 神经元释放窗口
    Dt_trans_gaba=0.5,   # ms，I 神经元释放窗口（不同！）
    tau_syn=tau_ampa,
    dt=dt,
)
# gate.steps_trans[:n_e]  = round(1.0 / dt)
# gate.steps_trans[n_e:]  = round(0.5 / dt)   ← 正确区分
```

### 备选：手动修复

```python
# 错误写法
trans_left = trans_left + fired * steps_trans_ampa   # I 神经元也用 AMPA 步数

# 正确写法
steps_per_neuron = torch.full_like(trans_left, steps_trans_ampa)
steps_per_neuron[:, n_e:] = steps_trans_gaba
trans_left = trans_left + fired * steps_per_neuron
```

当前 SpikeNet 默认 `Dt_trans_AMPA = Dt_trans_GABA = 1.0 ms`，两者相等时无数值影响，
但保持代码正确性以防未来参数变化。

---

## Bug 四：连接采样规则偏离 SpikeNet 约定

### 现象
用简单距离衰减 Bernoulli 采样构建 EE 连接，导致 out-degree 方差过大，
部分神经元极度高连接，网络动力学不稳定。

### SpikeNet 约定
SpikeNet 的 lattice pipeline 使用：
1. **先固定每个 pre 神经元的 out-degree**（从目标均值采样，低 CV）
2. **再按距离权重对 post 神经元采样**（近邻概率高）

### btorch 正确实现
```python
def _sample_connections_fixed_outdegree(
    coords_pre, coords_post, p0, decay, n_target,
    degree_std=None, rng=None,
):
    """
    每个 pre 神经元：
      1. 计算到所有 post 的距离概率 p_i ∝ exp(-r²/(2*decay²))
      2. 从低 CV 分布采样 out-degree k_i
      3. 按 p_i 无放回采样 k_i 个目标
    """
    ...
```

---

## 其他注意事项

### 1. `environ.context(dt=...)` 是必须的
btorch 的所有时间常数（`tau_ref`、`tau_ampa` 等）**在 forward 内**通过
`environ.dt` 获取当前时间步长，而非构造时固定。仿真循环必须套：

```python
with torch.no_grad(), environ.context(dt=cfg.dt):
    for t in range(t_steps):
        ...
```

若忘记此 context，时间常数默认值（通常 `dt=1.0`）会使指数衰减完全错误。

### 2. `functional.init_net_state` vs `module.init_state`
btorch 提供两种初始化方式：

```python
# 方式 A：通过 functional 模块（推荐用于顶层调用）
functional.init_net_state(neuron, batch_size=1, device=device, dtype=dtype)
functional.init_net_state(synapse, batch_size=1, device=device, dtype=dtype)

# 方式 B：直接调用模块方法
neuron.init_state(batch_size=1, device=device, dtype=dtype)
synapse.init_state(batch_size=1, device=device, dtype=dtype)   # 同时初始化 ampa/gaba
```

`SpikeNetCompositePSC.init_state` 会自动向下传递给 `self.ampa` 和 `self.gaba`，
无需单独初始化子通道。

### 3. 外部输入的处理方式差异
SpikeNet 的外部输入通过 `steps_trans` 步展开累加，btorch demo 中用**瞬时累加
+ 指数衰减**近似：

```python
alpha_ext = math.exp(-dt / tau_ampa_ext)
ext_gs = alpha_ext * ext_gs + poisson_spikes * g_ext
ext_current = ext_gs * (E_ampa - v)
```

这是次要近似，不引起过度激活，但会使外部输入的时间轮廓略有不同。

### 4. `SpikeNetCompositePSC` 不接受分开的 E/I spike

`SpikeNetCompositePSC.forward(z)` 调用 `self.ampa(z) + self.gaba(z)`，
两个通道都接收同一个 `z`。权重矩阵负责选择性（AMPA 矩阵只有 E 列非零，
GABA 矩阵只有 I 列非零），不需要手动分拆。

实现 ChemSyn model-0 时**分别调用子通道**而不是顶层 `forward`：

```python
ampa_psc = synapse.ampa(release)   # 用 release 替代 z
gaba_psc = synapse.gaba(release)
synapse.psc = ampa_psc + gaba_psc  # 手动更新 .psc 属性
```

### 5. GPU 环境要求（RTX 5090）
RTX 5090 使用 CUDA Compute Capability sm_120，标准 PyTorch（≤ cu124）不支持。
必须使用包含 nightly PyTorch 的 `spike_gpu` conda 环境：

```bash
conda activate spike_gpu   # PyTorch 2.13.0.dev+cu130
```

`spike` 环境（PyTorch 2.6.0+cu124，支持至 sm_90）在 RTX 5090 上会报
`no kernel image is available for execution on the device`。

### 6. Slurm 作业内存设置
`DefMemPerCPU = 100 MB`，`--cpus-per-task=8` 时默认只分配 800 MB，
四个并行 Python+PyTorch 进程立即被 OOM Killer 杀死。必须显式指定：

```bash
#SBATCH --mem=64G
```

---

## 参数对照表（Chen & Gong 2021）

| 参数 | SpikeNet 值 | btorch demo 值 | 说明 |
|------|------------|---------------|------|
| `Dt_trans_AMPA` | 1.0 ms | 1.0 ms | 突触前释放窗口 |
| `Dt_trans_GABA` | 1.0 ms | 1.0 ms | |
| `tau_decay_AMPA` | 5.0 ms | 5.8 ms | Chen&Gong 校准值 |
| `tau_decay_GABA` | 3.0 ms | 6.5 ms | |
| `g_EI` | — | 13.5e-3（原文）→ 6.75e-3（切换最优） | |
| `g_II` | — | 25e-3（原文）→ 12.5e-3（切换最优） | |
| `g_mu`（EE 均值）| — | 4e-3 | 不宜上调（ee_scale>1.5 → 沉默或失控） |
| `N_ext` | 1000 | 1000 | 外部泊松神经元数 |
| `rate_ext_E` | 0.85 kHz | 0.85 kHz | |
| `dt` | 0.1 ms | 0.1 ms | |

---

## 最小可运行 Demo

以下 demo 构建一个 **20 神经元（16 E + 4 I）** 的小型 WTA 网络，
使用 `ChemSynModel0Gate` 封装突触前门控，展示核心迁移模式：

```python
"""
minimal_spikenet_btorch_demo.py
最小 SpikeNet → btorch 迁移示例（20 神经元，CPU，无 Slurm）

依赖：conda activate spike_gpu（或 spike，如果在 CPU 上运行）
运行：python minimal_spikenet_btorch_demo.py
"""
import math
import torch
import numpy as np
import sys
from pathlib import Path

# 将 btorch 加入路径（根据实际位置修改）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "btorch"))

from btorch.models import environ, functional
from btorch.models.neurons.spikenet import SpikeNetNeuron
from btorch.models.synapse import SpikeNetCompositePSC, ChemSynModel0Gate

# ── 超参数 ────────────────────────────────────────────────────────────────────
N_E, N_I = 16, 4
N_TOTAL   = N_E + N_I
DT        = 0.1        # ms
T_MS      = 500.0      # 模拟时长
DEVICE    = torch.device("cpu")
DTYPE     = torch.float32

# SpikeNet ChemSyn model-0 参数
TAU_AMPA   = 5.8       # ms，AMPA 衰减时间常数
TAU_GABA   = 6.5       # ms，GABA 衰减时间常数
DT_TRANS_AMPA = 1.0    # ms，E 神经元突触前释放窗口
DT_TRANS_GABA = 1.0    # ms，I 神经元突触前释放窗口
G_EE       = 4e-3      # EE 权重
G_EI       = 6.75e-3   # E→I 权重
G_IE       = 5e-3      # I→E 权重
G_II       = 12.5e-3   # II 权重
G_EXT      = 2e-3      # 外部输入权重
RATE_EXT   = 0.85      # 外部泊松输入率（kHz）
N_EXT      = 100       # 外部泊松神经元数
TAU_EXT    = 5.8       # ms，外部输入衰减时间常数
E_AMPA     = 0.0       # mV，AMPA 反转电位

# ── 构建稀疏延迟权重矩阵 ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)
DELAY = 1   # 单一延迟步（0.1 ms）

def make_weight_matrix(pre_idx, post_idx, weight, n):
    """构建 CSR 稀疏权重矩阵 [n_post, n_pre]"""
    rows = torch.tensor(post_idx, dtype=torch.long)
    cols = torch.tensor(pre_idx,  dtype=torch.long)
    vals = torch.full((len(rows),), weight, dtype=DTYPE)
    w = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n)
    ).to_sparse_csr()
    return w

# EE：E→E 全连接（演示用，实际应稀疏）
pre_ee  = np.repeat(np.arange(N_E), N_E)
post_ee = np.tile(np.arange(N_E), N_E)
mask_ee = pre_ee != post_ee          # 去除自连接
pre_ee, post_ee = pre_ee[mask_ee], post_ee[mask_ee]
w_ee = make_weight_matrix(pre_ee, post_ee, G_EE, N_TOTAL)

# I→E：I→E 连接
pre_ie  = np.repeat(np.arange(N_E, N_TOTAL), N_E)
post_ie = np.tile(np.arange(N_E), N_I)
w_ie = make_weight_matrix(pre_ie, post_ie, -G_IE, N_TOTAL)   # 负号=抑制

# E→I：E→I 连接
pre_ei  = np.repeat(np.arange(N_E), N_I)
post_ei = np.tile(np.arange(N_E, N_TOTAL), N_E)
w_ei = make_weight_matrix(pre_ei, post_ei, G_EI, N_TOTAL)

# I→I：I→I 连接
pre_ii  = np.repeat(np.arange(N_E, N_TOTAL), N_I)
post_ii = np.tile(np.arange(N_E, N_TOTAL), N_I)
mask_ii = pre_ii != post_ii
pre_ii, post_ii = pre_ii[mask_ii], post_ii[mask_ii]
w_ii = make_weight_matrix(pre_ii, post_ii, -G_II, N_TOTAL)   # 抑制

exc_w = {DELAY: w_ee + w_ei}   # 兴奋性：EE + EI（EI 目标是 I 神经元）
inh_w = {DELAY: w_ie + w_ii}   # 抑制性：IE + II

# ── 构建模型 ──────────────────────────────────────────────────────────────────
neuron = SpikeNetNeuron(
    n_neuron=N_TOTAL,
    v_threshold=torch.full((N_TOTAL,), -50.0),
    v_reset=torch.full((N_TOTAL,), -60.0),
    tau_ref=torch.full((N_TOTAL,), 2.0),
).to(DEVICE)

synapse = SpikeNetCompositePSC(
    n_neuron=N_TOTAL,
    exc_weights_by_delay=exc_w,
    inh_weights_by_delay=inh_w,
    tau_ampa=TAU_AMPA,
    tau_gaba=TAU_GABA,
    use_sparse=True,
).to(DEVICE)

# ChemSyn model-0 门控模块（封装 Bug 1 + Bug 3）
gate = ChemSynModel0Gate.from_ei_populations(
    n_e=N_E, n_i=N_I,
    Dt_trans_ampa=DT_TRANS_AMPA,
    Dt_trans_gaba=DT_TRANS_GABA,
    tau_syn=TAU_AMPA,
    dt=DT,
).to(DEVICE)

functional.init_net_state(neuron,  batch_size=1, device=DEVICE, dtype=DTYPE)
functional.init_net_state(synapse, batch_size=1, device=DEVICE, dtype=DTYPE)
gate.init_state()

alpha_ext = math.exp(-DT / TAU_EXT)
ext_gs    = torch.zeros(1, N_TOTAL, device=DEVICE)

# ── 仿真循环 ──────────────────────────────────────────────────────────────────
t_steps    = int(T_MS / DT)
lambda_ext = N_EXT * RATE_EXT * DT / 1000.0   # 每步期望泊松事件数

spike_counts = np.zeros(N_TOTAL)

with torch.no_grad(), environ.context(dt=DT):
    for t in range(t_steps):
        # 外部泊松输入
        poi = torch.poisson(torch.full((1, N_TOTAL), lambda_ext, device=DEVICE))
        ext_gs = alpha_ext * ext_gs + poi * G_EXT
        ext_current = ext_gs * (E_AMPA - neuron.v)

        # 神经元更新
        z = neuron(synapse.psc + ext_current)

        # ── ChemSyn model-0 突触前门控（核心迁移模式）──
        release = gate(z)                    # (1, N_TOTAL) release 张量

        # 将 release 分别传入各子通道（不调用 synapse(z)！）
        ampa_psc = synapse.ampa(release)
        gaba_psc = synapse.gaba(release)
        synapse.psc = ampa_psc + gaba_psc

        spike_counts += z[0].cpu().numpy()

# ── 输出结果 ──────────────────────────────────────────────────────────────────
mean_rate = spike_counts.sum() / (N_TOTAL * T_MS * 1e-3)
e_rate    = spike_counts[:N_E].mean() / (T_MS * 1e-3)
i_rate    = spike_counts[N_E:].mean() / (T_MS * 1e-3)

print(f"Total spikes : {int(spike_counts.sum())}")
print(f"Mean rate    : {mean_rate:.2f} Hz")
print(f"E mean rate  : {e_rate:.2f} Hz")
print(f"I mean rate  : {i_rate:.2f} Hz")
print("Expected: physiological range 5–30 Hz (not 200+ Hz)")
```

### 运行验证

```bash
conda activate spike_gpu
python minimal_spikenet_btorch_demo.py
# Expected output:
# Total spikes : ~300–600
# Mean rate    : 5–20 Hz
# E mean rate  : 5–15 Hz
# I mean rate  : 10–30 Hz
```

若去掉 `gate(z)` 改为直接 `synapse(z)`，输出会变成 200+ Hz，验证门控的必要性。

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `btorch/btorch/models/synapse.py` | `ChemSynModel0Gate` 类（Bug 1 + Bug 3 封装） |
| `btorch/btorch/models/test_chemsyn_model0_gate.py` | 一致性测试（7 项，含数值等价验证） |
| `spikenet_btorch_demo/chen_gong_2021_static_double_peak.py` | 完整迁移实现，含 ChemSyn model-0、CLI 扫参接口 |
| `spikenet_btorch_demo/scan_chen2021_dynamics.py` | 批量扫参运行器（ext/inh/ee/noise/tau 五维）|
| `spikenet_btorch_demo/slurm_model0_sweep.sh` | Slurm 提交脚本（含内存修复）|
| `spikenet_btorch_demo/make_winner_video.py` | winner 切换可视化视频生成器 |
| `spikenet_btorch_demo/static_gaussian/scan_model0_sweep2/` | 17 组扫参结果（含 inh_0p50 最优解）|
| `spikenet_btorch_demo/static_gaussian/scan_model0_sweep2/inh_0p50/winner_video.mp4` | winner 切换演示视频（21.7 s，3× 慢放）|
