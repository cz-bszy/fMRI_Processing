# 时间序列验证与 volume-vs-surface 比较 (stage 10)

本文说明 `stages/10_validate.sh` 计算的每一个指标: 定义、在 TR = 2–3 s 的 legacy 数据
(ABIDE / ADNI, 3–4 mm 体素, 无 fieldmap, 5–10 min 扫描) 上"好 / 坏"大致是什么样子、
如何用它们在 volume 与 surface 两条 stream 之间以及在各去噪策略 (denoising strategy)
之间做选择, 最后列出尚未自动化的进一步验证思路。接口 (文件名、列名) 以
`docs/DESIGN.md` 第 12 节为准; 本文只解释含义和用法。

所有"经验范围"都是经验值: 指标的绝对水平随 TR、扫描长度、体素大小、atlas 粒度和
站点而变, 请以**同一数据集内的相对比较**为准, 不要跨数据集套用阈值。

## 1. 概览

### 1.1 输入 (只读 stage 07 的 ROI 表, 从不读 4D 数据)

| 文件 | 用途 |
| --- | --- |
| `<RUN>_space-<TPL>_atlas-<A>_desc-<S>_timeseries.tsv` | volume stream, 去噪后 ROI 均值序列 (header = ROI 名, n/a = 无效 ROI) |
| `<RUN>_space-<TPL>_atlas-<A>_desc-preproc_timeseries.tsv` | 去噪前、已全局缩放 (SCALE_TARGET) 的 ROI 均值序列 → tSNR 分子、variance_removed、lowfreq_power_fraction |
| `<RUN>_space-<TPL>_atlas-<A>_desc-<S>_coverage.tsv` | `roi name network atlas_voxels valid_voxels coverage_fraction included`: 列 → atlas label 的映射 |
| `<RUN>_space-fsLR_atlas-<A>_desc-{<S>,preproc}_timeseries.tsv` + `_coverage.tsv` | surface stream; 列 = dlabel parcel, 按 label key 升序 (Schaefer 与 volumetric 顺序一致, 名字可能略有差异); coverage 多一列 `data_voxels` |
| `<RUN>_desc-censor.1D` | 1 = keep, 0 = censored |
| `<RUN>_desc-confounds_timeseries.tsv` | `framewise_displacement` 列 (Power FD) |
| `<RUN>_desc-<S>_denoise.json` | `polort`, `dof_remaining` |
| `$RESOURCE_DIR/atlases/<A>/labels.tsv` | `index name network`: 网络标签 |
| `$RESOURCE_DIR/atlases/<A>/<A>_space-MNI152NLin6Asym_res-02_dseg.nii.gz` | parcel 质心 (world mm, 经 affine), 缓存到 `centroids.tsv`; 资源目录只读时只在内存中计算 |

### 1.2 输出

Per run (`derivatives/sub-X/func/`):

* `<RUN>_desc-validation.tsv` — long 格式 `stream strategy atlas metric value`, 每个 stream × strategy × atlas 13 行。
* `<RUN>_desc-validation.json` — 同样的数字 (`streams.<stream>.<strategy>.<atlas>.<metric>`), 外加 `n_volumes`, `n_censored`,
  `lowfreq_white_noise_baseline`, `stream_comparison`, `notes` (所有"为什么某个值是 n/a"的原因)。
* `<RUN>_desc-streamcompare.tsv` — 只在两条 stream 都存在时: `strategy atlas scope roi metric volume surface value`;
  `scope=run` 行是在 common ROIs 上重算的每个指标 (`value` = surface − volume; `fc_similarity` 与 `n_roi_common` 本身),
  `scope=roi` 行是每个 ROI 的 `roi_tsnr` (两条 stream)。只剩一条 stream 时旧文件会被删除。

Group (`derivatives/group/`, `10_validate.sh --group`):

* `validation_long.tsv` — 所有 run 的 long 表, 前置 `subject run_label group` (site 来自 manifest), 并追加 `fc_typicality` 行。
* `stream_comparison.tsv` — 每个 strategy × atlas × metric: `direction n median_volume median_surface median_diff
  n_surface_higher n_volume_higher wilcoxon_p significant better median_value`。
* `strategy_comparison.tsv` — 每个 stream × atlas × metric × strategy: `n median q25 q75 best`。
* `fc_typicality.tsv` — 每个 run × stream × strategy × atlas 的 leave-one-out typicality。
* `stream_comparison.png` — 配对点线图 (每条线一个 run, 颜色 = site, 菱形 = 中位数)。
* `validation_report.html` — 自包含报告 (中文说明 + 上述表和图)。

### 1.3 运行

```bash
bash stages/10_validate.sh -c config/datasets/x.conf sub-0001   # 每个被试, 在 07 之后
bash stages/10_validate.sh -c config/datasets/x.conf --group    # 所有被试之后一次
```

`run_pipeline.sh` 的 `validate` 阶段自动做这两步 (`validate` 在 `qc` 之前, 所以 stage 08 的被试报告会嵌入
per-run 的 validation / streamcompare 表)。Per-run 步骤按 stage hash 跳过 (依赖 `07_timeseries__<RUN>` 的标记和
`DENOISE_STRATEGIES ATLASES CUSTOM_ATLASES CENSOR_MODE`); group 步骤总是重算 (只需几秒)。

### 1.4 保留帧 (retained frames) 与无效 ROI 的规则

* **保留帧** = censor 向量为 1 的帧。`CENSOR_MODE=NTRP` / `ZERO`: 序列长度 = censor 向量长度, 直接用 censor 向量取子集。
  `CENSOR_MODE=KILL`: 3dTproject 已删除被 censor 的帧, 序列长度 = 保留帧数, 此时所有行都保留, 而逐帧协变量 (FD)
  用 censor 向量取子集后再与之对齐。pre-denoise 表从不被 3dTproject 缩短, 总是用 censor 向量取子集。
  长度既不等于 censor 长度也不等于保留帧数 → 该表跳过, 原因写进 `notes`。FD 的长度必须恰好等于 censor 向量长度
  (无 censor 时等于序列长度), 否则 `fd_fc_coupling` = n/a 并记录 note, 绝不静默截断。
* **无效 ROI**: stage 07 已把 coverage < `MIN_ROI_COVERAGE` 的 ROI 写成 n/a; 在保留帧上为常数的列同样无效。
  无效 ROI 在计算 FC 前被丢弃 (从不插补), 数量记为 `n_roi_nan`。
* **Stream 比较** 只在两条 stream 都有效的 ROI (common ROIs) 上重算所有指标, 按唯一 ROI 名称显式对齐 (身份不匹配 → 不比较并说明);
  `n_roi_nan / n_retained / dof_remaining` 仍描述各自的 stream。

## 2. 指标定义与解读

### 2.1 `roi_tsnr_median` / `roi_tsnr_p10`

* **定义**: 每个 ROI, tSNR = mean_t(pre-denoise scaled ROI series) / SD_t(denoised ROI series), 两者都只在保留帧上算。
  `median` = 全部有效 ROI 的中位数, `p10` = 第 10 百分位 (最差 10% ROI 的水平)。
* **为什么这样定义**: 去噪后的序列是零均值 (polort + bandpass), 自身的 mean/SD 没有意义; 信号水平来自 stage 03
  全局缩放 (`SCALE_TARGET=10000`) 后的 pre-denoise ROI 均值。
* **方向**: 越高越好 (但看陷阱)。
* **经验范围**: ROI 均值对几十到几百个体素做了空间平均, 因此远高于体素级 tSNR (3 mm / TR 2 s legacy 数据的体素级
  GM tSNR 常见 30–80)。Schaefer 100/200 在 bandpass + 24P/WM/CSF 后 `roi_tsnr_median` 常见 100–400。
  `p10` 明显低于 median 的一半提示信号缺失区 (眶额、颞极、下颞叶的 susceptibility dropout); `p10` < 30–50 的 run
  应检查 coverage 表和 boldref 的 QC 图。
* **陷阱**: (1) 单调随去噪强度上升: 回归量越多、low-pass 越窄, 残差 SD 越小, tSNR 越高, 但神经信号也被一起移除。
  (2) surface stream 的 ribbon-constrained mapping + 32k 重采样相当于额外的空间平均, 会抬高 tSNR。
  (3) surface stream 没有自己的 preproc 表时 (旧版 stage 07) 会借用 volume 的 pre-denoise 均值 (同一个全局缩放因子);
  此时 `variance_removed` / `lowfreq_power_fraction` 为 n/a, `notes` 有记录。
  **必须与 `split_half_r`、`network_contrast`、`dof_remaining` 一起读。**

### 2.2 `variance_removed_median`

* **定义**: 每个 ROI, 1 − var(denoised, retained) / var(pre-denoise 经 polort 阶 Legendre 去趋势, retained); ROI 中位数。
  polort 取自 `_denoise.json` (默认 2)。
* **方向**: 描述性 (descriptive)。
* **经验范围**: bandpass 0.01–0.1 Hz 在 TR 2 s 下就去掉了约 60% 的白噪声功率 (通带只占 0.09 / 0.24 的频带), 加上
  运动 / WM / CSF 回归, 0.6–0.9 常见。> 0.95: 该 run 几乎全是伪影, 或 DOF 极低 (过拟合)。< 0.3: 去噪几乎没起作用
  (检查回归量是否为常数 / 全零、滤波是否生效)。
* **陷阱**: 不是"越大越好"; GSR 策略必然更大。跨策略比较时它只回答"去掉了多少", 去掉的是不是噪声要看
  `network_contrast` / `fd_fc_coupling` 是否随之改善。

### 2.3 `split_half_r`

* **定义**: 保留帧按时间顺序分成前后两半 (contiguous halves), 各自算 Pearson FC → Fisher z → 上三角, 两个向量的
  Pearson r (`fmriproc.utils.split_half_reliability`)。每半少于 20 帧 → n/a。
* **方向**: 越高越好 (FC 的 run 内 split-half consistency)。
* **经验范围**: 强烈依赖长度。150 帧 (5 min, TR 2 s) 的 run 每半只有 75 帧, Schaefer 100 常见 0.3–0.6;
  300 帧的 run 常见 0.5–0.8。`n_retained` 充足却 < 0.2 → 该 run 的 FC 基本不可重复 (运动、扫描器漂移、
  觉醒状态变化)。
* **陷阱**: 两半共有的伪影 (残余全局信号、持续的头动模式、心跳 / 呼吸的低频混叠) 同样会提高 r, 所以
  reliability 高 ≠ validity 高, 必须对照 `fd_fc_coupling` 与 QC-FC。只在 `n_retained` 量级相同的 run 之间比较;
  跨站点先看 `n_retained` 的分布 (ABIDE 各站点 120–300 帧不等)。

### 2.4 `network_contrast`

* **定义**: (网络内边的平均 z − 网络间边的平均 z) / SD(网络间边的 z)。网络标签来自 `labels.tsv` 的 `network` 列,
  缺失时从 Schaefer 名字解析 (`7Networks_LH_Vis_1` → Vis, `17Networks_RH_DefaultA_PFCd_2` → DefaultA);
  没有网络标签的自定义 atlas → n/a。
* **方向**: 越高越好 (网络结构越清晰)。
* **经验范围**: Schaefer 7 网络、non-GSR 策略下 1.0–2.5; GSR 后 1.5–4 (GSR 引入负相关, 压低 between-network 均值)。
  < 0.5 → 检查 atlas 是否对齐 (T1→MNI、EPI→T1 的 QC 图), 或该 run 被运动主导。
* **陷阱**: 与 GSR 强相关, 只在同类策略 (GSR vs GSR, non-GSR vs non-GSR) 之间比较。17 网络的值通常低于 7 网络
  (网络更小, between 对更多), 不同 atlas 之间不可比。

### 2.5 `homotopic_contrast`

* **定义**: 同伦配对 (homotopic pairs) 的平均 z − 其余左右半球间配对的平均 z。配对: 对每个 LH parcel, 在同一 network
  的 RH parcel 中取质心离 LH 质心镜像 (x → −x, MNI world mm, 由 dseg 经 affine 算出) 最近者, 距离 ≤ 20 mm 才接受。
  半球来自名字中的 `_LH_` / `_RH_`。少于 3 对 (或没有半球 / 质心信息) → n/a。
* **方向**: 越高越好。
* **经验范围**: Fisher z 差 0.3–0.8 常见; 接近 0 → 严重问题 (左右错配、极差的配准、atlas 与数据模板不同)。
* **陷阱**: (1) volume stream 中线附近的 parcel (medial visual、precuneus、medial SFG) 左右体素相邻, 插值 / 部分容积
  会造成"假"的同伦相关 → volume 略高不一定是优势; surface stream 没有这种跨半球混叠。(2) 配准误差把左右 parcel 混在
  一起时也会虚高。看差值是否集中在 medial parcel (可用 streamcompare 的 per-ROI 表和 per-ROI FC 自行核对)。

### 2.6 `dmn_contrast`

* **定义**: Default network 中名字含 `PCC` / `pCunPCC` 的 parcel 与含 `PFC` 的 parcel 之间的平均 z, 减去这些 parcel 与
  SomMot parcel 的平均 z。依赖 Schaefer 命名 (17 网络的 DefaultA/B/C、SomMotA/B 按前缀匹配); 缺任一组 → n/a。
* **方向**: 越高越好 (经典 DMN 前后节点耦合相对感觉运动网络的分离度)。
* **经验范围**: non-GSR 下 0.2–0.5; GSR 后 0.4–0.9。≤ 0 → DMN 前后节点没有耦合, 常见于高运动 run 或 PFC dropout。
* **陷阱**: GSR 抬高它 (负相关); 只能在同类策略内比较。parcel 数少 (Schaefer 100 的 PCC parcel 只有 2–4 个) 时受单个
  parcel 覆盖率影响大。

### 2.7 `lowfreq_power_fraction`

* **定义**: 对 pre-denoise、polort 去趋势后的 ROI 序列, 在保留帧上用最小二乘谱 (Lomb–Scargle, 不插值, 不会像插值那样
  注入低频功率) 计算 Fourier 网格频率上的功率, 取 0.01–0.1 Hz 功率 / 0.01 Hz–Nyquist 功率, 再取 ROI 中位数。
* **方向**: 描述性。
* **经验范围**: 白噪声基线 = 0.09 / (1/(2·TR) − 0.01): TR 2 s → 0.375, 2.5 s → 0.474, 3 s → 0.563
  (写在 JSON 的 `lowfreq_white_noise_baseline`)。真实 BOLD 高于基线 (TR 2 s 常见 0.5–0.75)。接近或低于基线 →
  该 run 以高频噪声为主 (扫描器尖峰、未被 despike 的伪影) 或 TR / STC 参数错误。远高于 (> 0.9) → 低频漂移 / 运动主导
  (polort 未去掉的慢漂移、呼吸混叠)。
* **陷阱**: 与 TR 强相关, 跨 TR 不能比。它描述的是 pre-denoise 数据, 所以同一 run 各策略行的值相同, 两条 stream 各自计算。

### 2.8 `fd_fc_coupling`

* **定义**: 保留帧上 FD (Power) 与逐帧共波动幅度 (co-fluctuation amplitude) 的 |Spearman ρ|。共波动幅度 = z-scored
  ROI 序列两两乘积 (edge time series) 的 RSS, 用恒等式 Σ_{i<j}(z_i z_j)² = ((Σ z_i²)² − Σ z_i⁴) / 2 计算, 不构造 T×R×R
  数组。KILL 模式下 FD 先按 censor 向量取子集。FD 为常数、少于 10 个有效帧、少于 2 个有效 ROI → n/a。
* **方向**: 越低越好 (最终序列中残余的运动耦合)。
* **经验范围**: 零假设下 E|ρ| ≈ 0.8 / √n (n = 150 → 0.065, n = 300 → 0.046)。< 0.1 视为没有残余耦合; 0.1–0.2 轻度;
  > 0.2 说明运动仍在驱动瞬时 FC (需要更严格的 censoring、GSR 或 aCompCor)。
* **陷阱**: (1) censoring 截断了 FD 的取值范围, 阈值越严 ρ 越低, 一部分是统计效应。(2) 帧数不同的 run 零假设水平不同。
  (3) 它是"帧级"耦合, 不等价于 QC-FC (跨被试的 FC–mean FD 相关, stage 09), 两者要一起看。

### 2.9 `gs_residual_sd`

* **定义**: z-scored 有效 ROI 的跨 ROI 平均序列在保留帧上的 SD。ROI 相互独立 → 1/√R (R = 100 → 0.10); 完全同步 → 1。
  它是平均 FC 的单调函数: gs_residual_sd² ≈ (1 + (R − 1) · mean r) / R。
* **方向**: 描述性。
* **经验范围**: GSR 策略 0.10–0.2 (接近 1/√R); non-GSR 常见 0.3–0.6。non-GSR 下 > 0.7 → 强残余全局信号
  (呼吸、运动、vigilance 下降), 但真实的全局神经信号也在其中。
* **陷阱**: 不是质量指标; 用于解释策略之间 `network_contrast` / `dmn_contrast` 的差异, 以及识别全局信号异常大的 run。

### 2.10 `n_roi_nan`, `n_retained`, `dof_remaining`

* `n_roi_nan`: 无效 (n/a 或常数) 的 ROI 数, 越低越好。surface stream 的 n_roi_nan 多来自 FOV 之外 / 无 goodvoxel 的
  parcel (下颞、眶额), volume 的来自 coverage < `MIN_ROI_COVERAGE`。
* `n_retained`: 保留帧数 (与 stream 无关)。
* `dof_remaining`: 来自 `_denoise.json` (fit_rows − design_rank 的代数维度，不是有效独立样本量), 与 stream 无关。
  **任何比较都要在可接受的 DOF (≥ `MIN_DOF`) 下进行**: DOF 很低时 tSNR 很高但 FC 估计本身不稳定。

### 2.11 Stream 比较专用: `fc_similarity`, `n_roi_common`

* `fc_similarity`: 同一 run、同一策略、同一 atlas 下 volume 与 surface 的 Fisher-z FC 上三角的 Pearson r (common ROIs)。
  > 0.8: 两条 stream 给出同一个 connectome, 选哪条影响不大; 0.6–0.8: 有系统差异 (常来自 medial / ventral parcel);
  < 0.6: 先检查该被试的 bbregister (`bbr_cost`, `bbr_vs_init_mm`)、surface QC (`pct_badvertices`) 和两条 stream 的 coverage。
* `n_roi_common`: 两条 stream 都有效的 ROI 数。

### 2.12 Group 专用: `fc_typicality`

* **定义**: 该 run 的 Fisher-z FC 向量与同 stream × strategy × atlas 下其余 run 的平均 FC (leave-one-out) 的 Pearson r;
  一条边只在至少 3 个其他 run 上有效时参与; 每个组合至少 4 个 run, ROI 数不同的 run 被忽略。来自 stage 07 的
  `_connectivity.tsv` (pre-denoise 的不算)。
* **方向**: 越高越好 (个体数据质量的代理)。
* **经验范围**: 0.5–0.8; 最低的几个 run 优先人工检查 (报告列出最低 10 个)。
* **陷阱**: 组平均混合了患者 / 对照 / 多站点; 全组共有的伪影也会提高 typicality; 一条 stream 的 typicality 更高也可能
  只是因为它更平滑。

### 2.13 方向表 (metric-direction table, `compare_streams.DIRECTION`)

| 方向 | 指标 |
| --- | --- |
| higher is better | `roi_tsnr_median`, `roi_tsnr_p10`, `split_half_r`, `network_contrast`, `homotopic_contrast`, `dmn_contrast`, `fc_typicality` |
| lower is better | `fd_fc_coupling`, `n_roi_nan` |
| descriptive (不参与 "better" 判断) | `variance_removed_median`, `lowfreq_power_fraction`, `gs_residual_sd`, `fc_similarity`, `n_roi_common`, `n_retained`, `dof_remaining` |

## 3. 用这些指标选 volume 还是 surface

### 3.1 看哪张表

* per run: `_desc-streamcompare.tsv` 的 `scope=run` 行 (common ROIs 上重算, 成对可比)。
* group: `stream_comparison.tsv`, 每个 strategy × atlas × metric 一行: `n` (有配对值的 run 数), `median_volume`,
  `median_surface`, `median_diff` (= surface − volume), `n_surface_higher` / `n_volume_higher`, `wilcoxon_p`
  (signed-rank, n ≥ 6, 否则 n/a; 差值全 0 → n/a), `significant` (strategy × atlas 指标 family 内 BH q < 0.05), `better` (按方向表; descriptive
  指标为 `descriptive`)。报告顶部的小结表按 strategy × atlas 列出显著支持 surface / volume 的指标和未定的指标。

### 3.2 推荐的判读顺序

1. **排除项**: `dof_remaining` 与 `n_retained` 与 stream 无关。`n_roi_nan` 哪条 stream 少? surface 常在腹侧 (FOV 边缘、
   goodvoxel 之外) 丢更多 parcel; 如果研究关心这些区域, 这一条就能决定。
2. **有效性 (validity) 优先于 tSNR**: `network_contrast`、`homotopic_contrast`、`dmn_contrast`、`fc_typicality` 是否一致
   偏向一边; 再看 `split_half_r`。注意 `homotopic_contrast` 对 volume 有中线混叠的偏置 (2.5)。
3. **伪影**: `fd_fc_coupling` 更低的 stream 更好; 同时看 stage 09 的 QC-FC 表 (`derivatives/group/qcfc_*.tsv`: 显著边比例、
   median |r|、距离依赖)。
4. **tSNR 最后看**: 只有前三条都不反对时, 更高的 `roi_tsnr` 才算加分; 它天然偏向更平滑的 surface stream。
5. `fc_similarity` > 0.8 且各指标差异不显著 → 两条 stream 等价, 按下游分析的需要选 (需要 surface-based 统计 /
   与 HCP 类数据对齐 → surface; 需要皮层下、更少的依赖 (无 recon-all)、更少的 n/a → volume)。

### 3.3 常见结果模式

* surface 的 tSNR、`split_half_r` 更高, `network_contrast` 稍高, `homotopic_contrast` 略低 (没有中线混叠),
  `n_roi_nan` 略高 → 典型的"surface 更干净但覆盖略差"; 若 `dmn_contrast` / `fc_typicality` 也更高, 选 surface。
* volume 几乎所有指标更好 → 多半是 surface 分支本身有问题 (recon-all 质量差: Euler holes 高; `pct_badvertices` 高;
  FOV 短使 ribbon 采样失败) — 先修 surface QC 再比较, 不要据此下结论。
* 站点间结论不一致 (图中颜色 = site) → 分站点报告, 或两条 stream 并行报告。
* `fc_similarity` 个别 run 很低 (< 0.5) 而群体中位数正常 → 个体配准问题, 看被试报告。

### 3.4 统计注意

Wilcoxon 需要 n ≥ 6 名独立被试，同人配对 runs 先取均值；保留原始 p，显著标记使用 BH q。看效应量 (`median_diff` 相对该指标的 IQR、
`n_surface_higher` 对 `n_volume_higher` 的比例) 而不只看 p。run 数少 (< 10) 时把结论写成"倾向", 不要写成"显著更好"。

## 4. 用这些指标选去噪策略

* 表: `strategy_comparison.tsv` 给出每个 stream × atlas × metric 下各策略的被试均值分布 (median / q25 / q75)。
  `best` 字段兼容保留但为空；可用样本不匹配时不自动评选。报告第 3 节以 metric × strategy 的宽表展示。
* **GSR 与 non-GSR 分开比**: `network_contrast`、`dmn_contrast`、`gs_residual_sd` 在 GSR 下系统性变化 (负相关被引入),
  两类之间的差异反映的是 GSR 的数学效应而不是质量。legacy 数据上最稳妥的做法是各选一个并行报告
  (对应 v1 的 `Retain_GRS` / `NoGRS`)。
* 在同类策略中优先: `fd_fc_coupling` 与 QC-FC 更低、`split_half_r` / `network_contrast` / `fc_typicality` 更高、
  `dof_remaining` ≥ `MIN_DOF`、`variance_removed_median` 不极端 (< 0.95)。
* TR 2–3 s、150–300 帧的现实: bandpass 0.01–0.1 Hz 的 DOF 代价约为全长的 60%; `36p` (32 个回归量) 或
  `acompcor` + 24P 在 150 帧的 run 上会把 DOF 压到 < 15 → 这些 run 在比较前应剔除或标注 (`low_dof`); 短 run 多的站点
  倾向 `wmcsf24` / `legacy8` 一类回归量少的策略。
* 典型预期 (同一数据集内): `wmcsf24` 基线; `wmcsf24gsr` → `network_contrast` / `dmn_contrast` 升、`fd_fc_coupling` 降、
  `gs_residual_sd` → 1/√R; `acompcor` → 与 `wmcsf24` 相近, `fd_fc_coupling` 略低; `legacy8` / `legacy9gsr` → 与 v1 可比,
  指标略差 (只有 6 个运动参数)。

## 5. 尚未自动化的进一步验证 (建议手工做)

### 5.1 视觉检查: seed-based DMN 图

用去噪后的平滑 4D 数据 (`<RUN>_space-<TPL>_res-<R>_desc-<S>sm<F>_bold.nii.gz`) 和 PCC 种子
(MNI 0, −52, 26, r = 6 mm), 只用保留帧, 算种子相关并做 Fisher z:

```python
import numpy as np, nibabel as nib
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
run = "derivatives/sub-0001/func/sub-0001_task-rest"
bold = f"{run}_space-MNI152NLin6Asym_res-2_desc-wmcsf24sm6_bold.nii.gz"
mask = f"{run}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
keep = np.loadtxt(f"{run}_desc-censor.1D") > 0.5           # NTRP/ZERO: 序列全长; KILL: 已缩短, 省略这一步
seed = NiftiSpheresMasker([(0, -52, 26)], radius=6).fit_transform(bold)[keep, 0]
brain = NiftiMasker(mask_img=mask).fit()
data = brain.transform(bold)[keep]                          # (T, V) float32
s = (seed - seed.mean()) / seed.std()
d = (data - data.mean(axis=0)) / np.where(data.std(axis=0) > 0, data.std(axis=0), np.nan)
r = s @ d / len(s)                                          # 逐体素 Pearson r, 不构造 V x V 矩阵
z = np.arctanh(np.nan_to_num(r).clip(-0.999, 0.999))
brain.inverse_transform(z).to_filename("sub-0001_pcc_seed_z.nii.gz")
```

期望: 内侧前额叶 (约 0, 52, −6)、双侧角回、外侧颞叶为正; GSR 策略下 SomMot / 岛叶为负。组平均 z 图
(对所有 run 平均) 用 fsleyes 叠在模板上看。没有清晰 DMN 的个体 run 应与 `dmn_contrast` 最低的 run 对应。
surface stream 可用 `wb_command -cifti-correlation` 对 `_desc-<S>_bold.dtseries.nii` 做同样的事
(先用 `-cifti-restrict-dense-map` 或手工删掉 censored 列)。

### 5.2 与已发表的组平均 FC 比较

把所有 run 的 `_connectivity.tsv` 做 Fisher z 后平均, 得到本数据集的组平均 FC, 与公开的 Schaefer 组平均矩阵
(如 HCP 的 Schaefer 100/200 组平均, 或 Yeo 7 网络的块结构) 做上三角相关。经验上跨扫描器 / 跨人群的 r 在 0.5–0.7;
明显更低说明系统性问题 (atlas 未对齐、TR 错误、大量被试 FOV 缺失)。两条 stream 各算一次, 也可作为选择依据。
若没有可用的公开矩阵, 至少检查网络块结构: 同网络块内 z 为正且高于块外 (即 `network_contrast` 的图形化版本),
可用 `brain-figures` 一类工具画 region × region 矩阵。

### 5.3 Test-retest (有多个 session 时)

manifest 中同一 `subject` 有多个 `session` 时: (a) 边级 ICC(2,1) (session 为重复测量), 报告 ICC 中位数和 > 0.4 的边比例;
(b) fingerprinting: 用 session 1 的 FC 在 session 2 中找最相似的被试, 识别率 (Finn 2015)。两者按 stream × strategy
分别做, 与 `split_half_r` 的结论对照: run 内 reliability 高但 session 间 ICC 低 → 状态性 (state) 伪影主导。
ABIDE II 部分站点、ADNI 有纵向数据可以做; 注意纵向间隔和年龄效应。

### 5.4 站点与扫描时长的影响

`validation_long.tsv` 有 `group` (site) 列: 每个指标按站点画分布 (或 Kruskal–Wallis), 看差异是否能由 `n_retained`、
TR、体素大小解释。`split_half_r`、`fd_fc_coupling`、`fc_typicality` 都随长度变化, 比较站点前应把所有 run 截到相同的保留帧数
(取最短站点的长度, 在 ROI 表上截断后重算 `split_half_reliability` 和 FC), 或在混合模型中把 site 作为随机效应。
站点效应大时, volume-vs-surface 和策略的结论应分站点复核 (图中颜色即 site)。

### 5.5 Censoring 阈值敏感性

`CENSOR_FD` 进入 stage 04 的 hash, 下游 05/07/10 通过 `--dep` 链自动重算, 但输出会覆盖; 每次跑完先备份 group 表:

```bash
cp derivatives/group/validation_long.tsv derivatives/group/validation_long_fd0.5.tsv
CENSOR_FD=0.3 ./run_pipeline.sh -c x.conf --stages "confounds denoise timeseries validate"
cp derivatives/group/validation_long.tsv derivatives/group/validation_long_fd0.3.tsv
CENSOR_FD=0.2 ./run_pipeline.sh -c x.conf --stages "confounds denoise timeseries validate"
```

比较 `fd_fc_coupling`、QC-FC、`n_retained`、`dof_remaining`、`network_contrast`: 合理的阈值是 QC-FC 和 `fd_fc_coupling`
趋于平坦、而保留帧损失 < 20–30%、`dof_remaining` ≥ `MIN_DOF` 的最松阈值。legacy 数据 (TR 2–3 s, 儿童 / 老年人)
常落在 0.3–0.5 mm。也比较 `CENSOR_MODE` NTRP 与 KILL: 指标应基本一致 (stage 10 对两种长度处理一致),
差异大说明插值把伪影带进了通带。

### 5.6 IA vs IA2 slice timing 敏感性 (未解决的站点)

Siemens interleaved: 层数为奇数时从 1 开始 (`IA`), 偶数时从 2 开始 (`IA2`); 没有 DICOM 证据的站点只能二选一。
做法: 复制 acquisition 表, 把该站点的 `slice_order` 改成另一个变体, 用另一个 `OUT_DIR` (或把 stage 03 的输出目录
先备份) 只跑该站点:

```bash
OUT_DIR=/data/out_ia2 ACQ_TABLE=config/datasets/abide_acquisition_IA2.tsv \
  ./run_pipeline.sh -c x.conf -s "sub-A sub-B" --stages "ingest anat_recon anat_prep func_prep confounds denoise timeseries validate"
```

(recon-all 可通过 `FS_DIR` 指向已有目录复用。) 然后对同一 run 比较两个变体的 `_connectivity.tsv` (Fisher z 上三角的 r)、
`roi_tsnr`、`lowfreq_power_fraction` 和 `split_half_r`。TR 2–3 s 下相邻层之间的最大时间误差是 TR/2 ≈ 1–1.5 s,
FC 差异通常很小 (r > 0.95); 如果 IA 与 IA2 的差异小于"做 STC 与不做 STC"的差异, 选择就无关紧要, 直接记录为 `unknown` /
`skip` 更诚实。要判断哪一个正确, 看 STC 后相邻层 (奇 / 偶) 之间 tSNR 或高频功率的锯齿是否消失: 正确的顺序消除锯齿,
错误的顺序会加重它 (对 `_space-T1w_desc-preproc_bold.nii.gz` 按原始层索引算逐层 tSNR 即可, 需要 stage 03 的
`KEEP_WORK=yes` 保留 STC 后未重采样的中间文件)。

### 5.7 其他值得看的东西

* stage 08 的 carpet plot (去噪前后): 去噪后仍有跨全脑的竖条 → 运动 / 呼吸残留, 与 `fd_fc_coupling` 高的 run 对应。
* stage 09 QC-FC 的距离依赖: 短距离边的 FC–FD 相关高于长距离 → 运动伪影的典型签名; 合格的策略应把它压平。
* 年龄 / 诊断组之间 `fd_fc_coupling`、`n_retained` 的差异: 组间 FC 差异可能只是运动差异, 报告时要一起给出。

## 6. 文件格式速查

| 文件 | 列 |
| --- | --- |
| `<RUN>_desc-validation.tsv` | `stream strategy atlas metric value` |
| `<RUN>_desc-streamcompare.tsv` | `strategy atlas scope roi metric volume surface value` |
| `group/validation_long.tsv` | `subject run_label group stream strategy atlas metric value` |
| `group/stream_comparison.tsv` | `strategy atlas metric direction n median_volume median_surface median_diff n_surface_higher n_volume_higher wilcoxon_p significant better median_value` |
| `group/strategy_comparison.tsv` | `stream atlas metric direction strategy n median q25 q75 best` |
| `group/fc_typicality.tsv` | `subject run_label group stream strategy atlas n_runs fc_typicality` |
| `atlases/<A>/centroids.tsv` (缓存) | `index x y z` (world mm) |

缺失值一律写 `n/a`; JSON 中为 `null`。


## 2026-09 QC interpretation update

ROI comparisons require unique matching names and align columns explicitly; equal
column counts are not identity evidence. Different names without an explicit mapping
are not compared. Missing or failed required QC is `incomplete`, never `pass`.
Censor vectors must be finite binary values with a valid time axis; nonfinite masked
image signals fail their QC block rather than being replaced by zero.

Group paired inference uses one observation per subject: paired run values are averaged
within subject before Wilcoxon testing. Raw p values are retained; BH q values control
the family of available metric tests within each strategy × atlas. `significant` means
q < 0.05 and remains exploratory. Strategy summaries use subject means and no automatic
best-strategy decision because availability can differ. QC-FC uses mean run Fisher-z
FC and mean run FD per subject, with edge-family BH q values. The distance association
is descriptive because edges are dependent. FC typicality excludes all runs of the
same subject and equally weights the other subjects. Cross-site QC-FC may reflect site
confounding; it is not a causal estimate of motion artifact.

`split_half_r` measures within-run first/second-half FC consistency, not test-retest
reliability. Report the original retained duration and DOF alongside this value.

`dof_remaining` is the algebraic residual dimension (`fit_rows - design_rank`),
not an effective independent sample size. NTRP can have more fitted rows because
it interpolates censored frames; this does not restore observed information.
Retained observations, retained duration and censor fraction remain separate QC
quantities. AFNI model feasibility is a distinct prerequisite from algebraic DOF.

Stage 08 requires the explicit stage-04 censor vector, including when every frame
is retained. Missing, unreadable, nonbinary or mismatched censor vectors make
retained counts/duration unknown and QC incomplete. Censor-dependent signal and
ROI metrics are not computed; independent registration and acquisition metadata
remain available in the report.
