# 各阶段的作用与注意事项（fMRI_Processing v2.2）

本文件逐阶段说明：**为什么要做这一步、实际执行了什么、关键参数、在低质量老数据
（TR 2–3 s、3–4 mm 体素、无场图、无 JSON）上常见的问题、怎么检查结果、输出文件**。
命令与参数均取自 `stages/*.sh` 的实际实现；接口契约见 `docs/DESIGN.md`，
时间序列有效性指标见 `docs/VALIDATION.md`，surface 分支见 `docs/SURFACE.md`。

文中的"实测"数字来自 2026-09-24 在本机（Ryzen 7 9700X，Docker Desktop Hyper-V
VM 12 GB / 16 CPU）上用 `config/datasets/abide_smoke.conf`（synth 模式）对两名 ABIDE
被试的完整运行：`sub-0028744`（ABIDEII-GU_1，STC 已验证）与 `sub-0029150`
（ABIDEII-NYU_2，层序未确认、z 向 FOV 只有 102 mm）。它们只说明流水线在这类数据上
能正确运行、QC 指标落在什么量级，**不构成群体层面的方法学结论**。

执行顺序（由数据流决定，与 `STAGES` 中的书写顺序无关）：

```
数据集级  00 ingest → fetch
每被试    01 anat_recon → 02 anat_prep → 03 func_prep → 04 confounds → 05 denoise
          → 06 surface（可选）→ 07 timeseries → 10 validate → 08 qc
数据集级  10 validate --group → 09 group_qc
```

每个阶段都写 `logs/<sub>/<stage>.log`（含每条执行的命令），并在
`work/<sub>/.done/` 留下参数 hash；参数、上游结果或阶段脚本不变时自动跳过
（见 README §11）。每次运行实际执行的代码会冻结在 `logs/code_<run_id>/`。

---

## 00 ingest（数据集级）

**目的**：把各种原始布局（DPABI 的 `FunImg/T1Img`、BIDS）统一成一个经过校验的
`rawdata/` 与 `manifest.tsv`，并在处理开始前就把每个 run 的 TR、丢弃帧数和
slice timing 决策定下来、写成记录。之后的阶段只读 manifest 与 sidecar，不再猜测。

**做了什么**
1. 发现 run：DPABI 为 `<site>/FunImg/<sub>/*.nii*` 与 `<site>/T1Img/<sub>/*.nii*`
   （site 目录名即 acquisition group）；BIDS 为 `sub-*/[ses-*/]func/<BIDS_FUNC_GLOB>`。
2. 头文件校验：4D、≥ 30 帧、TR 单位（ms/s）、qform/sform code、倾斜角、数据类型、
   z 向 FOV。错误的 run 在 `ingest_report.tsv` 里记 `status=error` 并排除，其余照常。
3. 写规范化副本（体素数据按字节流原样复制，只改 348 字节头）：code 3（Talairach）
   → 1；sform_code 0 时从 qform 复制；**去掉 AFNI 头扩展（ecode 4）**（见下）。
4. 写 sidecar：`RepetitionTime`、`SliceTiming`（仅当 STC 被批准时）、
   `SliceTimingSource`、`SliceTimingEvidence`、`SliceTimingSkipReason`、`DropVolumes`。
5. 写 `manifest.tsv`、`participants.tsv`（`acq_group`）、`ingest_report.tsv`。

**关键参数**：`INPUT_LAYOUT`、`ACQ_TABLE`（站点采集表）、`TASK_NAME`、
`DPABI_FUNC_DIR/DPABI_T1_DIR`、`SUBJECT_LIST`、`DROP_VOLUMES`（表中未写时的默认值）。

**注意事项**
- **层序宁缺毋滥**。STC 只在有证据时做：采集表的 `evidence` 分 A/B/C，`unknown`
  或未确认的厂商推断（如 Siemens 偶数层"先偶后奇"）默认 `skip`。层序奇偶弄反等于
  把每层时间错位约 TR/2，比不做 STC 更糟。ABIDEII-NYU_2（34 层、自定义序列）因此跳过。
- **表中层数/TR 与头文件不一致时不做 STC**，也不会"修正"，只在报告里写 warning。
- **AFNI 头扩展会覆盖 NIfTI 头**：ABIDEII-GU_1 的原始文件带 AFNI 扩展，里面的 TR
  是 1 s（NIfTI 头是正确的 2 s）；AFNI 程序优先读扩展，导致后续 AFNI 生成的文件
  都是 TR 1 s，STC 因"TR 不一致"被静默跳过。现在 ingest 在副本里去掉该扩展，
  `header_changes` 会写明（本次测试中发现并修复）。
- ABIDE 的头文件原点常是通用值，**不能用于 EPI→T1 初始化**（阶段 03 总是全局搜索）。
- `DropVolumes`：GU_1 在共享前已丢过 2+2 帧，表中设 2；其余默认 4（TR 2 s 下 8 s）。
  阶段 03 还会用全局信号检测非稳态帧，检测数多于丢弃数时给出 warning。
- z 向 FOV < 110 mm 标 `short z-FOV`（NYU_2 为 102 mm），这类数据的小脑、颞极、
  顶部皮层常缺失，后面会表现为 ROI 覆盖率不足。

**怎么检查**：`rawdata/ingest_report.tsv` 逐行看 `status`、`tr_header`/`tr_used`、
`stc_decision`/`stc_reason`/`evidence`、`drop_volumes`、`header_changes`、`warnings`。
实测：GU_1 → `apply (IA, A)`，NYU_2 → `skip (unknown, C)` 且 `short z-FOV (102 mm)`。

**输出**：`rawdata/manifest.tsv`、`participants.tsv`、`ingest_report.tsv`、
`rawdata/sub-*/{anat,func}/`（副本 + sidecar + `*.ingest.json` 复制记录）。

---

## fetch（数据集级，一次性）

**目的**：把模板/图谱等外部资源一次下载到 `$RESOURCE_DIR`，之后可离线运行。

**做了什么**：TemplateFlow 的 fsLR-32k 球面/midthickness/内侧壁、MNI152NLin6Asym
2 mm 的 Schaefer 分区与标签；HCPpipelines 的 `Atlas_ROIs.2.nii.gz` 与
`{L,R}.atlasroi.32k_fs_LR.shape.gii`；CBIG 的 Schaefer fsLR `dlabel`。
`--check` 只报告缺失（退出码 0 = 齐全），编排器在缺资源时自动下载。

**注意事项**：需要一次网络；`RESOURCE_DIR` 放在持久位置（Docker 命名卷或集群共享目录），
集群上先在联网机器准备好再拷贝。实测：volume 资源 3 s、surface 资源 21 s 下载完成。

---

## 01 anat_recon（FreeSurfer，仅 `ANAT_MODE=freesurfer`）

**目的**：得到皮层表面（surface 分支必需）、aseg 分割和 bbregister 需要的白质边界。

**做了什么**：`recon-all -sd $FS_DIR -s <sub> -i <T1> -all -parallel -openmp $NTHREADS
$RECON_FLAGS`（任一维 FOV > 256 mm 时自动加 `-cw256`）。完成判据：
`scripts/recon-all.done` + `surf/{lh,rh}.pial` + `mri/aseg.mgz`；未完成的目录会续跑；
存在 `scripts/IsRunning*` 时拒绝触碰（防止两个进程写同一被试）；先探测 `FS_DIR`
能否建立符号链接。

**注意事项**
- **最耗时的一步**：每被试数小时。`FS_DIR` 必须在 Linux 文件系统或 Docker 命名卷上：
  Windows 挂载目录不支持 recon-all 需要的符号链接，且小文件 I/O 极慢。
- 容器被强行终止后会留下 `IsRunning` 文件，确认没有进程在跑后手动删除再续跑。
- 1.2–1.4 mm 矢状位 T1（NYU、UM_1）与儿童脑的表面质量需要看 Euler 数/孔洞数
  （`holes_total`，warn 100 / fail 200）和阶段 08 的表面图。
- `synth` 模式不跑 recon-all，因此没有 surface 分支。

---

## 02 anat_prep

**目的**：T1w 参考像、脑掩膜、组织分割、用于噪声信号的 WM/CSF 掩膜，以及 T1w→模板的
非线性形变。所有变换都是世界坐标（ITK/LTA），不做 `fslreorient2std`。

**做了什么**
- freesurfer 模式：`nu.mgz` → T1w，`brainmask.mgz` 二值化 + 填洞 → 脑掩膜，`aseg.mgz` → 分割。
- synth 模式：`N4BiasFieldCorrection` → `mri_synthstrip`（脑掩膜）→ 把 T1 **裁剪到
  脑框 + 15 mm**（`fslstats -w` + `fslroi`，世界坐标不变）→ `mri_synthseg
  $SYNTHSEG_FLAGS`（默认 `--robust`）→ 标签回到 T1 网格。
- 组织掩膜（在 1 mm 网格上腐蚀，迭代次数≈毫米）：WM = 2/41 腐蚀 `WM_ERODE`（2）；
  CSF = 侧脑室 4/43 腐蚀 `CSF_ERODE`（1），体素 < 50 时逐级放宽并记录；GM 为皮层与
  皮层下灰质标签；另有不腐蚀的全部白质 `WMbbr`（FLIRT-BBR 用）。
- `antsRegistrationSyN.sh`（`NORM_QUALITY=precise`）或 `...Quick.sh`（`quick`），
  brain-to-brain 到 1 mm MNI152NLin6Asym，固定随机种子 `ANTS_SEED`；有限差分 Jacobian。

**注意事项**
- **内存**：整颅视野的 `SynthSeg --robust` 在 12 GB 的 Docker VM 里 OOM
  （TensorFlow `ResourceExhaustedError`）；裁剪后两次实测峰值 7.0 与 7.8 GiB（每 15 s 采样），SynthSeg 约 23 s 完成。
  内存更小时设 `SYNTHSEG_FLAGS=`（不带 `--robust`）。
- 儿童/萎缩脑的侧脑室很小，腐蚀后 CSF 掩膜可能只剩几十个体素（GU_1 儿童：1 mm 网格
  1899 体素，BOLD 网格 46 体素）；CSF 信号因此噪声大，但不会混入灰质。
- 非 MNI152NLin6Asym 模板时图谱也必须换成同一空间的版本，否则 parcel 错位几毫米。

**怎么检查**：`anat/sub-X_desc-anatqc.json` 与报告的 Anatomy 部分：`norm_dice`
（warn 0.93 / fail 0.88）、`template_corr`、`jacobian_nonpos_frac`（应为 0，> 0 表示
形变折叠）、Jacobian p01/p99；图：掩膜应贴合脑表面、WM/CSF 掩膜完全在白质/脑室内。
实测（quick SyN）：norm Dice 0.983/0.981，模板相关 0.77/0.76，无折叠。

**输出**：`derivatives/sub-X/anat/`（T1w、脑掩膜、aseg、`label-{WM,CSF,GM,WMbbr}`、
`xfm/T1w_to_MNI_*`、模板空间 T1w、`desc-anatqc.json`）。

---

## 03 func_prep（每个 run）

**目的**：把原始 BOLD 变成两份"只插值一次"的预处理序列（T1w 空间 3 mm 与模板空间
2 mm），同时估计噪声信号所需的一切。这是最关键的阶段：运动、层时、配准都在这里。

**做了什么（按顺序）**
1. 丢弃前 `DropVolumes` 帧（float32 副本）；剩余 < 50 帧拒绝处理；非稳态帧检测。
2. **原始 QC**：在任何清洗前做 `3dToutcount`（离群比例）与 `3dTqual`，避免后续步骤掩盖坏帧。
3. `3dDespike -NEW`（`DESPIKE=yes`），记录被修改的体素-时间点比例。
4. **运动估计**（在 STC 之前）：离群最少的一帧作初始参考 → `mcflirt` 第一遍 →
   其时间中值为 boldref → `mcflirt -mats -plots -rmsrel -rmsabs` 第二遍。只保留矩阵和参数。
5. **STC**：仅当 sidecar 有经过校验的 `SliceTiming`：`3dTshift -TR <TR>s -tzero
   <采集中点> -tpattern @slice_timing.1D -quintic`。校验读的是 rawdata 头文件本身。
6. boldref `N4` → `mri_synthstrip` EPI 掩膜（失败时 `3dAutomask`）。
7. **配准**（全局搜索，不信任头文件原点）：freesurfer 模式 `mri_coreg` →
   `bbregister --bold`；synth 模式 FLIRT `corratio` ±90° 搜索 → FLIRT-BBR。
   BBR 代价 > `BBR_MAX_COST`（0.9）或相对初值移动 > `BBR_MAX_DISP_MM`（15 mm）时拒绝、
   退回初值并记录 `bbr_rejected`。
8. **单次插值**：每帧用 `antsApplyTransforms -n LanczosWindowedSinc` 一次性施加
   运动 ∘ EPI→T1w（∘ T1w→模板），得到 T1w 3 mm 网格与模板 2 mm 网格两份序列。
9. 掩膜 = T1 脑掩膜 ∩ 膨胀的 EPI 支持区 ∩ 全程有信号的体素；T1w 空间组织掩膜
   （阈值 0.9，体素 < `MIN_TISSUE_VOX` 时逐级放宽并记录）；一个全局缩放因子（中位数 → 10000）。
10. **强制自检**：EPI 掩膜与 T1 脑掩膜的 Dice ≥ 0.5、重采样序列均值与 boldref 相关 ≥ 0.9，
    否则整个 run 失败（变换链方向错误时会被拦下，而不是产出看似正常的错图）。

**关键参数**：`DROP_VOLUMES`、`DESPIKE`、`STC`（auto/require/off）、`STC_INTERP`、
`EPI_MASK_METHOD`、`BBR_MAX_COST`、`BBR_MAX_DISP_MM`、`FUNC_T1W_RES`、`MNI_RES`。

**注意事项**
- 低对比度 EPI 上 BBR 会"无报错地失败"，所以看 `bbr_cost`、`bbr_vs_init_mm` 和图，
  而不是只看是否报错。
- 倾斜扫描（NYU 23°、TRINITY 12.5°）不做 header-only deoblique，倾斜由世界坐标变换处理。
- 无场图：眶额、颞极的变形与信号丢失无法校正，只能在覆盖率和 dropout 上体现。

**怎么检查**：`func/<RUN>_desc-prep_info.json` 与报告的 Acquisition & provenance、
Registration 部分：`stc_applied/stc_reason`、`coreg_method`、`bbr_cost`、
`bbr_vs_init_mm`、`coreg_dice`（warn 0.90 / fail 0.80）、`hmc_consistency_r`；
EPI→T1 图中蓝色白质轮廓应与 EPI 灰白质交界吻合，红色轮廓显示 FOV 与 dropout。
实测：两例 FLIRT-BBR 均被接受（代价 0.46/0.48，偏离初值 2.6/4.7 mm），coreg Dice
0.957/0.962，一致性 r ≥ 0.999；GU_1 做 STC 后 GM tSNR 由 50.5 变为 54.4；每 run 约 3.5 min。

**输出**：`func/` 下 `desc-prep_info.json`、运动参数与 RMS、离群/质量序列、
`space-T1w_*` 与 `space-MNI152NLin6Asym_res-2_*` 的 boldref、掩膜与 `desc-preproc_bold`。

---

## 04 confounds（每个 run）

**目的**：一次性算出所有候选噪声回归量与逐帧质量指标，供不同去噪策略按需选择。

**做了什么**（T1w 空间序列，只读掩膜内体素）：6 个运动参数及导数/平方（24P）；
WM、CSF、全局信号及其展开；aCompCor（WM、CSF 各 `ACOMPCOR_N`=5 个主成分，PCA 前
先做 `HIGHPASS_SEC`=128 s 的 DCT 高通）；cosine 基；FD（Power，50 mm 半径）与
Jenkinson RMS；DVARS 与 standardized DVARS；离群比例；censor 向量
（FD > `CENSOR_FD`=0.5 mm；可选 `CENSOR_PREV` 前一帧、`CENSOR_NEXT` 之后 N 帧、
`CENSOR_DVARS`，最后 `CENSOR_MIN_SEGMENT` 把短于 N 帧的保留段也删掉）。

**注意事项**
- FD 阈值与 TR 有关：同样的运动在 TR 3 s 下 FD 更大；儿童/ASD 组头动更大，
  censoring 比例本身就是需要在组间报告的协变量。
- 被 censor 的帧不能进入时间滤波；阶段 05 在同一次投影里处理（见下）。
- 每删一帧就少一个自由度。带通时 FD > 0.2 mm 会让 150 帧的 run 自由度变成负数，
  所以带通为默认时保持 0.5 mm；更严格的删帧要配合高通，并作为敏感性分析（README 9.1 有实测表）。

**怎么检查**：报告的 Motion 部分：`fd_mean`（warn 0.2 / fail 0.5 mm）、
`pct_censored`（warn 20 / fail 50 %）、`minutes_retained`（< `MIN_RETAINED_MIN`=4 min 标记）、
FD–DVARS 相关、最长未 censor 片段。实测：FD 均值 0.17/0.13 mm，censor 2%/0.6%。

**输出**：`desc-confounds_timeseries.tsv/.json`、`desc-censor.1D`（1 保留 / 0 剔除）。

---

## 05 denoise（每个 run × 每个策略）

**目的**：用**一次联合投影**同时去除噪声回归量、多项式趋势、带外频率和被 censor 的帧
（`3dTproject`），避免"先滤波后回归"把噪声重新引入通带（Hallquist 2013、Lindquist 2019）。

**做了什么**：按策略选列 → 中心化、单位化的 `regressors.1D` → 数据减去时间均值 →
`3dTproject -ort … -polort 2 -dt <TR> -passband 0.01 0.1 [-censor … -cenmode NTRP]`，
作用于模板空间序列（`SURFACE=yes` 时也作用于 T1w 空间序列供阶段 06）。
`_denoise.json` 记录回归量个数与剩余自由度；可选 `3dBlurInMask` 平滑副本（`SMOOTH_FWHM`）。

**策略**（名称说明做了什么，`gsr` = 做了全局信号回归）：`wmcsf24`、`wmcsf24gsr`、
`36p`、`acompcor`、`acompcorgsr`、`legacy8`、`legacy9gsr`（与 v1 的对应关系见 README）。
`acompcor` = 12 个运动参数 + WM/CSF 各 5 个 aCompCor 成分 + aCompCor 之前去掉的全部
DCT 余弦项（`cosine_XX`，150 帧时 4 个）：成分是在余弦去除之后估计的，模型里必须包含同一组
余弦；有限长度的 DCT 与 3dTproject 的傅里叶阻带基并不相同，所以不算重复（设计矩阵满秩）。

**注意事项：自由度预算**
- TR 2 s 时 0.01–0.1 Hz 带通本身就消耗约 N × 0.64 个自由度。实测 150 帧、删 3 帧的 run：
  `wmcsf24` 26 个回归量 → 剩余 **21**；176 帧、删 1 帧 → 33。剩余自由度只数保留帧
  （NTRP 的插值帧参与拟合但不提供信息；v2.1 曾把它们算进去）。`36p` 或加大 censoring 后，
  150 帧以下的 run 很容易低于 `MIN_DOF`=15（只标记，不中止），此时 FC 估计不稳定。
  短 run 可改 `FILTER_MODE=highpass` 或用回归量更少的策略。
- GSR 取舍：GSR 对全局运动/呼吸伪影最有效，但改变 FC 分布（产生负相关），两类结果
  建议并行报告，不要跨类比较 network contrast。
- 平滑只作为最后一步的可选副本；ROI 时间序列永远取自未平滑数据。

**怎么检查**：报告的 Denoising strategies 表：`dof_remaining`、`tsnr_gain`、
`variance_removed`、`fd_dvars_corr_post`。`fd_dvars_corr_post` **为正**说明头动造成的
信号跳变还在（去噪不足，例如只回归 6 个头动参数时 GU_1 为 +0.20）。**为负**说明高 FD 帧
的残差比安静帧还小：头动扩展回归量（及与头动相关的 aCompCor 成分）在这些帧取极值、
杠杆高，回归几乎把这些帧拟合掉，相当于软删帧，这些帧的神经信号也一起去掉；带通让
剩余自由度很少时会放大这一效应。实测（用 numpy 重建 3dTproject 投影，与流水线输出
一致）：GU_1 `wmcsf24` −0.42，去掉带通 −0.21，去掉导数项仍为 −0.23；同样 126 列、
24 个自由度的随机回归量为 +0.29，只做带通为 +0.56。所以负值来自头动回归量本身，
不是自由度少造成的，也不只是导数项。短 run 用 24 参数模型时 −0.2 ~ −0.45 很常见，
本身不算失败；若同时 DOF 很低，优先靠 censoring 处理高运动帧，或改
`FILTER_MODE=highpass` 保留自由度。

**输出**：`desc-<S>_regressors.1D`、`desc-<S>_denoise.json`、
`space-MNI152NLin6Asym_res-2_desc-<S>_bold.nii.gz`（及可选平滑副本）。

---

## 06 surface（可选：`SURFACE=yes`，需 freesurfer 模式、`MNI_RES=2`）

**目的**：在皮层表面上采样 BOLD（沿皮层带取加权平均、平滑不跨越脑沟），输出
HCP/fMRIPrep 兼容的 fsLR-32k CIFTI（91k 灰质坐标：皮层顶点 + 皮层下体素）。

**做了什么**：FreeSurfer 表面转 GIFTI（`mris_convert --to-scanner`，球面不加）、
midthickness、皮层 ROI；HCP goodvoxels（剔除变异系数异常的体素，σ 5 mm、系数 0.5）；
`-volume-to-surface-mapping -ribbon-constrained`（层厚 ≥ 3.5 mm 时 `-voxel-subdiv 7`，
否则 5）；10 mm 最近邻膨胀填洞；`-metric-resample ADAP_BARY_AREA` 到 fsLR-32k；
皮层下取自 2 mm 模板空间序列与 `Atlas_ROIs.2`；预处理序列与每个策略各一个 dtseries，
可选 `-cifti-smoothing`（`SURF_SMOOTH_FWHM`）。另写 `desc-sampled_mask.dscalar.nii`：
只由膨胀填上邻居值的 vertex 为 0，阶段 07 不把它们算作覆盖。

**注意事项**：3–4 mm 体素大于皮层厚度，部分容积严重；无场图时眶额/颞极错位；
表面质量取决于 recon-all（看 Euler 数）。surface 流不一定"更好"，应由阶段 10
的配对比较判断（详见 `docs/SURFACE.md`）。

**怎么检查**：`<RUN>_desc-surfqc.json`：`pct_goodvoxels_excluded`、`pct_badvertices`、
`tsnr_cortex_median`；报告中的表面 tSNR 图。

---

## 07 timeseries

**目的**：得到最终的 ROI 时间序列与 FC，同时保证每个 ROI 的数值都有足够的数据支撑。

**做了什么**：图谱按世界坐标重采样到 BOLD 网格（`antsApplyTransforms -n GenericLabel
-t identity`）；ROI 均值只用"在脑掩膜内、全程有限、非常数"的体素；有效体素比例 <
`MIN_ROI_COVERAGE`（0.5）的 ROI 整列为 `n/a`（**不插补**）；FC 为保留帧上的 Pearson r；
另外输出去噪前的 ROI 均值（用于 ROI tSNR）；CIFTI 用 `wb_command -cifti-parcellate`。

**注意事项**
- 实测：NYU_2（z-FOV 102 mm）右侧颞极 `RH_Limbic_TempPole_1` 覆盖率 0.36 → `n/a`，
  左侧 0.63 保留。组分析时 `n/a` 的边要按缺失处理，不能填 0。
- `parcellations/ThomasYeo_100.nii` 实为 **Schaefer2018 100 Parcels 7 Networks**
  （同一标签，LAS 存储）：两者在世界坐标上逐体素一致，结果完全相同，同时配置两者是冗余的。
- 图谱必须与 `TEMPLATE_NAME` 同一空间。

**输出**：`*_atlas-<A>_desc-<S>_timeseries.tsv`（表头为 ROI 名称）、`_coverage.tsv`、
`_connectivity.tsv`、`_timeseries.json`；`desc-preproc_timeseries.tsv`；surface 流为 `space-fsLR_*`。

---

## 10 validate（每被试；`--group` 数据集级）

**目的**：回答"最终时间序列好不好、volume 与 surface 哪个更好、哪个去噪策略更合适"。
tSNR 单独升高不算证据（过度去噪也会让它升高），所以同时看可靠性、网络结构和残余运动耦合。

**指标**（定义见 `docs/VALIDATION.md` 与报告 glossary）：ROI tSNR、variance removed、
split-half FC 可靠性、network / homotopic / DMN contrast、低频功率比、FD–FC 耦合、
残余全局信号、NaN ROI 数、保留帧与剩余自由度；两条流都存在时逐 parcel 配对比较与
FC 相似度；组级为配对差值、Wilcoxon（≥ 6 名被试）、BH 校正、FC typicality（≥ 4 个 run）。
组级比较只用纳入的 run（见 09 的纳入标准）；`validation_long.tsv` 保留全部 run 并标 `included`。

**实测（2 名被试，只能描述）**：Schaefer-100 上 split-half r 约 0.54（两种策略相近）；
network contrast 0.98（wmcsf24）/ 1.05（wmcsf24gsr）；FD–FC 耦合 0.22 / 0.26。
两名被试不足以得出策略优劣；正式比较需要全体数据（Wilcoxon 需 ≥ 6、QC-FC 需 ≥ 10）。

---

## 08 qc（每被试）

**目的**：把每个 run 的所有 QC 指标、红黄绿标记和图集中到一个离线可看的 HTML 报告。

**做了什么**：`<RUN>_desc-qc_metrics.json`（带 pass/warn/fail）、`roiqc.tsv`、图；
`derivatives/sub-X.html`（图片内嵌，每节有中文阅读说明）。缺失的输入只显示 `n/a`，
QC 本身不修改任何结果。

**报告里依次看什么**：Summary 表（overall 与各 flag）→ Acquisition & provenance
（STC、丢帧、配准方法）→ Anatomy（掩膜、组织、模板配准）→ Registration
（EPI→T1、EPI→模板）→ Motion（FD/DVARS 与 censor 帧）→ Carpet（去噪前与 FD 峰对齐的
竖条纹在去噪后应减弱）→ tSNR 图 → Confounds 相关 → Denoising 表 → FC 矩阵与直方图
（沿对角线的网络块、GSR 后分布居中）→ ROI 覆盖率与 ROI tSNR → Time-series validation。

阈值（`default.conf`，可改）：FD 均值 0.2/0.5 mm；censor 20/50 %；GM tSNR 40/20；
coreg Dice 0.90/0.80；norm Dice 0.93/0.88；表面孔洞 100/200。实测两例 overall 均为 pass。

---

## 09 group_qc（数据集级）

**目的**：在群体层面找离群 run，并检验残余运动是否仍在驱动 FC。

**做了什么**：`group_qc.tsv`（每 run 一行，站点内 ≥ 5 个 run 时按站点做 robust z，
|z| > 3 标记）；站点分布图；QC-FC（边的 FC 与平均 FD 的相关、显著边比例、
中位 |r|、距离依赖）需 ≥ `QCFC_MIN_SUBJECTS`（10）名被试；`group_report.html`。

**纳入与剔除**：按 conf 中事先写定的 `EXCLUDE_*` 标准判断每个 run（平均 FD、最大 FD、
FD > 0.2 mm 的比例、保留分钟数、按策略的剩余自由度、run 级 QC fail），写入
`inclusion.tsv`（`included`、`included_<S>` 和原因）。不删除任何数据；QC-FC 与阶段 10 的
组级比较只用纳入的 run。报告的"纳入与剔除"一节列出被剔除的 run 与原因。
设置 `PHENOTYPE_TSV` 后按组（例如诊断）比较剔除人数和剩余被试的平均 FD。

**注意事项**：ASD 与对照、儿童与成人的头动系统性不同；剔除越严，剩下的 ASD 样本越偏向
头动小、症状轻的被试。报告各组剔除人数，比较剩余被试的平均 FD，并把平均 FD 作为组分析的协变量。

---

## 跑完一个数据集后按什么顺序看 QC

1. `logs/status_<run>.tsv` 与终端摘要：有没有失败的阶段（失败只影响该被试）。
2. `rawdata/ingest_report.tsv`：每个 run 的 TR、STC 决策与证据、丢帧、header 修改、warning。
3. `derivatives/group/group_report.html`：离群 run（站点内 robust z）、FD/censor/tSNR 分布。
4. 标记为 warn/fail 的被试的 `derivatives/sub-X.html`：先看配准图，再看 Motion 与 Carpet。
5. `derivatives/group/validation_report.html`：策略与 volume/surface 比较（被试足够多时）。
6. QC-FC（`qcfc_*.tsv`）：选定策略后残余运动对 FC 的影响。
7. `derivatives/group/inclusion.tsv` 与组报告的"纳入与剔除"一节：确认剔除原因；标准写在 conf
   的 `EXCLUDE_*` 里，要在看结果前定好，并在文章中报告（含各组剔除人数）。

## 本次本地测试中发现并修复的问题

| 问题 | 表现 | 修复 |
|---|---|---|
| SynthSeg `--robust` 在 12 GB VM 中 OOM | 阶段 02 失败 | 先裁剪到脑框 + 15 mm；新增 `SYNTHSEG_FLAGS` |
| GU_1 原始文件的 AFNI 扩展写着 TR 1 s | 已验证层序的站点被静默跳过 STC | ingest 去掉 AFNI 扩展；STC 校验读 rawdata 头 |
| 没有任何被试到达 QC 时 | `group_qc` 抛 `TypeError` | 缺列时 QC-FC 为空并正常结束；新增回归测试 |
| 运行期间修改仓库脚本 | 正在执行的阶段读到错位内容而失败 | 每次运行冻结代码到 `logs/code_<run_id>/` |
| Windows 下 CRLF 检查失效 | 18 个文件带 CRLF 未被发现 | 改用 `tr` 计数；已恢复为 LF |
| 测试在 UTF-8 locale 下误判 | 容器中 3 个 bash 测试失败 | `LC_ALL=C sort` |
| surface 覆盖率被膨胀虚高 | FOV 外的 parcel 在 surface 流里也有"数据" | 阶段 06 输出 `desc-sampled_mask`，阶段 07 不计入 |
| CBIG dlabel 与 label 表有 19 个名称不同 | 阶段 10 报 ROI identities differ | 按 label key 对齐名称，要求半球与网络一致 |
| `mris_convert -c` 改写输出名 | 阶段 06 找不到 thickness 文件 | 先写 FreeSurfer 风格文件名再移动 |
| NTRP 下剩余自由度把删掉的帧也算进去 | 高头动 run 的 DOF 警告失灵 | `dof_remaining` 只数保留帧；另记 `algebraic_dof` |
| 去噪后 FD–DVARS 为负的解释不准确 | 可能误导策略选择 | 对照实验：来自头动回归量在高运动帧的高杠杆；说明已改写 |
| 容器内测试用了 FSL 自带的 Python | 缺 nilearn，1 个测试被跳过 | 测试优先用 `PYTHON_BIN` 或 conf 默认值 |

## v1 → v2 每一步的变化

| 步骤 | v1 | v2 |
|---|---|---|
| 输入 | 只认 BIDS + JSON | DPABI/BIDS，采集表按证据决定 STC，头文件规范化 |
| 解剖 | recon-all 的 brain.mgz + FAST | aseg 或 SynthSeg，1 mm 上腐蚀，ANTs SyN 到 1 mm 模板 |
| 运动/层时 | 运动估计后再 STC，插值两次 | STC 前估计运动，与配准一起单次插值 |
| 配准 | FLIRT BBR，失败才回退 | bbregister/FLIRT-BBR + 代价/位移检查 + 自检 |
| 平滑 | 回归前 6 mm | 最后一步、可选；ROI 用未平滑数据 |
| 去噪 | 6P + WM/CSF(+GS)，命名反了 | 多策略、联合投影、censoring、自由度记录，`gsr` 即做了 GSR |
| QC | 无 | 每被试 HTML、组级 QC-FC、时间序列有效性、volume/surface 比较 |
