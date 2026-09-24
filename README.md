# fMRI_Processing v2.2

面向低质量传统数据（TR 2–3 s、3–4 mm 体素、无 fieldmap、多数无 JSON sidecar：ABIDE、ADNI、DPABI 布局队列）的静息态 fMRI 预处理流水线。bash 阶段脚本 + 一个小型 Python 包（`py/fmriproc`），在容器 `zhaochang07/myubuntu:neuro-v2` 内运行。

接口契约是 `docs/DESIGN.md`（文件名、目录、JSON 键、处理顺序、`lib/common.sh` API）；验证指标见 `docs/VALIDATION.md`，surface 分支见 `docs/SURFACE.md`，Docker/Singularity 的构建与运行见 `docker/README.md`。

本轮升级限定当前 ABIDE 单回波静息态流程，所有测试在本地进行。v2.1 修复了 NTRP 拟合行与 DOF 定义、aCompCor DCT 基、具名 ROI 对齐、缺失 QC 的状态及重复 run 的伪重复；去噪输入中心化按 run × space 复用。实现通过合成测试不等于真实数据或跨软件版本验证，末尾记录当前验证边界。v2.2 在两名 ABIDE 被试上完成了端到端测试并修复了其中发现的问题；剩余自由度只数保留帧，新增删帧选项和组水平的纳入标准（§9.1）。版本变化见 `CHANGELOG.md`。

---

## 1. 这条流水线解决什么问题

- 输入是每个被试一个 T1w + 一个或多个 BOLD run，**没有可信的 slice timing、没有 fieldmap、头文件原点不可靠**。
- 输出是可直接做 FC / 图论分析的 ROI 时间序列（volume，可选 fsLR-32k surface），以及能说明“这份数据可不可用”的 QC 指标和报告。
- 设计原则（`docs/DESIGN.md` §1，不可回退）：不猜 slice order；运动在 STC 前估计、与 EPI→T1、T1→MNI 合成后 **只插值一次**；数据不乘 mask；nuisance 信号取自未平滑数据、平滑放最后；滤波 + 趋势 + nuisance + censoring 用 `3dTproject` 一次联合投影；BBR 结果必须与初始化比较；ROI 覆盖不足给 `NaN`；每个阶段记录 JSON provenance 并可按参数 hash 重启；去噪策略按其行为命名（`...gsr` = 做了 GSR）。

## 2. v1 → v2 变化一览

| 方面 | v1（`legacy/`） | v2 | 原因 |
|---|---|---|---|
| 结构 | `main.sh` + `FC_step0–6` + `pipeline_helpers.sh` | `run_pipeline.sh` + `stages/00–10` + `lib/common.sh` + `config/` | 数据集只改一个 conf；阶段可单独重跑 |
| Slice timing | 按厂商/站点假设 | 只用**已验证**的 timing（acquisition TSV 或 sidecar），否则跳过并记录 `stc_reason` / `SliceTimingEvidence` | 错误的 interleave parity 比不做 STC 更糟 |
| 运动校正与重采样 | mcflirt 输出再 FLIRT/FNIRT，多次插值 | 运动在 STC 前估计（两遍 mcflirt，中位数参考）；`antsApplyTransforms` LanczosWindowedSinc **单次插值**到 T1w 网格和模板网格 | 少一次插值 = 少一次平滑和 ringing |
| 方向 / 头文件 | `fslreorient2std`、header deoblique | 不改体素存储，全部用世界坐标 ITK/LTA 变换 | ABIDE 头文件原点是通用值，header 对齐不可信 |
| 掩膜 | 数据乘 mask | mask 单独成文件，数据永不被乘 | 不可逆；下游无法恢复 |
| EPI→T1 配准 | FLIRT 6-dof 互信息 | `mri_coreg` → `bbregister --bold`（freesurfer 模式）或 FLIRT 6-dof → FLIRT-BBR（synth 模式）；BBR 位移 > `BBR_MAX_DISP_MM` 或 cost > `BBR_MAX_COST` 即回退到初始化并标记 | 低对比 EPI 上 BBR 会失败，必须核对 |
| T1→MNI | FNIRT 到 3 mm 标准脑 | ANTs SyN（`precise`/`quick`）到 1 mm MNI152（FSL = `MNI152NLin6Asym`），Dice、Jacobian、模板相关 QC | 更稳、可量化 |
| 组织掩膜 | FAST + `tissuepriors/` | aseg / SynthSeg 标签，在 1 mm 网格上腐蚀（`WM_ERODE`、`CSF_ERODE`），BOLD 网格上体素不足则放宽并记录 | 先验不适合儿童/老年脑 |
| Nuisance 与滤波 | FEAT `.fsf` 模板、分步进行 | fMRIPrep 列名的 confounds TSV + FD/DVARS + aCompCor + censor 向量；`3dTproject` 一次投影 | 分步滤波与回归会重新引入噪声 |
| GSR 命名 | `NoGRS` = **做了** GSR，`Retain_GRS` = 未做 | `wmcsf24` / `wmcsf24gsr` 等，`gsr` 后缀 = 回归了全局信号 | v1 命名与实际相反 |
| 平滑 | nuisance 之前 | 最后一步、可选副本（`SMOOTH_FWHM`）；ROI 序列永远来自未平滑数据 | 平滑会把 WM/CSF 信号混进 GM |
| QC | `QC_nor` 截图 | 每 run `desc-qc_metrics.json` + 每被试自包含 HTML + 组级 `group_qc.tsv` / QC-FC + 阶段 10 时间序列有效性指标 | 需要能量化“数据可不可用” |
| Surface | 无 | 可选 fsLR-32k CIFTI 分支（阶段 06），与 volume 流逐 parcel 对比（阶段 10） | 用户需要 surface 输出并比较优劣 |
| 重启 | `START_STEP` | 每阶段参数 hash 标记，改参数/脚本/上游自动失效；`--force` | 只重算该重算的 |
| 并行 | GNU parallel + `set -e`，一个被试失败整批死亡 | 每被试独立 worker，状态表 `logs/status_<ts>.tsv`，退出码 1 但其余被试继续 | v1 的已知 bug |

## 3. 环境要求

- **主机**：Windows 11 + Docker Desktop（Hyper-V 或 WSL2 后端均可）+ PowerShell 7（`pwsh`）；或 Linux/macOS/Git Bash（`docker/run_docker.sh`）。仓库须以 **LF** 检出（`.gitattributes` 已强制；若曾用 `core.autocrlf=true` 检出，见故障排除）。
- **镜像**：`zhaochang07/myubuntu:neuro-v2`（约 79 GB，Ubuntu 22.04）：FSL 6.0.7、AFNI 25.3、FreeSurfer 7.4.1（含 SynthStrip/SynthSeg、bbregister）、ANTs 2.6、Connectome Workbench 2.1、GNU parallel，Python 环境 `/opt/micromamba/envs/neuro`（nilearn 0.12、nibabel 5.3、templateflow、jinja2、pandas、scipy、sklearn、matplotlib）。PATH 只在登录 shell 里设置，所有脚本必须通过 `bash -lc` 运行（启动脚本已处理）。
- **FreeSurfer license**：默认 `%USERPROFILE%\Desktop\license.txt`，只读挂载到 `/opt/freesurfer/license.txt`。synth 模式也需要（阶段 03 的 SynthStrip EPI mask 路径会检查）。
- **内存**：冒烟测试 ≥ 8 GB（conf 里 `MIN_MEM_GB=6` 是下限保护）；recon-all / precise SyN / `mri_synthseg --robust` 建议 16 GB。启动脚本会读 `docker info` 并在低于阈值（`-MinMemoryGB`，默认 6）时警告。
- **资源预算**：重阶段检查 `N_JOBS × NTHREADS` 和 `N_JOBS × MIN_MEM_GB`，以进程 CPU affinity、cgroup 和 Slurm 的较小限制为准；`CPU_BUDGET`、`MEMORY_BUDGET_GB` 可进一步设限。线程数不会被静默改写；预算不足时应降低并发或扩大实际分配。
- **磁盘**：Docker 命名卷承载 `freesurfer/`（约 1 GB/被试）、`work/`（每 run 数 GB，阶段结束后按 `KEEP_WORK` 清理）、`resources/`（模板与图谱，数百 MB）；`derivatives/`、`rawdata/`、`logs/` 写到主机的 `-Out` 目录。
- **网络**：首次运行需要下载模板/图谱（TemplateFlow、GitHub raw），见 `stages/fetch_resources.sh`。
- **主机测试**（可选）：Python 3.11 + numpy/scipy/pandas/nibabel/nilearn/sklearn/matplotlib/jinja2 即可运行 `tests/run_tests.sh`，不需要神经影像工具。

## 4. 快速开始（Windows 11 + Docker）

测试数据：`E:\ASD\test_abide_ASD`（DPABI 布局，7 个 ABIDE 站点 × 2 被试，TR 2 s，全部 LAS，无 sidecar）。以下命令在仓库根目录的 PowerShell 7 中执行。

### 4.1 准备

```powershell
cd E:\ASD\fMRI_Processing
docker info --format '{{.MemTotal}}'          # 字节；冒烟测试至少 8 GB，recon-all 建议 16 GB
Get-Help .\docker\run_docker.ps1 -Detailed    # 全部参数
```

Docker 内存：Docker Desktop → Settings → Resources → Advanced → Memory（Hyper-V 后端在这里改；WSL2 后端改 `%USERPROFILE%\.wslconfig` 的 `[wsl2] memory=16GB` 后 `wsl --shutdown`）。在 31 GB 主机上给 VM 分配 20 GB 曾导致 VM 无法启动，8–12 GB 是本机可行的范围。

启动脚本做的事：仓库只读挂到 `/opt/fmriproc`，`-Data` 只读挂到 `/data`，`-Out` 挂到 `/out`，license 只读挂到 `/opt/freesurfer/license.txt`；三个**命名卷** `<prefix>_freesurfer`、`<prefix>_work`、`<prefix>_resources` 分别挂到 `/out/freesurfer`、`/out/work`、`/out/resources`（Windows bind mount 慢且不能放 recon-all 的符号链接）；然后执行 `bash -lc "bash /opt/fmriproc/run_pipeline.sh -c /opt/fmriproc/<Config> <其余参数>"`。docker 命令会先打印再执行；`-PrintOnly` 只打印不执行。

### 4.2 冒烟测试：`config/datasets/abide_smoke.conf`

两个被试（`config/datasets/abide_smoke_subjects.txt`）：`sub-0028744`（ABIDEII-GU_1，43 层，STC 已验证并应用）和 `sub-0029150`（ABIDEII-NYU_2，34 层，slice order 未解决 → STC 跳过；z 向 FOV 只有 102 mm）。`ANAT_MODE=synth`（SynthStrip + SynthSeg，不跑 recon-all）、`NORM_QUALITY=quick`、`SURFACE=no`、`MNI_RES=2`、`N_JOBS=1`、`NTHREADS=4`、`MIN_MEM_GB=6`、一个 Schaefer-100 图谱 + 自定义 `Yeo100`、策略 `wmcsf24 wmcsf24gsr`、`SMOOTH_FWHM=0`。

```powershell
# 全流程（ingest → … → validate → qc → validate --group → group_qc）
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_smoke `
    -Config config/datasets/abide_smoke.conf -VolumePrefix smoke

# 分步：先 ingest 并查看会处理哪些被试
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_smoke -Config config/datasets/abide_smoke.conf -VolumePrefix smoke --stages ingest
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_smoke -Config config/datasets/abide_smoke.conf -VolumePrefix smoke --list-subjects

# 只跑一个被试的解剖阶段；--dry-run 只打印命令
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_smoke -Config config/datasets/abide_smoke.conf -VolumePrefix smoke `
    -Subjects "sub-0028744" -Stages "anat_prep" --dry-run
```

结果：`E:\ASD\out_smoke\derivatives\sub-*.html`（每被试报告）、`derivatives\group\group_report.html`、`derivatives\group\validation_report.html`、`logs\status_<时间戳>.tsv`（每被试每阶段 ok/failed/skipped 与耗时）、`logs\pipeline_<时间戳>.log`。

**冒烟测试能验证的**：DPABI 布局 ingest 与 acquisition TSV 的 STC 决策（一例应用、一例跳过）；SynthStrip/SynthSeg 解剖路径与 quick SyN；synth 模式的 FLIRT-BBR 配准检查与回退；单次插值重采样到 T1w 与 2 mm 模板网格；confounds/FD/DVARS/aCompCor/censor；两种策略的 `3dTproject` 投影与 DOF 记账；Schaefer-100 与自定义 volume 图谱的 ROI 提取、覆盖率、FC；阶段 10 的 volume 流指标；每被试 HTML 报告；组级 `group_qc.tsv` 与报告；整套编排（状态表、失败隔离、hash 跳过）。

**冒烟测试不能验证的**：recon-all、`bbregister`/`mri_coreg`/`lta_convert` 路径与 aseg 组织掩膜；surface 分支（阶段 06、fsLR 图谱、`_desc-streamcompare.tsv`）；`NORM_QUALITY=precise` 的配准质量；`acompcor`/`36p` 等其它策略；平滑副本（`SMOOTH_FWHM=0`）；QC-FC（需要 ≥ `QCFC_MIN_SUBJECTS`=10 个 run）；组级 stream 比较的 Wilcoxon 检验（需要 ≥ 6 对）与 `fc_typicality`（需要 ≥ 4 个 run）；站点内离群检测（每站点只有 1 个被试）；`N_JOBS>1` 下的内存行为。

### 4.3 过夜 FreeSurfer 版：`config/datasets/abide_local_fs.conf`

同样两个被试，`ANAT_MODE=freesurfer`、`SURFACE=yes`、`NORM_QUALITY=precise`（默认）、`NTHREADS=4`。`recon-all -all -parallel -openmp 4` 每被试约 6–12 h，`N_JOBS=1` 串行。

```powershell
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_fs `
    -Config config/datasets/abide_local_fs.conf -VolumePrefix fs -Cpus 4

# 完成后把 FreeSurfer 结果从命名卷复制到 E:\ASD\out_fs\freesurfer_export（命名卷在资源管理器里不可见）
.\docker\run_docker.ps1 -Out E:\ASD\out_fs -VolumePrefix fs -ExportFreesurfer
```

在冒烟测试之外额外验证：recon-all、`bbregister` 检查/回退、aseg 组织掩膜、surface 分支全部产物、每 run 的 volume-vs-surface 比较表。仍不能验证：组级 Wilcoxon/typicality（样本太少）、QC-FC。中断后重跑同一命令即可续跑（recon-all 从 `scripts/IsRunning*` 判断是否有另一进程；确认没有后删除该文件）。

### 4.4 完整测试集：`config/datasets/abide_test.conf`

14 个被试、freesurfer + surface、`NTHREADS=8`、三种策略、两个 Schaefer 图谱 + Yeo100。这是给集群/大内存机器的配置，本机不建议。

### 4.5 常用参数

`run_pipeline.sh` 的选项（可跟在启动脚本后面直接传）：

```
-s, --subjects "A B"     被试（可省略 sub- 前缀，空格或逗号分隔，可重复）
    --subjects-file F    文件，一行一个 id（容器内路径）
    --stages "a b ..."   要跑的阶段（默认 conf 的 STAGES）
-j, --jobs N             并行被试数（覆盖 N_JOBS）
    --force              选中的阶段即使 hash 未变也重跑
    --dry-run            只记录命令不执行
    --list-subjects      打印将处理的被试后退出（需要先 ingest）
```

阶段名与顺序（无论 `--stages` 里怎么写，都按此顺序执行）：数据集级 `ingest fetch` → 每被试 `anat_recon anat_prep func_prep confounds denoise surface timeseries validate qc` → 数据集级 `validate --group`（选了 `validate` 时）、`group_qc`（选了 `group_qc` 或 `qc` 时）。`fetch` 在 surface/timeseries/validate 需要的资源缺失时自动执行（先 `fetch_resources.sh --check`）。

任何 conf 变量都可用环境变量覆盖：`-Env 'SURFACE=no','NTHREADS=8'`（PowerShell 会话内）或 `-Env SURFACE=no,NTHREADS=8`（`pwsh -File` 调用时）。启动脚本自己的参数：`-Data -Out -Config -License -Image -VolumePrefix -Cpus -Memory -Env -MinMemoryGB -Subjects(-s) -Stages -Shell -ExportFreesurfer -PrintOnly`。`--jobs`、`--force`、`--dry-run`、`--list-subjects`、`--subjects-file` 直接透传；单短横线的 `-c/-j/-h` 在 PowerShell 会话里要放在裸 `--` 之后（`pwsh -File` 不识别 `--`，改用双短横线形式）。

```powershell
# 交互式 shell（同样的挂载；容器内 FMRIPROC_CONFIG 已设置）
.\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out_smoke -Config config/datasets/abide_smoke.conf -VolumePrefix smoke -Shell
#   容器内：bash /opt/fmriproc/run_pipeline.sh -c $FMRIPROC_CONFIG --list-subjects
#           bash /opt/fmriproc/stages/fetch_resources.sh --check
#           bash /opt/fmriproc/stages/03_func_prep.sh sub-0028744
```

### 4.6 Git Bash / Linux / macOS

`docker/run_docker.sh` 与 `.ps1` 行为一致（`--data --out --config --license --image --volume-prefix --cpus --memory --min-memory-gb --env KEY=VALUE --shell --export-freesurfer --print`，其余参数透传）。脚本内部已设置 `MSYS_NO_PATHCONV=1`，Git Bash 下 `--shell` 会自动使用 `winpty`。

## 5. 描述一个新数据集

### 5.1 数据集 conf

`config/datasets/<name>.conf`，bash `: "${KEY:=value}"` 形式，只需写与 `config/default.conf` 不同的项。优先级：环境变量 > 数据集 conf > `default.conf`。`default.conf` 每个变量都有注释，是参数的权威列表。至少要给：

```bash
: "${INPUT_DIR:=/data}"           # 容器内路径
: "${OUT_DIR:=/out}"
: "${INPUT_LAYOUT:=dpabi}"        # dpabi | bids
: "${ACQ_TABLE:=$REPO_DIR/config/datasets/<name>_acquisition.tsv}"
: "${ANAT_MODE:=freesurfer}"      # freesurfer | synth
: "${SURFACE:=no}"                # yes 需要 ANAT_MODE=freesurfer 且 MNI_RES=2
```

`SUBJECT_LIST` 指向一个被试列表文件（一行一个 id，`#` 注释）时，ingest 只导入这些被试，`run_pipeline.sh` 也只处理它们；`-s` 只能在其中再取子集。

### 5.2 Acquisition TSV（`ACQ_TABLE`）

制表符分隔，列：`group tr n_slices slice_order stc drop_volumes pe_dir evidence note`。

- `group`：站点名，与 manifest 的 `group` 匹配；`*` 为默认行。
- `slice_order`：`SA SD IA IA2 ID ID2`（DPABI 编码，`IA` = 1,3,5…2,4…；`IA2` = 2,4…1,3…；`ID`/`ID2` 为对应降序）、`file:<path>`（每层一个秒数）或 `unknown`。
- `stc`：`apply` | `skip`；`unknown` 强制 `skip`。
- 表中 `tr`/`n_slices` 与 NIfTI 头不一致 → 该 run **跳过 STC 并在 `ingest_report.tsv` 里警告**，绝不“修正”头文件（ABIDEII-USM_1 的 41 层被试就是这样被自动跳过的）。
- `drop_volumes` 覆盖 `DROP_VOLUMES`；`pe_dir` 目前只写入 sidecar。

**STC 证据分级**（`evidence` 列，自由文本，建议前缀 A/B/C；原样写入 sidecar `SliceTimingEvidence` 并出现在 QC 的 `stc_evidence`）：

| 级别 | 含义 | 例子 |
|---|---|---|
| A | 直接、无歧义的来源 | DPABI 官方被试表；厂商产品序列 + 奇数层（interleave 无 parity 歧义） |
| B | 有文档但间接/需推断 | 发表的站点协议；Siemens 偶数层从第 2 层开始的规则 |
| C | 未解决 → `stc=skip` | 自定义序列 + 偶数层，parity 未确认（ABIDEII-NYU_2） |

写表的规则：宁可跳过也不猜。EMC/UCD/ABIDEII-USM 中明确依赖厂商惯例推断的行默认跳过 STC，保留候选层序与证据；并非所有 B 级都跳过。表中数值合法不等于协议已直接验证。对未解决的站点可将候选层序作为另行设计的敏感性分析（见 `docs/VALIDATION.md`）。

### 5.3 DPABI 与 BIDS 布局

- **dpabi**：`<INPUT_DIR>/[<site>/]FunImg/<sub>/<一个>.nii[.gz]` 与 `T1Img/<sub>/<一个>.nii[.gz]`；每个被试文件夹**恰好一个** NIfTI；站点文件夹名即 `group`；被试文件夹名规范化为 `sub-<id>`；任务名为 `TASK_NAME`（默认 `rest`）。ingest 会写一份头文件规范化的副本（qform/sform code 置 1、sform 缺失时取 qform、TR 存为秒；体素数据不动）。
- **bids**：`sub-*/[ses-*/]func/*task-rest*_bold.nii*`（`BIDS_FUNC_GLOB`）与 `anat/*_T1w.nii*`；已有 sidecar 优先（同名 JSON，其次数据集根目录的 `task-<task>_bold.json`），acquisition 表只补空缺；`group` 取自 `participants.tsv` 的 `acq_group`/`site`/`site_id` 列。

两种布局都生成 `rawdata/manifest.tsv`（`subject session task run group bold t1w run_label`）、`participants.tsv` 和 `ingest_report.tsv`（每 run 的头文件检查、STC 决策与原因、警告）。少于 30 个 volume 的 run 被拒绝。

## 6. 阶段概览

每个阶段一段话；命令级细节、参数取舍和典型故障见 `docs/STEPS_zh.md`。

**00 ingest**（数据集级，Python）— 发现 run、校验头文件（4D、TR、单位、qform/sform、倾斜、dtype、FOV）、写规范化副本与 sidecar（`RepetitionTime`、`SliceTiming`（仅应用时）、`SliceTimingSource/Evidence`、`DropVolumes`、`AcquisitionGroup`）、写 manifest/participants/ingest_report。无 hash 标记：幂等且便宜，表格每次重写，图像副本只在 `--overwrite`/`FORCE=yes` 时重写。

**fetch**（数据集级）— 一次性下载 fsLR-32k 网格、HCP `Atlas_ROIs.2`、Schaefer volume/dlabel 与标签到 `$RESOURCE_DIR`（`templateflow/`、`hcp/`、`atlases/<A>/`）。`--check` 只报告缺失（退出 0 = 齐全）。

**01 anat_recon** — `ANAT_MODE=freesurfer`：`recon-all -all -parallel -openmp $NTHREADS`，完成的判据是 `scripts/recon-all.done` + `surf/{lh,rh}.pial` + `mri/aseg.mgz`；存在 `scripts/IsRunning*` 时拒绝触碰；先探测 `FS_DIR` 能否建符号链接。`synth` 模式无事可做。

**02 anat_prep** — T1w 参考（`nu.mgz` 或 N4 后的原图）、脑掩膜（`brainmask.mgz` 或 SynthStrip）、分割（aseg 或 `mri_synthseg`，默认 `SYNTHSEG_FLAGS=--robust`；输入先裁剪到 SynthStrip 脑框 + 15 mm，内存约减半）、在 1 mm 网格上腐蚀得到 WM/CSF/GM 掩膜、`antsRegistrationSyN[Quick].sh` brain-to-brain 到 1 mm 模板（固定 `ANTS_SEED`）、解剖 QC JSON（Euler 数、Dice、Jacobian 分位数、模板相关）。

**03 func_prep**（每 run）— float32 副本、丢弃前 `DropVolumes` 个 volume、非稳态检测；原始 QC（`3dToutcount`、`3dTqual`）；`3dDespike`；两遍 `mcflirt`（只保留矩阵/参数）；`3dTshift`（仅有 `SliceTiming` 时，`STC=auto|require|off`）；boldref N4 + EPI 掩膜；配准（`mri_coreg`→`bbregister` 或 FLIRT→FLIRT-BBR，位移/cost 超限即回退并记 `bbr_rejected`）；`antsApplyTransforms` 单次重采样到 T1w 网格（`FUNC_T1W_RES`）和模板网格（`MNI_RES`）；掩膜 = T1 脑掩膜 ∩ 膨胀的 EPI 支持 ∩ 时间最小值 > 0；一个全局缩放因子（T1w 空间脑内中位数 → `SCALE_TARGET`）；`desc-prep_info.json` 记录一切。

**04 confounds**（Python，用 T1w 空间 BOLD）— 24 个运动参数、WM/CSF/全局均值及其导数/平方、aCompCor（各 `ACOMPCOR_N` 个，PCA 前 DCT 高通 `HIGHPASS_SEC`）、cosines、FD（Power）与 Jenkinson RMS、DVARS、离群比例、censor 向量（`CENSOR_FD`、`CENSOR_PREV`、`CENSOR_DVARS`）。

**05 denoise** — 每个策略：选列 → 中心化/单位化的 `regressors.1D` → 数据去均值 → `3dTproject -ort -polort -dt [-passband] [-censor -cenmode]`，作用于模板空间 BOLD（`SURFACE=yes` 时也作用于 T1w 空间 BOLD 供阶段 06）。`_denoise.json` 记录 DOF；`dof_remaining < MIN_DOF` 只标记不致命。可选 `3dBlurInMask` 平滑副本。

**06 surface**（可选）— FreeSurfer 表面 → GIFTI（`--to-scanner`）、midthickness、皮层 ROI、HCP goodvoxels、ribbon-constrained 映射（`-voxel-subdiv` 5，层厚 ≥ 3.5 mm 时 7）、膨胀、掩膜、`ADAP_BARY_AREA` 重采样到 fsLR-32k，皮层下来自 2 mm 模板空间序列 + `Atlas_ROIs.2`，每策略一个 dtseries，可选 `-cifti-smoothing`。

**07 timeseries** — volume：图谱重采样到 BOLD 网格（最近邻），对覆盖、有限、非常数体素取均值；覆盖率 < `MIN_ROI_COVERAGE` 的 ROI 为 `NaN`；写带标签表头的 TSV、覆盖率表、保留帧上的 Pearson FC；另存去噪前的 ROI 均值（用于 ROI tSNR）。CIFTI：`wb_command -cifti-parcellate`。

**10 validate**（每被试，在 qc 之前；`--group` 数据集级）— 只读阶段 07 的 ROI 表、censor 向量、confounds 与 `_denoise.json`，计算 §8 列出的时间序列有效性指标，两条流都存在时逐 parcel 比较；组级汇总、Wilcoxon、`fc_typicality`、`stream_comparison.png`、`validation_report.html`。

**08 qc** — 每 run `desc-qc_metrics.json`、`roiqc.tsv` 与图；每被试自包含 HTML 报告（总是重新生成，以纳入阶段 10 的表）。缺失的输入只变成 `n/a`，QC 不修复任何东西。

**09 group_qc**（数据集级）— `group_qc.tsv`（每 run 一行，站点内 robust z 离群标记）、`qcfc_<S>_<A>.tsv` 与 `qcfc_summary.tsv`（≥ `QCFC_MIN_SUBJECTS` 个 run 时）、`group_report.html`。

## 7. 输出目录

```
$OUT_DIR/
  rawdata/manifest.tsv participants.tsv ingest_report.tsv
  rawdata/sub-X[/ses-Y]/anat/sub-X_T1w.nii.gz  func/<RUN>_bold.nii.gz + .json
  freesurfer/sub-X/                         SUBJECTS_DIR（命名卷）
  work/sub-X/.done/<stage>[__<RUN>].hash    阶段标记
  work/sub-X/{anat,func/<RUN>}/             中间文件（可删）
  derivatives/sub-X/anat/
    sub-X_desc-preproc_T1w.nii.gz  _desc-brain_mask  _desc-aseg_dseg  _label-{WM,CSF,GM,WMbbr}_mask
    xfm/T1w_to_MNI_{0GenericAffine.mat,1Warp,1InverseWarp}
    sub-X_space-<TPL>_desc-preproc_T1w  _space-<TPL>_desc-brain_mask  sub-X_desc-anatqc.json
    sub-X_hemi-{L,R}_{white,pial,midthickness}.surf.gii  ..._space-fsLR_den-32k_midthickness.surf.gii   (surface)
  derivatives/sub-X/func/
    <RUN>_desc-prep_info.json  _desc-hmc_motion.par  _desc-hmc_{relrms,absrms}.txt  _desc-outliers_timeseries.1D
    <RUN>_from-bold_to-T1w_itk.txt  _space-T1w_{boldref,desc-brain_mask,desc-preproc_bold,label-*_mask}
    <RUN>_space-<TPL>_res-<R>_{boldref,desc-brain_mask,desc-preproc_bold}
    <RUN>_desc-confounds_timeseries.tsv + .json  _desc-censor.1D
    <RUN>_desc-<S>_regressors.1D + _denoise.json  _space-<TPL>_res-<R>_desc-<S>_bold  [_desc-<S>sm<F>_bold]
    <RUN>_space-fsLR_den-91k_desc-{preproc,<S>}_bold.dtseries.nii  _desc-preproc_tsnr.dscalar.nii  _desc-sampled_mask.dscalar.nii  _desc-surfqc.json   (surface)
    <RUN>_space-<TPL>_atlas-<A>_desc-<S>_{timeseries,coverage,connectivity}.tsv + .json  _desc-preproc_timeseries.tsv
    <RUN>_space-fsLR_atlas-<A>_desc-{<S>,preproc}_...   (surface)
    <RUN>_desc-validation.{tsv,json}  [_desc-streamcompare.tsv]  _desc-qc_metrics.json  _atlas-<A>_desc-<S>_roiqc.tsv
  derivatives/sub-X/figures/                报告用 PNG
  derivatives/sub-X.html                    每被试 QC 报告
  derivatives/group/  group_qc.tsv inclusion.tsv group_report.html qcfc_*.tsv qcfc_summary.tsv
                      validation_long.tsv stream_comparison.tsv strategy_comparison.tsv fc_typicality.tsv
                      stream_comparison.png validation_report.html
  logs/sub-X/<stage>.log  logs/pipeline_<ts>.log  logs/status_<ts>.tsv  logs/tool_versions.json
```

`<RUN>` = `sub-X[_ses-Y]_task-<task>[_run-<N>]`；`<TPL>` = `TEMPLATE_NAME`（默认 `MNI152NLin6Asym`）；`<R>` = `MNI_RES`；`<S>` = 策略；`<A>` = 图谱。

## 8. QC 指标词汇表

### 8.1 每 run `desc-qc_metrics.json`（`docs/DESIGN.md` §9）

| 组 | 键 | 含义 |
|---|---|---|
| acquisition | `tr n_volumes_raw n_dropped n_volumes minutes voxel_size stc_applied stc_evidence nss_detected despike_fraction` | `nss_detected > n_dropped` 说明丢弃不够 |
| motion | `fd_mean fd_median fd_max fd_pct_gt_02 fd_pct_gt_05 relrms_mean absrms_max n_censored pct_censored minutes_retained longest_segment` | FD 为 Power 定义（50 mm 半径，弧度→mm） |
| signal | `tsnr_gm_median tsnr_wm_median tsnr_brain_median`（去噪前）、`dvars_std_mean outlier_frac_mean quality_index_mean gcor fd_dvars_corr gs_fd_corr fwhm_acf` | `fd_dvars_corr` 高 = 运动直接进入信号 |
| registration | `coreg_method bbr_cost bbr_vs_init_mm coreg_dice norm_dice jacobian_p01/p99 jacobian_nonpos_frac template_corr` | `coreg_dice` = EPI 支持 vs T1 脑掩膜（BOLD 网格） |
| masks | `brain_mask_voxels wm_mask_voxels csf_mask_voxels dropout_fraction` | 掩膜过小会被放宽并记 `tissue_erosion_relaxed` |
| 每策略 `S.` | `n_regressors dof_remaining tsnr_gm_median_post tsnr_gain variance_removed_gm fd_dvars_corr_post` | `tsnr_gm_median_post` = 去噪前均值 / 残差 SD |
| 每策略每图谱 `S.A.` | `roi_tsnr_median roi_tsnr_min n_roi_nan coverage_min split_half_r network_contrast fc_mean fc_sd` | |
| surface | `tsnr_cortex_median pct_badvertices pct_goodvoxels_excluded` | |

**时间序列 SNR 的定义**（最终数据零均值，所以不能用最终序列的均值/SD）：

- `roi_tsnr` = **去噪前、已缩放** BOLD 的 ROI 均值随时间的平均 ÷ **去噪后** ROI 序列在保留帧上的 SD。
- `variance_removed` = 1 − var(去噪后) / var(去噪前、polort 去趋势后)。
- `split_half_r` = 保留帧前后两半各算 Fisher-z FC，上三角之间的 Pearson r。

**标志阈值**（`default.conf` 的 `QC_*`，报告里的红黄绿）：

| 指标 | 警告 | 失败 | 方向 |
|---|---|---|---|
| `fd_mean` (mm) | > 0.2 | > 0.5 | 越低越好 |
| `pct_censored` (%) | > 20 | > 50 | 越低越好 |
| `tsnr_gm_median` | < 40 | < 20 | 越高越好（3 mm、TR 2 s 的传统数据多在 30–60） |
| `coreg_dice` | < 0.90 | < 0.80 | 越高越好 |
| `norm_dice` | < 0.93 | < 0.88 | 越高越好 |
| `holes_total`（Euler） | > 100 | > 200 | 越低越好；儿童/运动伪影的 T1 常见偏高 |
| `dof_remaining` | < `MIN_DOF`=15 | — | 见 §9 |
| `minutes_retained` | < `MIN_RETAINED_MIN`=4 | — | censoring 后有效时长 |

经验上：`bbr_rejected=true` 或 `coreg_method` 回退到 `mri_coreg`/`flirt` 的 run 必须目视检查报告里的 EPI→T1 白质轮廓；`jacobian_nonpos_frac > 0` 意味着 SyN 折叠，配准不可用；`tissue_erosion_relaxed=true` 时 WM/CSF 回归量可能混入 GM 信号。

### 8.2 阶段 10 时间序列有效性指标（`docs/DESIGN.md` §12；每 run × stream × 策略 × 图谱）

| 指标 | 定义 | 方向 |
|---|---|---|
| `roi_tsnr_median`, `roi_tsnr_p10` | mean(去噪前 ROI 序列) / SD(去噪后 ROI 序列，保留帧) 的中位数 / 第 10 百分位 | 高好，但过度去噪/低通也会抬高，要和下面几项一起看 |
| `variance_removed_median` | 同 8.1，ROI 中位数 | 描述性 |
| `split_half_r` | 保留帧前后两半 FC 的可靠性 | 高好 |
| `network_contrast` | (网络内平均 z − 网络间平均 z) / SD(网络间)；需要 `network` 标签 | 高好 |
| `homotopic_contrast` | 同伦 parcel 对的平均 z − 非同伦跨半球 z；配对按同网络、镜像质心最近（≤ 20 mm） | 高好 |
| `dmn_contrast` | Default 网络中含 `PCC`/`pCunPCC` 与含 `PFC` 的 parcel 之间的平均 z − 它们与 `SomMot` 的平均 z | 高好 |
| `lowfreq_power_fraction` | 去噪前、去趋势 ROI 序列中 0.01–0.1 Hz 功率 / 0.01 Hz–Nyquist 功率（ROI 中位数） | 描述性 |
| `fd_fc_coupling` | 保留帧上 FD 与逐帧共波动幅度（z 化 ROI 乘积/边时间序列的 RSS）的 \|Spearman\| | 低好（残余运动耦合） |
| `gs_residual_sd` | 最终序列（z 化 ROI）跨 ROI 均值的 SD | 描述性（GSR 后应接近 0） |
| `n_roi_nan`, `n_retained`, `dof_remaining` | 覆盖不足的 ROI 数、保留帧数、剩余 DOF | `n_roi_nan` 低好 |
| `fc_similarity`（stream 比较） | 两条流 Fisher-z 上三角的 Pearson r | 描述性 |

组级 `stream_comparison.tsv` 先对同一被试的配对 run 求均值，再以被试为单位报告两流中位数、配对差、Wilcoxon p（≥ 6 人）及每策略 × 图谱指标 family 的 BH q。`significant` 使用 q，`better` 仅表示差异方向；这些仍是探索性比较。`fc_typicality.tsv` 的参考排除同一人的所有 run，并对其他人等权。QC-FC 同样按被试聚合并报告边 family 的 BH q；跨站点混杂仍需另外解释。

## 9. 去噪策略与自由度预算

| 策略 | 回归量 | 数量 | 说明 |
|---|---|---|---|
| `wmcsf24` | 24 运动参数 + WM 均值 + CSF 均值 | 26 | 默认；保守 |
| `wmcsf24gsr` | 同上 + 全局信号 | 27 | 默认之二；去掉全局伪影，代价是引入负相关、改变组间差异解释 |
| `36p` | 24 运动 + WM/CSF/GS 及各自导数、平方 | 36 | Satterthwaite 36P；短 run 上 DOF 紧张 |
| `acompcor` | 12 运动 + 5 WM + 5 CSF aCompCor | 22 | 不依赖 GS，但 aCompCor 在 3–4 mm 体素、腐蚀后的小 CSF 掩膜上成分不稳定 |
| `acompcorgsr` | 同上 + GS | 23 | |
| `legacy8` | 6 运动 + WM + CSF | 8 | 与 v1 `Retain_GRS` 对应 |
| `legacy9gsr` | 6 运动 + WM + CSF + GS | 9 | 与 v1 `NoGRS` 对应 |

**DOF 记账**（`_denoise.json`）：

```
dof_remaining    = N_retained − rank(joint design on the retained rows)   # 与 MIN_DOF 比较，删帧计入
algebraic_dof    = fit_rows − rank(joint design)
fit_rows = N                    # NTRP: 先插值，再拟合全时间轴
fit_rows = N_retained            # ZERO/KILL: 只拟合保留行，此时两者相等
afni_nominal_dof = N_retained − design_columns
```

设计包含 polynomial、AFNI Fourier stopband 和 nuisance 的联合列空间；aCompCor 保留 PCA 预滤波使用的全部 DCT，不能按频率假设它已被 Fourier 覆盖。秩阈值按 float32 设计精度记录。这里是代数残余维度，**不是有效样本量，也不是 AFNI 正则化平滑矩阵的有效自由度**。NTRP 插值不能恢复独立观测信息。AFNI 另要求至少 9 个保留观测且 nominal columns 少于保留观测；不满足会提前报错，不能用较低的联合秩绕过。
v2.2 起 `dof_remaining` 只数保留帧：NTRP 的插值行参与拟合，但不提供观测信息，v2.1 把它们也算进去，
每删一帧就多算一个自由度（例：GU_1 报告 24，实际 21）。`algebraic_dof` 仍记录拟合行的代数维度。

以下仅是删帧前的 nominal 列数预算近似，用于说明短扫描限制，不是新 `dof_remaining` 的实测结果：

| N（丢弃后） | 带通成本 | `wmcsf24`(26) 剩余 | `36p`(36) 剩余 | 说明 |
|---|---|---|---|---|
| 146（TRINITY 150−4） | ≈ 93 | ≈ 24 − censored | ≈ 14 − censored | `36p` 已低于 `MIN_DOF`=15 |
| 176（NYU_2 180−4） | ≈ 113 | ≈ 34 | ≈ 24 | |
| 296（300−4） | ≈ 189 | ≈ 78 | ≈ 68 | |

默认仍为 `wmcsf24`/`wmcsf24gsr`、`CENSOR_FD=0.5`、`CENSOR_MODE=NTRP`。方法选择应基于扫描时长、噪声和目标 estimand，不能为通过 DOF 检查而自动切换滤波或删帧阈值。`dof_remaining < MIN_DOF` 标记为低 DOF；AFNI nominal gate 不满足则停止该策略。NTRP 输出保留时间轴，FC/QC 仍按显式 censor 向量使用原始保留观测；KILL 输出变短，按原始时间索引对齐。

### 9.1 删帧与被试纳入

FD 按时间点计算，删帧删的是时间点（stage 04 的 censor 向量）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `CENSOR_FD` | 0.5 | FD（Power）超过此值的时间点被删除 |
| `CENSOR_PREV` | no | 同时删除前一帧 |
| `CENSOR_NEXT` | 0 | 同时删除之后 N 帧（Power 2014 用 2） |
| `CENSOR_MIN_SEGMENT` | 0 | 连续保留段短于 N 帧的也删除（Power 2014 用 5），最后执行 |
| `CENSOR_DVARS` | 0 | 标准化 DVARS 阈值 |

短扫描加带通时，每删一帧就少一个自由度。在 2026-09-24 测试的两名 ABIDE 被试上（150/176 帧，TR 2 s，wmcsf24）：

| 规则 | GU_1 删帧 / 实际自由度（带通 / 高通） | NYU_2 删帧 / 实际自由度（带通 / 高通） |
|---|---|---|
| FD > 0.5 mm（默认） | 2% / 21 / 112 | 0.6% / 33 / 140 |
| FD > 0.3 mm | 8% / 12 / 103 | 4% / 27 / 134 |
| FD > 0.2 mm | 25% / −13 / 78 | 18% / 3 / 110 |
| Power 2014 全套（0.2 mm，前 1 后 2，短段 < 5） | 63% / −71 / 20 | 49% / −53 / 54 |

所以带通为默认时保持 FD > 0.5 mm；更严格的删帧要配合高通（`FILTER_MODE=highpass`），并作为敏感性分析。
滤波方式对整个数据集只能选一种，不能按被试切换。

被试（run）纳入由 stage 09 与 10 `--group` 判断（`fmriproc/inclusion.py`）。不删除任何数据：决定和原因写在
`derivatives/group/inclusion.tsv`，组水平统计（QC-FC、stream 与策略比较）只用纳入的 run，`validation_long.tsv` 保留所有 run 并标 `included`。

| 变量 | 默认 | 剔除条件（0 = 关闭） |
|---|---|---|
| `EXCLUDE_FD_MEAN` | 0.5 | 平均 FD > 0.5 mm（Parkes 2018：宽松 0.55，严格 0.25） |
| `EXCLUDE_FD_MAX` | 5 | 任一时间点 FD > 5 mm |
| `EXCLUDE_PCT_FD_GT02` | 0 | FD > 0.2 mm 的时间点比例（Parkes 2018 严格：20%） |
| `EXCLUDE_MIN_RETAINED_MIN` | 4 | 删帧后保留不足 4 分钟 |
| `EXCLUDE_MIN_DOF` | 15 | 实际自由度不足，按策略分别判断 |
| `EXCLUDE_QC_FAIL` | yes | run 级 QC flag 为 fail 或 incomplete（配准、标准化、tSNR、Euler holes、删帧比例） |

- **标准要在看结果之前写定**，所有组用同一套；更严格的标准只做敏感性分析。
- **严格标准对 TR 2 s 数据很苛刻。** GU_1 平均 FD 只有 0.17 mm，却有 25% 的时间点超过 0.2 mm，会被"超过 20%"的标准剔除。
- **临床样本的头动常与诊断和年龄相关。** 剔除越严，剩下的样本越偏向头动小、症状轻的被试。设置 `PHENOTYPE_TSV` 后组报告会按组比较剔除人数和剩余被试的平均 FD（Mann-Whitney、Fisher 精确检验）。ABIDE 表型文件：`PHENOTYPE_ID_COLUMN=SUB_ID`、`PHENOTYPE_GROUP_COLUMN=DX_GROUP`、`PHENOTYPE_LABELS="1=ASD 2=TD"`。测试集自带的 `dpabi_parameters_by_subject.csv`（`SUB_ID`，`group` = ASD/TDC）在数据目录中存在时，三个 ABIDE conf 会自动使用它。组分析中把平均 FD 作为协变量。
- **DPABI 相关文献多用 Jenkinson FD。** 它的数值比这里的 Power FD 小，两种阈值不能混用。
- **TR 3 s 的数据帧间隔更长，FD 通常更大。** 可以在该数据集的 conf 里单独设阈值。

## 10. Surface 分支注意事项

- 前提：`ANAT_MODE=freesurfer`、`MNI_RES=2`（皮层下取自 2 mm 模板空间序列 + HCP `Atlas_ROIs.2`）、`fetch_resources` 已下载 fsLR 网格与 dlabel。`fp_init` 会拒绝不满足的组合。
- 表面用 `mris_convert --to-scanner` 导出，与阶段 03 的世界坐标 BOLD 直接对齐；不要对球面用 `--to-scanner`。
- 3–4 mm 体素下 ribbon-constrained 映射的部分容积严重：`-voxel-subdiv` 5（层厚 ≥ 3.5 mm 时 7），goodvoxels 排除高变异体素（`pct_goodvoxels_excluded`），坏顶点比例 `pct_badvertices` 与 `tsnr_cortex_median` 在 `desc-surfqc.json` 里。眶额、颞极因无 SDC 而失真，surface 采样比 volume ROI 平均更敏感。
- 皮层 parcel 序列来自 `-cifti-parcellate`；阶段 10 按唯一 ROI 名称显式对齐后比较共同有效 ROI。名称集合不一致或重复时不能靠列数猜对应关系，报告不可比较。自定义 volume 图谱没有 surface 版。
- surface 平滑用 `SURF_SMOOTH_FWHM`（`-cifti-smoothing`），与 volume 的 `SMOOTH_FWHM` 独立；ROI 序列仍来自未平滑的 dtseries。
- 不要只凭 surface 的 tSNR 判断优劣：它随过度去噪上升。看 `stream_comparison.tsv` 里可靠性、网络对比、`fd_fc_coupling` 的配对差和方向。

## 11. 重跑、FORCE 与阶段 hash

- 每个阶段完成后写 `work/sub-X/.done/<stage>[__<RUN>].hash`；包含 pipeline 版本、阶段脚本、共享 Bash 和该阶段 Python 依赖、配置与上游标记。Python 计算代码改变也会使对应缓存失效。只哈希小型代码文件，不全盘哈希影像。
- ingest 使用源/目标路径、大小、纳秒修改时间与预期头的复制记录，不再仅凭相同 NIfTI 头复用。旧派生副本没有记录时重复制；这用于普通文件替换检测，不检测刻意保持大小/时间的内容篡改。生成的 `participants.tsv` 用 `acq_group` 表示站点；DPABI 同 stem JSON 可提供运行级时序。
- 改了某个配置值 → 声明了该变量的阶段及其下游自动重跑；改了阶段脚本 → 该阶段重跑；上游重跑 → 下游失效。`ingest`（幂等）、`08_qc` 的被试报告、`09/10 --group` 没有标记，总是执行。
- `--force`（`FORCE=yes`）让**选中的**阶段无视标记重跑，配合 `--stages` 限定范围；`SKIP_EXISTING=no` 等价于全部强制。
- 彻底重来：删除 `work/sub-X`（命名卷内：`-Shell` 后 `rm -rf /out/work/sub-X`）或 `docker volume rm <prefix>_work`。`derivatives/` 的最终文件在阶段成功时才由临时名 `mv` 到位，中断不会留下似是而非的结果。
- `FORCE=yes` 对 `00_ingest` 意味着重写图像副本；对 `01_anat_recon` 不会删除已有的 recon-all 目录（存在 `IsRunning*` 时拒绝）。

**代码冻结**：每次运行开始时编排器把 `run_pipeline.sh`、`lib/`、`stages/`、`py/`、`config/`、`parcellations/` 复制到 `logs/code_<run_id>/`（约 5 MB，含 `code_manifest.md5` 与 `git_revision.txt`，数据集 conf 也一并复制），所有阶段都从这份副本执行。bash 是边读边执行脚本的，运行期间修改仓库里的脚本会让正在执行的阶段读到错位的内容而失败；冻结后修改只影响下一次运行，同时这份副本记录了产出结果的确切代码。`FREEZE_CODE=no` 关闭；旧的 `logs/code_*` 可以随时删除。

## 12. 测试

```bash
# Git Bash，仓库根目录；主机 python 3.11 即可
bash tests/run_tests.sh            # bash -n、shellcheck（若安装）、CRLF 检查、tests/test_common_sh.sh、python unittest
bash tests/test_common_sh.sh       # 只跑 lib/common.sh 与 run_pipeline.sh 的 bash 单元检查（假阶段脚本）
PYTHONPATH=py python -m unittest tests.test_validate -v
```

容器内：`bash -lc 'cp -r /opt/fmriproc /tmp/repo && cd /tmp/repo && bash tests/run_tests.sh'`。测试全部用合成数据，不需要真实影像。测试入口还运行中心化复用及容器参数构造的工具替身检查；它们不代替真实 AFNI/Singularity 集成测试。

## 13. 故障排除

**Docker 内存不足 / 进程被杀（exit 137）**：`docker info --format '{{.MemTotal}}'` 看 VM 实际内存；Hyper-V 后端在 Docker Desktop → Settings → Resources 调整（本机 8–12 GB 可行，20 GB 会导致 VM 起不来）。流水线在 `/proc/meminfo` 低于 `MIN_MEM_GB` 时拒绝重阶段；`mri_synthseg --robust`、`antsRegistrationSyN.sh` 和 recon-all 是内存大户（synth 模式已先把 T1 裁剪到脑框；12 GB 以下仍不够时设 `SYNTHSEG_FLAGS=`，即不带 `--robust`），`N_JOBS` 应满足 `N_JOBS × MIN_MEM_GB ≤ VM 内存`。冒烟 conf 已把 `MIN_MEM_GB` 设为 6、`NTHREADS` 设为 4。

**NTFS bind mount 与符号链接**：`E:\` 目录挂进容器是 9p/drvfs 类文件系统，小文件 I/O 慢 5–20 倍且不能建符号链接，recon-all 会在数小时后失败。阶段 01 会先探测并 `die`；`run_pipeline.sh` 预检对 `work/`、`freesurfer/` 落在此类文件系统上发出警告。解决办法就是启动脚本的命名卷；需要看 FreeSurfer 结果时用 `-ExportFreesurfer`（`tar -h` 解引用链接、排除 `fsaverage`）。`docker volume ls`/`docker volume rm` 管理卷；每个数据集用不同 `-VolumePrefix`。

**模板/图谱下载**：`fetch_resources` 需要容器能访问 `templateflow.s3.amazonaws.com`、`raw.githubusercontent.com` 和 TemplateFlow API；失败时它只记录、不阻断解剖/功能阶段，但 surface/timeseries/validate 会在资源缺失时失败。手动重试：`--stages fetch`；诊断：`-Shell` 后 `bash /opt/fmriproc/stages/fetch_resources.sh --check`。镜像/代理：`fetch_resources.sh --github-raw URL --templateflow-s3 URL --no-templateflow-api --retries N --timeout S`（或环境变量 `FMRIPROC_GITHUB_RAW`、`FMRIPROC_TEMPLATEFLOW_S3`）。资源在 `<prefix>_resources` 卷里持久保存，只下载一次。

**Git Bash 路径改写**：MSYS 会把 `/out`、`/data` 这类参数改成 `C:/Program Files/Git/out`。`docker/run_docker.sh` 已设置 `MSYS_NO_PATHCONV=1` 与 `MSYS2_ARG_CONV_EXCL='*'`；手工敲 `docker run` 时要自己加 `MSYS_NO_PATHCONV=1`。`docker run -it` 在 mintty 里报 "the input device is not a TTY"：用 `winpty docker ...`（脚本的 `--shell` 已自动处理）或改用 PowerShell。

**`python3` 与 `PYTHON_BIN`**：容器里裸 `python3` 是 FSL 自带的解释器，没有 nilearn；所有 Python 都必须走 `$PYTHON_BIN`（默认 `/opt/micromamba/envs/neuro/bin/python`）。预检会导入 numpy/scipy/pandas/nibabel/nilearn/sklearn/matplotlib/jinja2/fmriproc 并列出缺失项。主机上跑测试时 `python3` 可能是 Microsoft Store 的占位程序，`tests/run_tests.sh` 会自动挑选能 `import numpy` 的解释器，也可 `PYTHON_BIN=... bash tests/run_tests.sh`。

**CRLF**：`\r: command not found` 或 `set: pipefail: invalid option` 说明脚本带 CRLF。`git config core.autocrlf false` 后重新检出（`.gitattributes` 已声明 `eol=lf`）；启动脚本和 `tests/run_tests.sh` 都会检查并警告。

**PowerShell 参数**：`pwsh -File` 调用时 `-Env` 用一个逗号分隔的字符串（`-Env SURFACE=no,NTHREADS=4`），且不认识裸 `--`；会话内调用（`.\docker\run_docker.ps1`）可用数组 `-Env 'a=1','b=2'` 和 `-- -c ...`。含双引号的参数无法可靠透传。

**预检失败**：`missing commands: ...` = 没有通过 `bash -lc` 进入登录 shell，或镜像不对；`FreeSurfer license not found` = `-License` 路径错；`only N GB RAM visible` = 见上。预检把所有问题一次列全再退出；`--dry-run` 时只警告。

**阶段被跳过（`skip ... up to date`）**：hash 未变。要重算就 `--force`，或改一个相关参数。`--stages` 里没写的阶段不会执行也不会检查其标记，因此下游可能用旧结果——重跑上游时把下游一起列上。

**一个被试失败**：其余被试继续，`logs/status_<ts>.tsv` 与结尾摘要列出失败的阶段和日志路径（`logs/sub-X/<stage>.log`，无 ANSI 码）；退出码 1。修好后只需重跑，成功的阶段会被跳过。

## 14. 局限

- 无 fieldmap/SDC（ABIDE/ADNI 没有），眶额与颞极的磁化率失真原样保留；`coreg_dice`、`dropout_fraction` 与报告里的 EPI→T1 叠加是唯一的提示。
- 每被试一个 T1w（manifest 里该被试的第一条），解剖产物不带 `ses` 实体；多 session 的纵向设计未处理。
- STC 只在有验证 timing 时执行；没有 timing 的站点只能跳过或做 IA/IA2 敏感性分析。多波段/多回波数据不在目标范围。
- 模板固定为 FSL 的 MNI152（`MNI152NLin6Asym`），`TEMPLATE_NAME` 只是文件名里的标签；自定义图谱必须已在该空间（只做网格适配，不做空间检测），无 `labels.tsv` 的图谱缺少网络类指标。
- surface 分支要求 freesurfer 模式与 `MNI_RES=2`；儿童/老年脑用成人模板与 recon-all 默认参数，`holes_total`、`norm_dice` 要单独审。
- DPABI 布局要求每个被试文件夹恰好一个 NIfTI；少于 30 个 volume 的 run 被拒绝；QC-FC 需要 ≥ 10 名被试，stream 比较的检验需要 ≥ 6 名有配对结果的被试。
- 没有 ICA-AROMA/tedana 之类的数据驱动去噪，没有 GPU 加速；单被试内部只靠 `NTHREADS`，短 run 的 DOF 预算见 §9。
- v2.2 只在两名 ABIDE 被试上做过端到端验证（§15.2）；多站点大样本、ADNI（TR 3 s）和组水平统计尚未验证。2026-09-22 Docker Desktop 无法启动是残留的 AF_UNIX socket 造成的，处理方法见 `docker/README.md`，不需要重置 Docker 数据。
- 原有未提交源代码的本地恢复副本：`archive/snapshots/2026-09-22_pipeline_upgrade/source.zip`（894782 字节）。未包含原始影像；本地副本可恢复，外部备份状态未验证。

## 15. 本地升级验证记录

### 15.1 2026-09-22：合成测试与导入

本轮升级验证以本地 `refactor/v2` 工作树为基线；本节记录本地测试状态，发布版本以 Git 提交为准。旧源恢复副本、运行日志和影像数据仅保留在本机，不随代码发布。原始影像只读，未连接 BSCC 或其他服务器，也未更改共享 Python 环境。

- `bash tests/run_tests.sh` 的 Bash 部分：144 项检查通过，另有去噪中心化数值/调用次数与 Singularity 启动参数检查通过。修正原测试入口的 unittest discovery 参数后，`bash tests/run_tests.sh --no-bash` 运行 357 个 Python 测试，8 个因平台条件跳过，其余通过。日志为 `logs/local_tests.log`，包含最初 discovery 错误及修复后的结果。
- 最后补齐 stage 10 缺失 censor 的处理后，单独重跑 `python -B -m unittest discover -s tests -p test_validate.py`：63 个测试，4 个跳过，其余通过；其余未改变的模块沿用前述成功结果。缺失 censor 不推断全保留，stage 08 标记 incomplete，stage 10 拒绝产生验证结果。
- 测试环境：Windows、本机 Python 3.11.15；NumPy 2.4.4、SciPy 1.17.1、pandas 3.0.2、nibabel 5.4.2、nilearn 0.13.1、scikit-learn 1.8.0。此结果不能替代 Linux 镜像内固定版本环境的测试。shellcheck 未安装，未执行；Bash 语法和 LF 检查通过。
- 实际数据只执行 stage 00：`abide_smoke_subjects.txt` 两人导入到 `E:/ASD/fmriproc_local_test/rawdata`，2 个有效 run、0 错误。GU 为 64×64×43×152、TR=2 s，记录 43 个 SliceTiming 和 DropVolumes=2；NYU2 为 64×80×34×180、TR=2 s，无 SliceTiming、DropVolumes=4，记录 102 mm 短 z-FOV。此处是元数据决策，尚未执行 STC 或删点。源/目标 BOLD 形状和 TR、四份影像复制身份检查通过。完整命令及检查在 `E:/ASD/fmriproc_local_test/logs/ingest.log`。
- `docker/Dockerfile.clean` 由真实 Neurodocker 2.1.2 经 `docker/generate_runtime.sh` 生成；生成时的工具依赖仅在临时目录解包，已清理。10 个 RUN shell 片段及关键 POSIX 路径检查通过。该干净环境是候选升级，AFNI/Workbench 的真实下载地址、版本和 SHA256 需在构建机显式提供；详见 `docker/README.md`。
- 尚未完成：Docker 镜像 build、SIF 转换/实际挂载、真实 AFNI 联合设计对照、完整两人预处理、解剖/配准/表面目视 QC，以及真实运行时间和内存测量。它们不能由上述合成测试代替。后续应从当前源与这两个导入样本继续，不将原 ADNI pilot 记录当作此次 ABIDE v2.1 的验收结果。

### 15.2 2026-09-24：两名被试端到端测试（v2.2）

在 Windows 11 + Docker Desktop（VM 16 CPU / 11.7 GiB）上用现有镜像 `zhaochang07/myubuntu:neuro-v2` 完成。上面 15.1 列为未完成的完整两人预处理、目视 QC、运行时间和内存测量已在本轮完成。

- **自动化测试**：`bash tests/run_tests.sh` 在主机和镜像内均全部通过。bash 检查 160 项；Python 单元测试 394 个，镜像内 0 跳过，主机上 8 个需要 Linux bash 的测试跳过。
- **端到端**：`abide_smoke.conf`（synth，4 种去噪策略）和 `abide_local_fs.conf`（recon-all + surface）在 sub-0028744（GU_1，TDC）、sub-0029150（NYU_2，ASD）上所有阶段 ok。v2.2 最终代码下两套配置都完整重跑（版本号进入阶段 hash，全部重算；已完成的 recon-all 被沿用）：synth 11.1 min，FreeSurfer 25.8 min，退出码均为 0。首次运行时 recon-all 每人 74–80 min。容器内存峰值 7.0–7.8 GiB（SynthSeg，已裁剪；每 15 s 采样一次，真实峰值可能略高），FreeSurfer 配置单人不超过 2 GiB（recon-all 除外），所以 12 GiB 的 VM 上保持 `N_JOBS=1`。
- **QC**：4 个 run 没有 fail，唯一的 warn 是 GU_1 的 36p 剩余自由度 11（按保留帧计算）。FD 均值 0.13–0.17 mm，删帧 ≤ 2%，GM tSNR 50–55，配准 Dice 0.94–0.96，标准化 Dice 0.97–0.98。
- **纳入标准**（默认 `EXCLUDE_*`）：两人都被纳入；GU_1 的 36p 因自由度 11 < 15 只在该策略下被剔除，不进入组水平比较。测试集的 `dpabi_parameters_by_subject.csv`（带 BOM，`SUB_ID` 为数字）与被试名匹配成功，组报告列出 ASD 与 TDC 两组。
- **Volume 与 surface**（n = 2，只能描述）：两流 FC 相关 0.85–0.88。surface 的分半信度、同伦对比和 ROI tSNR 略低，头动耦合相同。
- **去噪后 FD–DVARS 为负**（−0.19 ~ −0.46）：用 numpy 重建 3dTproject 投影做对照，负值来自头动回归量在高运动帧的高杠杆，相当于软删帧；同样自由度的随机回归量不会产生负值。报告和 `docs/STEPS_zh.md` 的说明已改正。
- **本轮修复与新增**：见 `CHANGELOG.md` 的 2.2.0。
- **完整报告**（逐阶段耗时、QC 表、策略与流比较、纳入表、对照实验）在本机 `E:\ASD\fmriproc_out\TEST_REPORT.md`，与影像结果一起保留在本机，不随代码发布。
- **尚未完成**：更大样本（`abide_test.conf` 或服务器批量）上的组水平 QC-FC、流比较和 ASD/TDC 头动比较；NYU_2 的 slice 顺序确认；`docker/Dockerfile.clean` 干净镜像的构建与 SIF 转换。
