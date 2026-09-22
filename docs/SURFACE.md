# Surface branch (stage 06): fsLR-32k / 91k CIFTI 输出

本文说明可选的 surface 分支做什么、需要什么、在 3–4 mm 的 legacy 数据上有哪些局限，
以及 stage 07 / 08 / 10 如何使用它的输出并与 volume 流进行比较。文件名和 JSON key
以 `docs/DESIGN.md`（section 6、7 "06"、10、12）为准；本文描述的是
`stages/06_surface.sh`、`stages/fetch_resources.sh`、`py/fmriproc/fetch_resources.py`、
`py/fmriproc/surface_utils.py` 的实际行为。

## 1. 这个分支做什么

volume 流（默认）把 BOLD 一次性重采样到 template 空间，在 MNI 体素上做 denoising、
smoothing 和 atlas 提取。surface 分支在此之外、并行地产生一套 HCP 风格的 grayordinate
数据：

* 皮层：用 subject 自己的 FreeSurfer white / pial 面在 **T1w 空间的 BOLD** 上做
  ribbon-constrained volume-to-surface mapping，再重采样到 fsLR-32k 标准网格
  （每个半球 32492 个 vertex，medial wall 之外 29696 + 29716 个）。
* 皮层下：直接取 template 空间 2 mm 的 BOLD（FSL MNI152 2 mm 网格 = HCP grayordinate
  体素网格，`MNI152NLin6Asym`），用 HCP 的 `Atlas_ROIs.2.nii.gz` 选出 31870 个体素
  （19 个结构：thalamus、caudate、putamen、pallidum、hippocampus、amygdala、accumbens、
  ventral diencephalon、brain stem、cerebellum）。
* 两部分合成一个 91282 grayordinate 的 dense time series（`.dtseries.nii`），
  对 pre-denoise（`desc-preproc`）和每一个 denoising strategy 各生成一个，可选地再做
  geodesic smoothing。

它 **不** 替代 volume 流：两条流都保留，stage 07 分别提取 ROI 时间序列，stage 10 做
配对比较（section 7）。选择哪条流用于最终分析，应基于 stage 10 的比较结果和 section 6
的 caveats，而不是先验假设 surface 更好。

## 2. 要求

| 项目 | 说明 |
|---|---|
| `SURFACE=yes` | 默认 `no`。`fp_init` 会检查 `ANAT_MODE=freesurfer` 和 `MNI_RES=2`，否则直接报错。 |
| FreeSurfer recon-all | stage 01 完成的 `$FS_DIR/sub-X/surf/{lh,rh}.{white,pial,sphere.reg,thickness}`。`ANAT_MODE=synth` 没有面，分支不可用。 |
| FreeSurfer license | `FS_LICENSE` 或 `/opt/freesurfer/license.txt`（`mris_convert` 没有 license 会退出）。 |
| 工具 | `wb_command`（Connectome Workbench 2.1.0）、`mris_convert`、FSL `fslmaths/fslstats/fslnvols/fslval`、`$PYTHON_BIN`（nibabel）。 |
| 一次性资源下载 | `stages/fetch_resources.sh`，需要网络（见 2.1）。 |
| stage 05 的 T1w 空间 residual | `SURFACE=yes` 时 stage 05 对 T1w 空间的 preproc 序列也跑一遍 `3dTproject`，输出 `work/sub-X/func/<RUN>/denoise/<S>_space-T1w_bold.nii.gz`。这是 05 → 06 的 hand-over 文件（DESIGN.md section 10），在 `work/` 里；如果 stage 05 是在 `SURFACE=no` 下跑的，需要 `FORCE=yes STAGES=denoise` 重跑。 |
| 内存 | 分支本身很轻（3 mm T1w-space BOLD、32k metric、2 mm 子皮层），`fp_check_mem` 与其它 stage 一样按 `MIN_MEM_GB` 检查。 |

### 2.1 一次性资源下载（`fetch_resources`）

```
bash stages/fetch_resources.sh -c config/datasets/<dataset>.conf           # 下载
bash stages/fetch_resources.sh -c config/datasets/<dataset>.conf --check   # 只检查，exit 0 = 齐全
```

`run_pipeline.sh` 在 `SURFACE=yes` 或 `ATLASES` 非空时自动先跑 `--check`，缺什么就跑一次
下载；也可以把 `fetch` 显式写进 `STAGES`。下载写入 `$RESOURCE_DIR`
（默认 `$OUT_DIR/resources`）：

```
templateflow/                              TEMPLATEFLOW_HOME（fp_init 导出）
  tpl-fsLR/tpl-fsLR_hemi-{L,R}_den-32k_sphere.surf.gii
  tpl-fsLR/tpl-fsLR_space-fsaverage_hemi-{L,R}_den-32k_sphere.surf.gii     fsnative -> fsLR 重采样用
  tpl-fsLR/tpl-fsLR_hemi-{L,R}_den-32k_desc-nomedialwall_dparc.label.gii
  tpl-fsLR/tpl-fsLR_den-32k_hemi-{L,R}_{midthickness,inflated}.surf.gii    只用于 QC 图
  tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_atlas-Schaefer2018_desc-<N>Parcels<M>Networks_dseg.nii.gz (+ dseg.tsv)
hcp/Atlas_ROIs.2.nii.gz                    HCPpipelines/global/templates/91282_Greyordinates/
hcp/{L,R}.atlasroi.32k_fs_LR.shape.gii     HCPpipelines/global/templates/standard_mesh_atlases/
atlases/<A>/<A>_space-MNI152NLin6Asym_res-02_dseg.nii.gz   TemplateFlow 文件的拷贝（volume 流用）
atlases/<A>/labels.tsv                     列 index, name, network（network 从 7Networks_LH_Vis_1 之类的名字解析）
atlases/<A>/<A>.dlabel.nii                 CBIG .../Parcellations/HCP/fslr32k/cifti/<A>_order.dlabel.nii
```

`<A>` 只支持 `Schaefer2018_<N>Parcels_<M>Networks`（N = 100…1000，M = 7|17）；其它名字
只给 warning，不下载（自定义 atlas 走 `CUSTOM_ATLASES`）。规则：

* TemplateFlow 文件先用 `templateflow.api.get`，失败则直接从
  `https://templateflow.s3.amazonaws.com` 取；HCP / CBIG 文件从 `raw.githubusercontent.com` 取。
  `urllib`，3 次重试、每次连接 60 s 超时；写临时文件后 rename；已存在且非空的文件跳过。
* 下载内容做 sanity 检查：Git-LFS pointer、HTML 页面（captive portal / proxy）、非 gzip 的
  `.nii.gz`、非 XML 的 `.gii` 都算失败，不会留下看似合法的文件。
* HCP `atlasroi` 下不到时用 TemplateFlow 的 `desc-nomedialwall` label 转成同样的 metric ROI
  （`surface_utils label-to-roi`，非零 key = cortex；期望 29696 / 29716 个 vertex）。
* `SURFACE=no` 时跳过 fsLR / HCP 文件和 dlabel，只取 volume atlas + labels。
* 镜像：`--github-raw URL`、`--templateflow-s3 URL`（或环境变量 `FMRIPROC_GITHUB_RAW`、
  `FMRIPROC_TEMPLATEFLOW_S3`）；代理走 `https_proxy`。`--no-templateflow-api` 跳过 API。
* `--check` 不需要网络、不写任何文件；stdout 每行 `MISSING<TAB>描述<TAB>路径`。

注意：`docker run --rm` 的容器里写到容器文件系统的东西会随容器消失。`$RESOURCE_DIR`
必须落在持久位置（named volume 或 bind mount）。TemplateFlow 第一次 `import` 时会在
`TEMPLATEFLOW_HOME` 下解包一个全是 **0 字节占位文件** 的 skeleton，所以"文件存在"不等于
"已下载"，`--check` 和 `--locate` 只认非空文件。

## 3. 处理步骤（`stages/06_surface.sh <sub>`）

`SURFACE != yes` 时打印一行日志后 `exit 0`（不写 marker；stage 07 的 `--dep 06_surface__<RUN>`
按 "missing" 参与 hash，这是预期行为）。

### 3.1 Subject 级（每个 subject 一次）

marker `06_surface`（无 run），依赖 `01_anat_recon` 的 marker 和 `ANAT_MODE`；文件齐全且
hash 未变时跳过。全部在临时目录里完成，两个半球都成功后才 `mv` 到 `derivatives/sub-X/anat/`。

```
mris_convert --to-scanner lh.white  -> sub-X_hemi-L_white.surf.gii
mris_convert --to-scanner lh.pial   -> sub-X_hemi-L_pial.surf.gii
mris_convert lh.sphere.reg          -> work/.../sub-X_hemi-L_desc-reg_sphere.surf.gii     (球面不加 --to-scanner)
mris_convert -c lh.thickness lh.white -> thickness.shape.gii
wb_command -set-structure ... CORTEX_LEFT -surface-type ANATOMICAL -surface-secondary-type GRAY_WHITE|PIAL / SPHERICAL
wb_command -surface-average midthickness -surf white -surf pial            -> sub-X_hemi-L_midthickness.surf.gii
wb_command -metric-math 'thickness > 0' roi ; -metric-fill-holes ; -metric-remove-islands   -> native cortex ROI（medial wall = thickness 0）
wb_command -surface-resample midthickness sphere.reg tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere BARYCENTRIC
                                                                          -> sub-X_hemi-L_space-fsLR_den-32k_midthickness.surf.gii
```

`--to-scanner` 是关键：FreeSurfer 面默认是 tkr 坐标，与 T1w / BOLD 的 world 坐标差一个
`c_ras`（可达 ~20 mm）；没有它 ribbon mapping 会悄无声息地采到错误组织。v2 的 T1w 空间
BOLD 已由 stage 03 一次性重采样到 FreeSurfer conformed 网格（world = 输入 T1 的 scanner
坐标），所以面 **不需要任何仿射变换**，直接使用。

### 3.2 Run 级（每个 run）

marker `06_surface__<RUN>`，依赖 `05_denoise__<RUN>` 和 subject 级 marker，hash 变量
`SURF_SMOOTH_FWHM DENOISE_STRATEGIES`。开始前删除旧的 `<RUN>_space-fsLR_den-91k_*` 和
`<RUN>_desc-surfqc.json`，避免失败后留下半套过期结果。

1. **goodvoxels**（HCP `RibbonVolumeToSurfaceMapping` 规则）在 **T1w 空间的 preproc 序列**
   `<RUN>_space-T1w_desc-preproc_bold.nii.gz` 上计算（scaled、未去均值；denoised residual
   均值为 0，CoV 无定义，不能用）：
   * `wb_command -create-signed-distance-volume` 用 white / pial 面在 BOLD 网格上生成
     signed distance，`white_dist > 0 ∩ pial_dist < 0` = 每个半球的 ribbon，合并成 `ribbon_only`；
   * `cov = Tstd / Tmean`；在 ribbon 内归一化（除以 ribbon 内均值），再用 5 mm 高斯做局部
     归一化（`-s 5`，HCP 值），得到 `cov_norm_modulate`；
   * 阈值 = ribbon 内均值 + 0.5 × SD；`goodvoxels = (mean > 0) − (cov_norm_modulate > 阈值)`。
     这去掉血管、边缘和 partial-volume 严重的体素。
   * ribbon 内没有任何有信号体素 ⇒ 直接报错 "no BOLD signal inside the cortical ribbon"
     （面和 BOLD 不重叠：通常是 stage 03 coregistration 或 T1 / recon 不是同一个 subject）。

2. **mapping + 重采样**，对 preproc 序列和每个 strategy `S` 各做一次（每半球）：
   ```
   wb_command -volume-to-surface-mapping <bold_T1w> midthickness native.func.gii \
       -ribbon-constrained white pial -volume-roi <goodvoxels> -voxel-subdiv 5|7 -bad-vertices-out badvert
   wb_command -metric-dilate native.func.gii midthickness 10 native.func.gii -bad-vertex-roi badvert -nearest
   wb_command -metric-mask native.func.gii <cortex ROI> native.func.gii
   wb_command -metric-resample native.func.gii sphere.reg tpl-fsLR_space-fsaverage_..._sphere ADAP_BARY_AREA out.32k.func.gii \
       -area-surfs midthickness sub-X_hemi-L_space-fsLR_den-32k_midthickness -current-roi <cortex ROI> [-valid-roi-out]
   wb_command -metric-mask out.32k.func.gii L.atlasroi.32k_fs_LR.shape.gii out.32k.func.gii
   ```
   * `-voxel-subdiv`：`prep_info.json` 的 `voxel_size`（原始 BOLD 体素）最大边 ≥ 3.5 mm 时用 7，
     否则 5（wb 默认 3 在 bert 上 3 mm 漏掉 0.5 % 的 vertex，4 mm 漏掉 5.7 %；5 / 7 降到 < 0.1 %）。
   * 对 strategy 序列，`goodvoxels` 再与 `Tstd > 0` 相交：stage 05 的 `3dTproject -mask` 把
     brain mask 外置零，这些常数体素不能进入 ribbon 平均。
   * dilation 只填 `-bad-vertices-out` 标出的 vertex（`-bad-vertex-roi`）：denoised 序列的值可以
     是 0，不能用 "值为 0" 判断无数据。
   * 重采样到 fsLR 之后 **没有** 再 dilate（与 HCP 一致）：subject cortex ROI 与 fsLR atlasroi 的
     medial wall 边界不一致处的 vertex 在所有 dtseries 里都是 0，比例记录在
     `pct_fslr_vertices_nodata`；stage 07 按 dlabel parcel 统计有数据的 grayordinate 比例，
     低于 `MIN_ROI_COVERAGE` 的 parcel 为 `n/a`。

3. **网格检查 + dense time series**。template 空间的 BOLD 必须与 `Atlas_ROIs.2.nii.gz`
   同网格（`surface_utils check-grid`：dims + affine，`atol 1e-3`）：`same` 直接用；
   `reordered`（同一格点、存储轴序不同）用 `-volume-resample ENCLOSING_VOXEL` 无损重排；
   其它情况报错（需要 `MNI_RES=2`、`TEMPLATE_NAME=MNI152NLin6Asym`）。cortex 和 subcortex
   的 volume 数必须一致（`CENSOR_MODE=KILL` 时两者都变短，没问题）。
   ```
   wb_command -cifti-create-dense-timeseries out.dtseries.nii \
       -volume <bold_MNI2mm> Atlas_ROIs.2.nii.gz \
       -left-metric L.32k.func.gii -roi-left L.atlasroi.32k_fs_LR.shape.gii \
       -right-metric R.32k.func.gii -roi-right R.atlasroi.32k_fs_LR.shape.gii -timestep <TR>
   ```
   输入：preproc 用 `<RUN>_space-T1w_desc-preproc_bold.nii.gz` + `<RUN>_space-<TPL>_res-2_desc-preproc_bold.nii.gz`；
   strategy `S` 用 `work/.../denoise/<S>_space-T1w_bold.nii.gz` + `<RUN>_space-<TPL>_res-2_desc-<S>_bold.nii.gz`。

4. **tSNR**：`wb_command -cifti-reduce <preproc dtseries> TSNR` → `desc-preproc_tsnr.dscalar.nii`
   （mean / sample SD，沿时间；无数据的 vertex 为 0 或 NaN，QC 里不计）。

5. **可选 smoothing**（`SURF_SMOOTH_FWHM > 0`，默认 5 mm）：
   ```
   wb_command -cifti-smoothing <S>.dtseries.nii F F COLUMN <S>sm<F>.dtseries.nii -fwhm \
       -left-surface sub-X_hemi-L_space-fsLR_den-32k_midthickness.surf.gii -right-surface ... \
       -fix-zeros-volume -fix-zeros-surface
   ```
   皮层沿 subject 自己的 32k midthickness 做 geodesic 平滑，皮层下按结构分开平滑（不跨结构）；
   `-fix-zeros-*` 把 0（无数据 / FOV 外）当缺失而不是信号。`<F>` 去掉小数点（4.5 → `sm45`），
   与 stage 05 的 volume 平滑副本同一写法。ROI 时间序列（stage 07）**只用未平滑的** dtseries。

6. **QC JSON** `<RUN>_desc-surfqc.json`（`surface_utils surfqc`）：

   | key | 含义 |
   |---|---|
   | `pct_badvertices` | native cortex ROI 内 ribbon 多面体没碰到任何 goodvoxel 的 vertex 百分比（dilation 之前） |
   | `pct_goodvoxels_excluded` | ribbon 内 **有数据** 的体素中被 CoV 规则剔除的百分比 |
   | `pct_ribbon_outside_mask`, `n_ribbon_voxels` | ribbon 落在 BOLD 无信号区（FOV 外 / dropout）的比例 |
   | `tsnr_cortex_median`, `n_vertices_valid`, `n_vertices_total` | pre-denoise tSNR 在皮层 grayordinate 上的中位数（只计有限且 > 0 的 vertex） |
   | `tsnr_subcortex_median`, `n_subcortex_voxels_valid`, `n_subcortex_voxels_total` | 同上，皮层下 31870 个体素 |
   | `pct_fslr_vertices_nodata` | fsLR atlasroi 内没有从 subject cortex ROI 拿到数据的 vertex 百分比 |
   | `voxel_subdiv`, `strategies`, `surf_smooth_fwhm` | 本次运行的参数 |

   stage 08 把 `tsnr_cortex_median / pct_badvertices / pct_goodvoxels_excluded` 并入
   `<RUN>_desc-qc_metrics.json`，并在报告里画 fsLR 表面的 tSNR（需要 `templateflow/tpl-fsLR`
   的 inflated / midthickness 网格）。

7. `KEEP_WORK=no`（默认）时删除 `work/sub-X/func/<RUN>/surface/`；subject 级的
   `work/sub-X/anat/surface/`（sphere.reg、thickness、cortex ROI）保留，因为每个 run 都要用。

### 3.3 输出一览

```
derivatives/sub-X/anat/
  sub-X_hemi-{L,R}_{white,pial,midthickness}.surf.gii             T1w world 坐标
  sub-X_hemi-{L,R}_space-fsLR_den-32k_midthickness.surf.gii       subject 面在 fsLR-32k 网格上
derivatives/sub-X/func/
  <RUN>_space-fsLR_den-91k_desc-preproc_bold.dtseries.nii         pre-denoise, scaled
  <RUN>_space-fsLR_den-91k_desc-preproc_tsnr.dscalar.nii
  <RUN>_space-fsLR_den-91k_desc-<S>_bold.dtseries.nii             每个 strategy，未平滑
  <RUN>_space-fsLR_den-91k_desc-<S>sm<F>_bold.dtseries.nii        SURF_SMOOTH_FWHM > 0 时
  <RUN>_desc-surfqc.json
```

stage 07 在此基础上产生 `<RUN>_space-fsLR_atlas-<A>_desc-<S>_timeseries.tsv / _connectivity.tsv`
和 `<RUN>_space-fsLR_atlas-<A>_desc-preproc_timeseries.tsv`（`wb_command -cifti-parcellate ... -method MEAN`，
失败时回退 `-legacy-mode`；列序 = dlabel 的 label key 升序，列名 = dlabel 的 parcel 名）。

## 4. 与 HCP / fMRIPrep 的差异

* 没有 fieldmap，没有 SDC（section 6）；没有 MSMSulc / MSMAll，fsnative → fsLR 用 FreeSurfer
  `sphere.reg`（folding-based）经 `tpl-fsLR_space-fsaverage_..._sphere` 重采样（HCP
  `resample_fsaverage` 路线）。
* 皮层下不做 HCP 的 subject-ROI `-volume-parcel-resampling`，直接取 template 空间体素
  （fMRIPrep 同样如此）。
* goodvoxels 在 T1w 空间的 3 mm（`FUNC_T1W_RES`）网格上算，而不是原始 EPI 网格；面不做
  任何仿射（BOLD 已在 T1w 空间）。
* dilation 用显式的 bad-vertex ROI 而不是 "值 == 0"，以便对 zero-mean 的 residual 也正确。

## 5. 为什么 volume 仍是默认

1. surface 分支需要 recon-all（每 subject 数小时）加人工面 QC；recon 失败的 subject 只有
   volume 结果，两条流的样本集合可能不同。
2. 在 3–4 mm 数据上 surface 的理论优势（subject-specific GM 采样、不跨脑沟的平滑、
   folding-based 对齐）被 partial volume 和无 SDC 的配准误差大幅稀释（section 6）；收益要
   用 stage 10 在本数据集上实证，不能假设。
3. 皮层下 / 小脑无论如何都来自 volume 流；Schaefer dlabel 只有皮层。
4. 现有下游分析和 legacy v1 的结果都是 volume 的；surface 作为对照和敏感性分析更稳妥。

## 6. 在 3–4 mm legacy 数据上的 caveats

* **Partial volume 是主导因素。** 皮层厚 2–3 mm，EPI 体素 3–4 mm（ABIDE 测试数据：
  3.0–3.6 × 3.0–3.6 × 3.0–4.0 mm），一个体素同时包含 GM、WM、CSF；ribbon-constrained 平均
  只能按几何加权，不能恢复采集时已经混合的信号。goodvoxels 会剔除 CoV 异常的体素，剔除率
  （`pct_goodvoxels_excluded`）明显高于 HCP 数据（HCP 2 mm 上通常个位数百分比）属正常；
  过高（> 20–30 %）通常意味着 coregistration 或 recon 有问题，而不是数据本身。
* **没有 SDC。** EPI 在 OFC、颞极、颞下回有几毫米的几何畸变和 dropout，而面来自无畸变的
  T1。这些区域的 vertex 采到的是错位的组织甚至空气：表现为 `pct_badvertices` 局部升高、
  这些区域 tSNR 很低、Schaefer 的 OFC / temporal-pole parcel 覆盖率低。volume 流同样受畸变
  影响，但 atlas parcel 的体积平均对错位不那么敏感。解释 surface 结果时把这些 parcel 视为
  不可靠；不要试图用 `--use-syn-sdc` 类方法"修"3–4 mm 数据。
* **FreeSurfer 面的质量取决于 T1。** NYU / UM 等站点的 T1 是 1.2–1.4 mm 各向异性 sagittal
  MPRAGE，recon-all 会 conform 到 1 mm 但白质 / 软膜面在细节上不可靠，Euler number 偏低、
  拓扑缺陷多（看 `sub-X_desc-anatqc.json` 的 `euler_lh/rh`、`holes_total` 和报告里的面 overlay）。
  儿童（EMC 等 6–10 岁）头动大、对比度不同，SynthStrip / recon 的 pial 面容易包进硬膜。
  面错了，ribbon 就错，surface 流的所有下游都错——请把 surface 分支 **门控在 anat QC 之后**，
  面 QC 不过的 subject 只用 volume 结果。
* **FOV 截断。** 一些站点的 z-FOV 切掉小脑 / 顶点，subcortex 的 cerebellum 结构会有大片 0；
  `-fix-zeros-volume` 在平滑时把它们当缺失，stage 07 的 coverage 规则把不完整 parcel 标
  `n/a`。`pct_ribbon_outside_mask` 反映皮层顶部被切的比例。
* **fsLR-32k 相对 3–4 mm 数据过密。** 相邻 vertex（~2 mm）来自同一个体素，vertex 级数据
  高度冗余；parcel 平均或 4–6 mm 的 `-cifti-smoothing` 才是合适的分析单位。
* **DOF 与 censoring 与 volume 流完全相同**（同一组 regressor、同一 censor 向量）：surface
  不会改变 `dof_remaining`，也不会让 TR 2–3 s、150 volume 的 run 变得更可靠。

## 7. stage 10 如何比较两条流

`stages/10_validate.sh`（`fmriproc.validate`，DESIGN.md section 12）只读 stage 07 的 ROI 表，
对每个 run × stream (`volume` | `surface`) × strategy × atlas 计算同一组指标：
`roi_tsnr_median / p10`、`variance_removed_median`、`split_half_r`、`network_contrast`、
`homotopic_contrast`、`dmn_contrast`、`lowfreq_power_fraction`、`fd_fc_coupling`、
`gs_residual_sd`、`n_roi_nan`、`n_retained`、`dof_remaining`。

* Schaefer 的 volume dseg 和 fsLR dlabel 含同样的皮层 parcel、同样的顺序，所以两条流按
  **列序** 逐 parcel 配对（列名可能略有差异；列数不同则跳过并 warning）；只在两条流都有效
  的 ROI 上比较。
* 每个 run 写 `<RUN>_desc-streamcompare.tsv`：`fc_similarity`（两条流 Fisher-z FC 上三角的
  Pearson r，正常 > 0.8–0.9，低值指向配准或 recon 问题）、上述每个指标的 surface − volume
  配对差、逐 ROI 的两流 `roi_tsnr`。
* group 级（`fmriproc.compare_streams`）：`stream_comparison.tsv`（每 strategy × atlas × metric：
  n、两流中位数、配对差中位数、Wilcoxon signed-rank p、按指标方向判定哪条流更好）、
  `fc_typicality.tsv`（与 leave-one-out 组均 FC 的相关）、`stream_comparison.png`、
  `validation_report.html`。
* 判读原则：没有单一指标能定胜负。当 `split_half_r`、`network_contrast`、`homotopic_contrast`、
  `fc_typicality` 上升而 `fd_fc_coupling`、QC-FC 下降，且 DOF 可接受时，才认为一条流更好。
  tSNR 会因过度 denoising / 低通而虚高，要和可靠性指标一起看。surface 的 `roi_tsnr` 通常
  略高（goodvoxels 剔除了噪声体素），这本身不说明 FC 更好。

## 8. 重跑、清理与常见错误

* marker：`work/sub-X/.done/06_surface.hash`（subject 级）和 `06_surface__<RUN>.hash`。
  重跑 recon（stage 01 marker 变化）会使 subject 级和所有 run 级失效；改 `SURF_SMOOTH_FWHM`
  或 `DENOISE_STRATEGIES` 只使 run 级失效。`FORCE=yes STAGES=surface` 强制重跑。
* `KEEP_WORK=yes` 保留 `work/sub-X/func/<RUN>/surface/`（goodvoxels、ribbon、native
  metric、badvert 等），用于排查 mapping 问题。
* "no BOLD signal inside the cortical ribbon"：面和 T1w 空间 BOLD 不重叠。检查
  `<RUN>_space-T1w_boldref.nii.gz` 上叠加 `sub-X_hemi-L_white.surf.gii` 是否贴合（wb_view /
  freeview），以及 recon 与 rawdata 的 T1 是否同一 subject。
* "... is not on the grid of .../Atlas_ROIs.2.nii.gz"：template 空间输出不是 FSL MNI152 2 mm
  网格；确认 `MNI_RES=2`、`TEMPLATE_NAME=MNI152NLin6Asym`。
* "<S>_space-T1w_bold.nii.gz is missing"：stage 05 在 `SURFACE=no` 下跑过，或 `work/` 被清理；
  `FORCE=yes STAGES="denoise surface"`。
* "fsLR-32k registration spheres not found"：`stages/fetch_resources.sh` 没跑或 `$RESOURCE_DIR`
  不持久；`--check` 列出缺失文件。
* `-cifti-parcellate` 报缺失 brainordinate：stage 07 自动回退 `-legacy-mode`。


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
