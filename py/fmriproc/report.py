"""Per-subject QC report: one self-contained HTML file (docs/DESIGN.md section 9).

    python -m fmriproc.report --deriv-dir $OUT_DIR/derivatives --subject sub-X \
        --template MNI152NLin6Asym --strategies "wmcsf24 wmcsf24gsr" --atlases "A B"

Reads ``<RUN>_desc-qc_metrics.json``, ``<RUN>_desc-prep_info.json``, the anat QC
JSON, the optional stage-10 tables and the PNG files written by
``fmriproc.plots``; images are embedded (base64), there is no external CSS or
JavaScript, so the file works offline and can be mailed around. Anything that is
missing is shown as "n/a" / "figure not available" - the report is always written.

The explanatory notes are in Chinese with English technical terms (the reader
is a Chinese-speaking researcher); figure text stays English.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import BaseLoader, Environment, select_autoescape

from fmriproc import utils
from fmriproc.plots import anat_figures, figure_path, run_figures
from fmriproc.qc_metrics import FLAG_RANK, read_json_safe

LOG = logging.getLogger("fmriproc.report")
MAX_TABLE_ROWS = 400

# (metric key, column header, flag key or None)
SUMMARY_COLUMNS = [
    ("fd_mean", "FD mean (mm)", "fd_mean"),
    ("pct_censored", "censored (%)", "pct_censored"),
    ("minutes_retained", "retained (min)", "minutes_retained"),
    ("tsnr_gm_median", "tSNR GM, pre", "tsnr_gm_median"),
    ("coreg_dice", "coreg Dice", "coreg_dice"),
    ("norm_dice", "norm Dice", "norm_dice"),
    ("holes_total", "surface holes", "holes_total"),
]
STRATEGY_COLUMNS = [
    ("n_regressors", "regressors"), ("dof_remaining", "DOF left"), ("tsnr_pre_same_voxels", "tSNR pre"),
    ("tsnr_gm_median_post", "tSNR post"), ("tsnr_gain", "gain (post/pre)"), ("variance_removed_gm", "variance removed"),
    ("fd_dvars_corr_post", "FD-DVARS r, post"), ("tsnr_post_mask", "voxels"),
]
ATLAS_COLUMNS = [
    ("n_roi", "ROIs"), ("n_roi_nan", "ROIs n/a"), ("coverage_min", "min coverage"), ("roi_tsnr_pre_median", "ROI tSNR pre"),
    ("roi_tsnr_median", "ROI tSNR final (median)"), ("roi_tsnr_min", "ROI tSNR final (min)"),
    ("roi_variance_removed_median", "variance removed"), ("split_half_r", "split-half r"),
    ("network_contrast", "network contrast"), ("fc_mean", "FC mean"), ("fc_sd", "FC SD"),
    ("surf_split_half_r", "surface split-half r"), ("vol_surf_fc_r", "volume-surface FC r"),
]
PROVENANCE_ROWS = [
    ("source", "source file"), ("tr", "TR (s)"), ("n_volumes_raw", "volumes, raw"), ("n_dropped", "volumes dropped"),
    ("n_volumes", "volumes used"), ("nss_detected", "non-steady-state volumes detected"), ("voxel_size", "voxel size (mm)"),
    ("obliquity_deg", "obliquity (deg)"), ("despike", "3dDespike"), ("despike_fraction", "despiked fraction"),
    ("stc_applied", "slice timing correction applied"), ("stc_reason", "STC decision"),
    ("slice_timing_source", "slice timing source"), ("slice_timing_evidence", "slice timing evidence"),
    ("stc_interp", "STC interpolation"), ("tzero", "STC tzero (s)"), ("hmc_reference", "HMC reference"),
    ("coreg_method", "coregistration method"), ("bbr_cost", "BBR cost"), ("bbr_vs_init_mm", "BBR vs. initialisation (mm)"),
    ("bbr_rejected", "BBR rejected"), ("epi_mask_method", "EPI mask method"), ("scale_factor", "global scale factor"),
    ("func_t1w_res", "T1w-space BOLD grid (mm)"), ("mni_res", "template-space grid (mm)"), ("template", "template"),
    ("wm_mask_voxels", "WM mask voxels (BOLD grid)"), ("csf_mask_voxels", "CSF mask voxels"),
    ("gm_mask_voxels", "GM mask voxels"), ("tissue_erosion_relaxed", "tissue erosion relaxed"),
]
ANATQC_ROWS = [
    ("anat_mode", "anatomical mode"), ("euler_lh", "Euler number LH"), ("euler_rh", "Euler number RH"),
    ("holes_total", "surface holes (total)"), ("brain_volume_mm3", "brain volume (mm3)"),
    ("norm_quality", "normalisation preset"), ("norm_dice", "Dice(warped mask, template mask)"),
    ("template_corr", "correlation with the template"), ("jacobian_p01", "Jacobian p01"), ("jacobian_p50", "Jacobian p50"),
    ("jacobian_p99", "Jacobian p99"), ("jacobian_nonpos_frac", "Jacobian <= 0 fraction"),
    ("wm_voxels", "WM mask voxels"), ("csf_voxels", "CSF mask voxels"), ("gm_voxels", "GM mask voxels"),
]
SIGNAL_ROWS = [
    ("tsnr_gm_median", "tSNR GM (median, pre-denoise)"), ("tsnr_wm_median", "tSNR WM"), ("tsnr_brain_median", "tSNR brain"),
    ("dvars_std_mean", "standardised DVARS, mean"), ("outlier_frac_mean", "outlier fraction, mean (AOR)"),
    ("quality_index_mean", "3dTqual index, mean (AQI)"), ("gcor", "GCOR"), ("fd_dvars_corr", "FD-DVARS r, pre"),
    ("gs_fd_corr", "FD - |d global signal| r"), ("fwhm_acf", "smoothness, ACF FWHM (mm)"),
    ("dropout_fraction", "dropout fraction (T1 brain without EPI signal)"), ("brain_mask_voxels", "brain mask voxels (BOLD grid)"),
    ("tsnr_cortex_median", "surface: tSNR cortex (median)"), ("pct_badvertices", "surface: bad vertices (%)"),
    ("pct_goodvoxels_excluded", "surface: ribbon voxels excluded (%)"),
]
MOTION_ROWS = [
    ("fd_mean", "FD mean (mm)"), ("fd_median", "FD median"), ("fd_max", "FD max"), ("fd_pct_gt_02", "FD > 0.2 mm (%)"),
    ("fd_pct_gt_05", "FD > 0.5 mm (%)"), ("relrms_mean", "relative RMS mean (mm)"), ("absrms_max", "absolute RMS max (mm)"),
    ("n_censored", "censored frames"), ("pct_censored", "censored (%)"), ("minutes_retained", "minutes retained"),
    ("longest_segment", "longest uncensored segment (frames)"),
]

NOTES = {
    "summary": "总览表：每个 run 一行。绿色 pass / 黄色 warn / 红色 fail 由配置文件中的 QC_* 阈值决定，n/a 表示缺少该指标的输入文件；"
               "overall 取该 run 所有 flag 中最差的一个。warn 表示需要人工看图确认，fail 一般建议剔除或重新处理。",
    "provenance": "采集与处理记录（来自 prep_info.json）。重点看：slice timing correction (STC) 是否执行及其依据"
                  "（evidence 等级 A/B/C，unknown 时按设计不做 STC）；丢弃的 volume 数是否覆盖了检测到的 non-steady-state volume；"
                  "coregistration 最终采用的方法，以及 BBR 是否因代价或位移过大被拒绝 (bbr_rejected)。",
    "anat": "T1w 与 brain mask（黄）及组织 mask 轮廓。brain mask 应紧贴脑表面，不含硬脑膜/眼眶，也不能切掉皮层；"
            "WM / CSF mask 是腐蚀 (eroded) 之后用于提取噪声信号的区域，应完全落在白质 / 侧脑室内部，不能碰到灰质。",
    "template": "T1w → template 的 normalisation。红线是模板的灰白质边界 (GM/WM edge)，叠加在配准后的 T1w 上；下半部分相反，"
                "把被试的灰白质边界画在模板上。线与背景图像的灰白质交界重合说明配准良好，重点看脑室、皮层边缘和小脑。"
                "norm_dice 偏低、Jacobian ≤ 0 的比例大于 0 或 Jacobian p01/p99 极端，都提示形变场异常。",
    "surfaces": "FreeSurfer 表面与 T1w 切面的交线：蓝色 white surface 应沿灰白质交界，红色 pial surface 应沿皮层外缘；"
                "表面穿入白质、漏掉脑回或包进硬脑膜都会影响 surface 分支的采样。",
    "boldref": "BOLD reference（已变换到 T1w 空间）与 brain mask、组织 mask。mask 应覆盖有信号的全部脑组织；"
               "WM / CSF mask 体素过少（见 provenance 表）时 aCompCor 和 WM/CSF 回归量不稳定。",
    "epi_t1": "EPI → T1w coregistration。上半部分：蓝线为 T1w 的白质边界、黄线为 T1w brain mask，叠加在 BOLD reference 上，"
              "蓝线应与 EPI 图像中灰白质对比的交界吻合；整体平移或旋转说明配准失败。下半部分：红线为 BOLD mask 在 T1w 上的范围，"
              "用来看 FOV 覆盖和信号丢失 (dropout，常见于眶额叶、颞极)。coreg_dice 为 EPI 有信号区域与 T1w brain mask 的重合度。",
    "epi_tpl": "EPI → template（头动校正、EPI→T1w、T1w→template 三个变换合成后一次插值的结果）。黄线为模板 brain mask，"
               "蓝线为模板灰白质边界；应与 BOLD reference 的脑轮廓和灰白质对比一致。",
    "motion": "头动参数 (translation / rotation)、framewise displacement (FD, Power)、standardised DVARS 和 3dToutcount outlier fraction。"
              "红色阴影是被 censoring 的帧，虚线为 FD 阈值。FD 与 DVARS 同时出现尖峰说明头动造成了信号突变；"
              "保留时间 (minutes retained) 过短时该 run 的 FC 不可靠。",
    "carpet": "Carpet plot：每行一个体素（z-score），自上而下依次为 GM / WM / CSF，左侧色条标出组织类别，顶部为 FD 曲线。"
              "去噪前与 FD 峰对齐的竖条纹是头动造成的全脑同步信号变化，去噪后应明显减弱；若仍然存在，"
              "可考虑更严格的 censoring 或包含 global signal regression (GSR) 的策略。去噪后的数据在 template 空间。",
    "tsnr": "temporal SNR 图，前后使用同一色标。去噪前 tSNR = 时间均值 / 去趋势 (polort 2) 后的 SD；"
            "最终时间序列均值为 0，所以去噪后的 tSNR 定义为 去噪前均值 / 去噪后残差 SD（只用保留帧）。"
            "眶额、颞极等 dropout 区域 tSNR 低是正常的；整体偏低或左右明显不对称需要检查原始数据。",
    "confounds": "混杂变量之间的相关矩阵。头动参数与 global signal / WM / CSF 高度相关，说明头动对信号影响大；"
                 "回归量之间高度共线不影响残差，但会使单个回归系数失去意义。",
    "denoise": "每种去噪策略的定量结果。DOF left = 保留帧数 − 回归量个数 − 带通滤波消耗的自由度，低于 MIN_DOF 标为 warn；"
               "gain = 去噪后 / 去噪前 tSNR；variance removed = 1 − var(去噪后) / var(去噪前去趋势)；"
               "FD-DVARS r (post) 越接近 0 说明残余的头动相关信号越少。variance removed 很高而 DOF 很低时要警惕过度回归。",
    "fc": "功能连接矩阵（Pearson r，仅用保留帧），ROI 按 network 排序，应能看到沿对角线的网络内块状结构；灰色行列是覆盖率不足被置为 n/a 的 ROI。"
          "直方图：不做 GSR 时分布整体偏正，做 GSR 后中心接近 0。split-half r 为前后两半数据 FC 的相关（可靠性），"
          "network contrast =（网络内平均 z − 网络间平均 z）/ 网络间 SD，越大说明网络结构越清晰。",
    "roiqc": "每个 ROI 的时间序列质量：ROI tSNR = 去噪前 ROI 均值 / 最终 ROI 序列的 SD（保留帧）；variance removed；"
             "coverage = ROI 内有效体素比例，低于 MIN_ROI_COVERAGE 的 ROI 不输出时间序列（n/a，红色三角）。"
             "对应的表格为 *_roiqc.tsv。",
    "surface": "皮层表面 (fsLR-32k) 上的去噪前 tSNR。局部 tSNR 极低的区域通常对应 dropout 或 ribbon 采样失败 (bad vertices)。",
    "validation": "Stage 10 的时间序列有效性指标（volume 与 surface 两条 stream、各策略、各 atlas）。没有单一指标能决定优劣："
                  "split_half_r、network / homotopic / DMN contrast 越高越好，fd_fc_coupling 越低越好，同时要保证 dof_remaining 足够。"
                  "streamcompare 表给出 surface − volume 的配对差值和两条 stream 的 FC 相似度 (fc_similarity)。",
    "all": "该 run 的全部 QC 指标（<RUN>_desc-qc_metrics.json 的原始内容），供检索。",
}

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ subject }} - fMRI QC</title>
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --page:#f9f9f7; --card:#ffffff;
        --pass:#0ca30c; --warn:#fab219; --fail:#d03b3b; --na:#898781; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Helvetica,Arial,sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line); padding:18px 28px; }
header h1 { margin:0 0 4px; font-size:22px; }
header p { margin:0; color:var(--ink2); font-size:13px; }
nav { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:8px 28px; font-size:13px; }
nav a { color:var(--accent); text-decoration:none; margin-right:14px; white-space:nowrap; }
main { max-width:1500px; margin:0 auto; padding:18px 28px 60px; }
section { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px 20px; margin:0 0 18px; }
h2 { font-size:18px; margin:2px 0 10px; }
h3 { font-size:15px; margin:20px 0 8px; padding-top:10px; border-top:1px solid var(--line); }
h2 + h3 { border-top:none; padding-top:0; margin-top:6px; }
.note { background:#f3f6fb; border-left:3px solid var(--accent); padding:8px 12px; margin:8px 0 12px; color:var(--ink2); font-size:13px; border-radius:0 4px 4px 0; }
.problem { background:#fdf1f0; border-left:3px solid var(--fail); padding:8px 12px; margin:8px 0 12px; font-size:13px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:13px; margin:6px 0 10px; }
th, td { padding:4px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; background:#fafaf8; }
td:first-child, th:first-child { text-align:left; }
table.kv td { text-align:left; white-space:normal; }
table.kv td:first-child { color:var(--ink2); width:320px; }
.badge { display:inline-block; min-width:44px; padding:1px 8px; border-radius:10px; color:#fff; font-size:12px; font-weight:600; text-align:center; }
.badge.pass { background:var(--pass); } .badge.warn { background:var(--warn); color:#3a2a00; }
.badge.incomplete { background:#8855aa; } .badge.fail { background:var(--fail); } .badge.na { background:var(--na); }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:baseline; }
.dot.pass { background:var(--pass); } .dot.warn { background:var(--warn); } .dot.fail { background:var(--fail); } .dot.na { background:#d5d4ce; }
figure { margin:10px 0 16px; }
figure img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:4px; display:block; }
figcaption { color:var(--muted); font-size:12px; margin-top:3px; }
.missing { color:var(--muted); font-style:italic; font-size:13px; margin:6px 0 12px; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); font-size:13px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:0 28px; }
footer { color:var(--muted); font-size:12px; text-align:center; padding:12px; }
</style>
</head>
<body>
{% macro badge(flag) %}<span class="badge {{ 'na' if flag == 'n/a' else flag }}">{{ flag }}</span>{% endmacro %}
{% macro cell(value, flag=None) %}{% if flag %}<span class="dot {{ 'na' if flag == 'n/a' else flag }}" title="{{ flag }}"></span>{% endif %}{{ value | fmt }}{% endmacro %}
{% macro note(key) %}<div class="note">{{ notes[key] }}</div>{% endmacro %}
{% macro fig(item) %}{% if item.data %}<figure><img alt="{{ item.name }}" src="data:image/png;base64,{{ item.data }}"><figcaption>{{ item.file }}</figcaption></figure>{% else %}<p class="missing">figure not available: {{ item.file }}</p>{% endif %}{% endmacro %}
{% macro kv(rows) %}<table class="kv">{% for label, value in rows %}<tr><td>{{ label }}</td><td>{{ value | fmt }}</td></tr>{% endfor %}</table>{% endmacro %}
{% macro frame(table) %}<div class="scroll"><table><tr>{% for c in table.columns %}<th>{{ c }}</th>{% endfor %}</tr>
{% for row in table.rows %}<tr>{% for v in row %}<td>{{ v | fmt }}</td>{% endfor %}</tr>{% endfor %}</table></div>
{% if table.truncated %}<p class="missing">table truncated to {{ table.rows | length }} rows; see {{ table.file }}</p>{% endif %}{% endmacro %}

<header>
  <h1>{{ subject }} <span style="font-weight:400;color:var(--ink2)">resting-state fMRI QC</span></h1>
  <p>fMRI_Processing v2 &middot; template {{ template }} &middot; strategies: {{ strategies | join(', ') or 'n/a' }} &middot; atlases: {{ atlases | join(', ') or 'n/a' }} &middot; generated {{ generated }}{% if site %} &middot; site/group: {{ site }}{% endif %}</p>
</header>
<nav>
  <a href="#summary">Summary</a><a href="#provenance">Acquisition &amp; provenance</a><a href="#anat">Anatomy</a>
  {% for run in runs %}<a href="#{{ run.id }}">{{ run.label }}</a>{% endfor %}
</nav>
<main>

<section id="summary">
  <h2>Summary 总览</h2>
  {{ note('summary') }}
  {% if runs %}
  <div class="scroll"><table>
    <tr><th>run</th><th>overall</th>{% for _, header, _ in summary_columns %}<th>{{ header }}</th>{% endfor %}
      {% for s in strategies %}<th>{{ s }}: tSNR post</th><th>{{ s }}: ROI tSNR</th><th>{{ s }}: DOF</th>{% endfor %}</tr>
    {% for run in runs %}
    <tr><td><a href="#{{ run.id }}">{{ run.label }}</a></td><td>{{ badge(run.overall) }}</td>
      {% for key, _, flag in summary_columns %}<td>{{ cell(run.metrics.get(key), run.flags.get(flag, 'n/a') if flag else None) }}</td>{% endfor %}
      {% for s in strategies %}
      <td>{{ cell(run.metrics.get(s ~ '.tsnr_gm_median_post')) }}</td>
      <td>{{ cell(run.metrics.get(s ~ '.' ~ first_atlas ~ '.roi_tsnr_median') if first_atlas else None) }}</td>
      <td>{{ cell(run.metrics.get(s ~ '.dof_remaining'), run.flags.get(s ~ '.dof_remaining', 'n/a')) }}</td>
      {% endfor %}</tr>
    {% endfor %}
  </table></div>
  <p class="missing">ROI tSNR column: median over ROIs of atlas {{ first_atlas or 'n/a' }}. Thresholds: {% for name, t in thresholds.items() %}{{ name }} warn {{ t.warn | fmt }} / fail {{ t.fail | fmt }}{{ '; ' if not loop.last }}{% endfor %}</p>
  {% else %}<p class="missing">no runs found for {{ subject }}</p>{% endif %}
</section>

<section id="provenance">
  <h2>Acquisition &amp; provenance 采集与处理记录</h2>
  {{ note('provenance') }}
  {% if runs %}
  <div class="scroll"><table class="kv">
    <tr><th></th>{% for run in runs %}<th style="text-align:left">{{ run.label }}</th>{% endfor %}</tr>
    {% for key, label in provenance_rows %}
    <tr><td>{{ label }}</td>{% for run in runs %}<td>{{ run.prep.get(key) | fmt }}</td>{% endfor %}</tr>
    {% endfor %}
    <tr><td>tool versions</td>{% for run in runs %}<td>{% for k, v in (run.prep.get('tool_versions') or {}).items() %}{{ k }}: {{ v }}<br>{% endfor %}</td>{% endfor %}</tr>
  </table></div>
  {% endif %}
</section>

<section id="anat">
  <h2>Anatomy 结构像</h2>
  <div class="grid2"><div>{{ kv(anat_rows) }}</div><div>
    <table class="kv"><tr><td>norm Dice</td><td>{{ badge(anat_flags.get('norm_dice', 'n/a')) }}</td></tr>
    <tr><td>surface holes</td><td>{{ badge(anat_flags.get('holes_total', 'n/a')) }}</td></tr></table></div></div>
  <h3>T1w, brain mask, tissue masks</h3>{{ note('anat') }}{{ fig(anat_figs['anat-mosaic']) }}
  <h3>T1w &rarr; {{ template }}</h3>{{ note('template') }}{{ fig(anat_figs['anat-template']) }}
  {% if anat_figs['anat-surfaces'].data %}<h3>Surfaces on the T1w</h3>{{ note('surfaces') }}{{ fig(anat_figs['anat-surfaces']) }}{% endif %}
</section>

{% for run in runs %}
<section id="{{ run.id }}">
  <h2>{{ run.label }} {{ badge(run.overall) }}</h2>
  {% if not run.has_metrics %}<div class="problem">No <code>{{ run.label }}_desc-qc_metrics.json</code>: the QC metrics of this run were not computed (see logs/{{ subject }}/08_qc.log).</div>{% endif %}
  {% if run.metrics.get('qc_problems') %}<div class="problem">QC problems: {{ run.metrics.get('qc_problems') }}</div>{% endif %}
  <p>{% for name, flag in run.flags.items() %}<span style="margin-right:12px;white-space:nowrap"><span class="dot {{ 'na' if flag == 'n/a' else flag }}"></span>{{ name }}: {{ flag }}</span> {% endfor %}</p>

  <h3>Registration 配准</h3>
  {{ note('boldref') }}{{ fig(run.figs['boldref-mask']) }}
  {{ note('epi_t1') }}{{ fig(run.figs['epi-to-t1']) }}
  {{ note('epi_tpl') }}{{ fig(run.figs['epi-to-template']) }}

  <h3>Motion &amp; frame-wise quality 头动</h3>
  {{ note('motion') }}
  <div class="grid2"><div>{{ kv(run.motion_rows) }}</div><div>{{ kv(run.signal_rows) }}</div></div>
  {{ fig(run.figs['motion']) }}

  <h3>Carpet plots 去噪前后</h3>{{ note('carpet') }}{{ fig(run.figs['carpet']) }}
  <h3>Temporal SNR</h3>{{ note('tsnr') }}{{ fig(run.figs['tsnr']) }}
  <h3>Confounds 混杂变量</h3>{{ note('confounds') }}{{ fig(run.figs['confound-corr']) }}

  <h3>Denoising strategies 去噪策略</h3>{{ note('denoise') }}
  <div class="scroll"><table><tr><th>strategy</th>{% for _, header in strategy_columns %}<th>{{ header }}</th>{% endfor %}</tr>
  {% for s in strategies %}<tr><td>{{ s }}</td>{% for key, _ in strategy_columns %}<td>{{ cell(run.metrics.get(s ~ '.' ~ key), run.flags.get(s ~ '.' ~ key) if key == 'dof_remaining' else None) }}</td>{% endfor %}</tr>{% endfor %}
  </table></div>

  <h3>Functional connectivity &amp; ROI time series 功能连接与 ROI 时间序列</h3>{{ note('fc') }}
  <div class="scroll"><table><tr><th>atlas</th><th>strategy</th>{% for _, header in atlas_columns %}<th>{{ header }}</th>{% endfor %}</tr>
  {% for a in atlases %}{% for s in strategies %}<tr><td>{{ a }}</td><td style="text-align:left">{{ s }}</td>{% for key, _ in atlas_columns %}<td>{{ run.metrics.get(s ~ '.' ~ a ~ '.' ~ key) | fmt }}</td>{% endfor %}</tr>{% endfor %}{% endfor %}
  </table></div>
  {% for a in atlases %}
  {{ fig(run.figs['atlas-' ~ a ~ '_fc']) }}
  {% if loop.first %}{{ note('roiqc') }}{% endif %}
  {{ fig(run.figs['atlas-' ~ a ~ '_roiqc']) }}
  {% endfor %}

  {% if run.figs['surface-tsnr'].data %}<h3>Surface tSNR</h3>{{ note('surface') }}{{ fig(run.figs['surface-tsnr']) }}{% endif %}

  {% if run.validation or run.streamcompare %}
  <h3>Time-series validation 时间序列有效性 (stage 10)</h3>{{ note('validation') }}
  {% if run.validation %}{{ frame(run.validation) }}{% endif %}
  {% if run.streamcompare %}<p><b>volume vs. surface</b> (paired difference = surface &minus; volume; source: {{ run.streamcompare.file }})</p>
    {% if run.streamcompare.run %}{{ frame(run.streamcompare.run) }}{% endif %}
    {% if run.streamcompare.roi %}<details><summary>per-ROI tSNR of both streams ({{ run.streamcompare.roi.rows | length }} rows)</summary>{{ frame(run.streamcompare.roi) }}</details>{% endif %}{% endif %}
  {% endif %}

  <details><summary>all metrics of this run 全部指标</summary>{{ note('all') }}
  <table class="kv">{% for key, value in run.metrics.items() if key not in ('flags', 'thresholds') %}<tr><td>{{ key }}</td><td>{{ value | fmt }}</td></tr>{% endfor %}</table></details>
</section>
{% endfor %}
</main>
<footer>fMRI_Processing v2 QC report &middot; {{ subject }} &middot; {{ generated }}</footer>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def fmt(value: Any) -> str:
    """Human-readable cell: n/a for missing, yes/no for booleans, 4 significant digits."""
    if value is None:
        return utils.NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return " x ".join(fmt(v) for v in value) if value else utils.NA
    if isinstance(value, dict):
        return ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) or utils.NA
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return utils.NA
        if value == int(value) and abs(value) < 1e9:
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def embed(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def figure_items(fig_dir: Path, prefix: str, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    items = {}
    for name in names:
        path = figure_path(fig_dir, prefix, name)
        items[name] = {"name": name, "file": path.name, "data": embed(path)}
    return items


def table_payload(frame: pd.DataFrame | None, file_name: str) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    frame = frame.astype(object).where(frame.notna(), None)
    rows = frame.head(MAX_TABLE_ROWS).values.tolist()
    rows = [[v.item() if hasattr(v, "item") else v for v in row] for row in rows]
    return {"columns": [str(c) for c in frame.columns], "rows": rows, "truncated": len(frame) > MAX_TABLE_ROWS,
            "file": file_name}


def _read_optional_tsv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return utils.read_tsv(path)
    except Exception as err:  # noqa: BLE001 - a damaged optional table only loses its section
        LOG.warning("cannot read %s: %s", path, err)
        return None


def validation_table(path: Path) -> dict[str, Any] | None:
    """Long stage-10 table (stream strategy atlas metric value) -> metric x stream/strategy/atlas."""
    frame = _read_optional_tsv(path)
    if frame is None or frame.empty:
        return None
    needed = {"stream", "strategy", "atlas", "metric", "value"}
    if needed <= set(frame.columns):
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["column"] = frame["stream"].astype(str) + " | " + frame["strategy"].astype(str) + " | " + frame["atlas"].astype(str)
        order = list(dict.fromkeys(frame["metric"].astype(str)))
        wide = frame.pivot_table(index="metric", columns="column", values="value", aggfunc="first", dropna=False)
        wide = wide.reindex(order).reset_index()
        wide.columns.name = None
        frame = wide
    return table_payload(frame, path.name)


def streamcompare_tables(path: Path) -> dict[str, Any] | None:
    """Stage-10 stream comparison (long: strategy atlas scope roi metric volume surface value).

    Returns {"run": run-level pivot (metric rows; volume / surface / surface-volume
    per strategy | atlas), "roi": the per-ROI rows, "file": name}, or None.
    """
    frame = _read_optional_tsv(path)
    if frame is None or frame.empty:
        return None
    needed = {"strategy", "atlas", "scope", "metric", "volume", "surface", "value"}
    if not needed <= set(frame.columns):
        return {"run": table_payload(frame, path.name), "roi": None, "file": path.name}
    frame = frame.copy()
    for column in ("volume", "surface", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    is_roi = frame["scope"].astype(str) == "roi"
    run_rows, roi_rows = frame[~is_roi].copy(), frame[is_roi]
    run_payload = None
    if not run_rows.empty:
        run_rows["column"] = run_rows["strategy"].astype(str) + " | " + run_rows["atlas"].astype(str)
        order = list(dict.fromkeys(run_rows["metric"].astype(str)))
        pieces = []
        for kind, label in (("volume", "volume"), ("surface", "surface"), ("value", "surface - volume")):
            wide = run_rows.pivot_table(index="metric", columns="column", values=kind, aggfunc="first", dropna=False)
            wide.columns = [f"{c} | {label}" for c in wide.columns]
            pieces.append(wide)
        wide = pd.concat(pieces, axis=1).reindex(order)
        # stable sort: volume, surface, difference stay together per strategy | atlas
        wide = wide[sorted(wide.columns, key=lambda c: c.rsplit(" | ", 1)[0])].reset_index()
        run_payload = table_payload(wide, path.name)
    roi_payload = table_payload(roi_rows.drop(columns=["scope"]), path.name) if not roi_rows.empty else None
    if run_payload is None and roi_payload is None:
        return None
    return {"run": run_payload, "roi": roi_payload, "file": path.name}


def discover_runs(func_dir: Path, subject: str) -> list[str]:
    runs = set()
    for pattern, suffix in (("*_desc-qc_metrics.json", "_desc-qc_metrics.json"), ("*_desc-prep_info.json", "_desc-prep_info.json")):
        for path in func_dir.glob(f"{subject}_{pattern}"):
            runs.add(path.name[: -len(suffix)])
    return sorted(runs)


def manifest_group(manifest: Path | None, subject: str) -> str | None:
    if manifest is None or not Path(manifest).is_file():
        return None
    try:
        table = pd.read_csv(manifest, sep="\t", dtype=str, encoding="utf-8-sig")
    except Exception as err:  # noqa: BLE001
        LOG.warning("cannot read %s: %s", manifest, err)
        return None
    if not {"subject", "group"} <= set(table.columns):
        return None
    groups = [g for g in table.loc[table["subject"] == subject, "group"].dropna().unique() if g and g != "-"]
    return ", ".join(groups) if groups else None


def _rows(source: dict, spec: list[tuple[str, str]], skip_missing: bool = False) -> list[tuple[str, Any]]:
    rows = [(label, source.get(key)) for key, label in spec]
    return [(label, value) for label, value in rows if not (skip_missing and value is None)]


def run_context(func_dir: Path, fig_dir: Path, run: str, atlases: list[str]) -> dict[str, Any]:
    metrics_path = func_dir / f"{run}_desc-qc_metrics.json"
    metrics = read_json_safe(metrics_path)
    flags = metrics.get("flags") if isinstance(metrics.get("flags"), dict) else {}
    return {
        "label": run,
        "id": "run-" + "".join(ch if ch.isalnum() else "-" for ch in run),
        "has_metrics": bool(metrics),
        "metrics": metrics,
        "flags": flags,
        "overall": metrics.get("overall_flag", "n/a") if metrics else "n/a",
        "prep": read_json_safe(func_dir / f"{run}_desc-prep_info.json"),
        "figs": figure_items(fig_dir, run, run_figures(atlases)),
        "motion_rows": _rows(metrics, MOTION_ROWS),
        "signal_rows": _rows(metrics, SIGNAL_ROWS, skip_missing=True) or _rows(metrics, SIGNAL_ROWS[:3]),
        "validation": validation_table(func_dir / f"{run}_desc-validation.tsv"),
        "streamcompare": streamcompare_tables(func_dir / f"{run}_desc-streamcompare.tsv"),
    }


def build_report(deriv_dir: Path, subject: str, runs: list[str], template: str, strategies: list[str],
                 atlases: list[str], manifest: Path | None = None) -> str:
    sub_dir = Path(deriv_dir) / subject
    func_dir, anat_dir, fig_dir = sub_dir / "func", sub_dir / "anat", sub_dir / "figures"
    if not runs and func_dir.is_dir():
        runs = discover_runs(func_dir, subject)
    contexts = [run_context(func_dir, fig_dir, run, atlases) for run in runs]
    anatqc = read_json_safe(anat_dir / f"{subject}_desc-anatqc.json")
    with_metrics = [c for c in contexts if c["has_metrics"]]
    anat_flags = {k: v for k, v in (with_metrics[0]["flags"].items() if with_metrics else []) if k in ("norm_dice", "holes_total")}
    thresholds = {k: v for k, v in ((with_metrics[0]["metrics"].get("thresholds") or {}).items() if with_metrics else [])
                  if isinstance(v, dict)}
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(default_for_string=True, default=True))
    env.filters["fmt"] = fmt
    return env.from_string(TEMPLATE).render(
        subject=subject, template=template, strategies=strategies, atlases=atlases,
        first_atlas=atlases[0] if atlases else None, site=manifest_group(manifest, subject),
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), runs=contexts, notes=NOTES,
        summary_columns=SUMMARY_COLUMNS, strategy_columns=STRATEGY_COLUMNS, atlas_columns=ATLAS_COLUMNS,
        provenance_rows=PROVENANCE_ROWS, anat_rows=_rows(anatqc, ANATQC_ROWS), anat_flags=anat_flags,
        anat_figs=figure_items(fig_dir, subject, anat_figures()), thresholds=thresholds,
    )


def worst_overall(deriv_dir: Path, subject: str) -> str:
    """Worst overall flag over the runs of a subject (used by callers for logging)."""
    func_dir = Path(deriv_dir) / subject / "func"
    flags = [read_json_safe(p).get("overall_flag", "n/a") for p in func_dir.glob(f"{subject}_*_desc-qc_metrics.json")]
    return max(flags, key=lambda f: FLAG_RANK.get(f, -1), default="n/a")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deriv-dir", required=True, help="$OUT_DIR/derivatives")
    parser.add_argument("--subject", required=True, help="sub-<id>")
    parser.add_argument("--runs", default="", help="space separated run prefixes; default: discovered in func/")
    parser.add_argument("--template", required=True)
    parser.add_argument("--strategies", default="")
    parser.add_argument("--atlases", default="")
    parser.add_argument("--manifest", default=None, help="rawdata/manifest.tsv (site/group in the header)")
    parser.add_argument("--out", default=None, help="default: <deriv-dir>/<subject>.html")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    deriv_dir = Path(args.deriv_dir)
    if not (deriv_dir / args.subject).is_dir():
        LOG.error("no derivatives for %s under %s", args.subject, deriv_dir)
        return 2
    html = build_report(deriv_dir, args.subject, args.runs.split(), args.template, args.strategies.split(),
                        args.atlases.split(), Path(args.manifest) if args.manifest else None)
    out = Path(args.out) if args.out else deriv_dir / f"{args.subject}.html"
    utils._atomic_write(out, html)
    LOG.info("%s: report -> %s (%.1f MB, worst run flag: %s)", args.subject, out, len(html) / 1e6,
             worst_overall(deriv_dir, args.subject))
    return 0


if __name__ == "__main__":
    sys.exit(main())
