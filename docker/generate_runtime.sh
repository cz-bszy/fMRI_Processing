#!/usr/bin/env bash
# Generate only. Requires an already installed Neurodocker 2.1.2; never installs it.
set -euo pipefail
# Preserve container paths when a Windows Python Neurodocker is called from Git Bash.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
NEURODOCKER=${NEURODOCKER:-neurodocker}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:22.04}
case "$BASE_IMAGE" in ubuntu:22.04|ubuntu:22.04@sha256:*) ;; *) echo 'BASE_IMAGE must be Ubuntu 22.04 (optionally digest pinned)' >&2; exit 2;; esac
version=$("$NEURODOCKER" --version)
[[ $version =~ (^|[[:space:]])2\.1\.2($|[[:space:]]) ]] || { echo "Neurodocker 2.1.2 required, found: $version" >&2; exit 2; }
"$NEURODOCKER" generate docker --yes --pkg-manager apt --base-image "$BASE_IMAGE" \
    --install ca-certificates curl unzip tcsh bc parallel libgomp1 libglu1-mesa libgl1 libxt6 libxmu6 libxm4 libgsl27 libjpeg62 libpng16-16 libnetcdf19 libglib2.0-0 \
    --freesurfer version=7.4.1 install_path=/opt/freesurfer \
    --ants version=2.6.2 install_path=/opt/ants-2.6.2 \
    --fsl version=6.0.7.22 install_path=/opt/fsl \
    --miniconda version=py311_25.7.0-2 install_path=/opt/micromamba env_name=neuro env_exists=false \
        conda_opts='--override-channels -c conda-forge' \
        conda_install='python=3.11.11 pip=25.0.1' \
        pip_install='numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 nibabel==5.3.2 nilearn==0.12.1 scikit-learn==1.6.1 matplotlib==3.10.3 jinja2==3.1.6 templateflow==25.1.1' \
    --env PYTHON_BIN=/opt/micromamba/envs/neuro/bin/python PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1
cat <<'DOCKERFILE'

# AFNI has rolling binary URLs. Supply a verified binary URL, its sha256 and
# reported version together. Workbench candidate is 2.2.1. No placeholder hashes.
ARG AFNI_URL
ARG AFNI_SHA256
ARG AFNI_VERSION
ARG WORKBENCH_URL
ARG WORKBENCH_SHA256
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN for name in AFNI_SHA256 WORKBENCH_SHA256; do value=${!name}; \
      [[ "$value" =~ ^[a-f0-9]{64}$ && "$value" != 0000000000000000000000000000000000000000000000000000000000000000 ]] || exit 2; done && \
    [[ "$AFNI_URL" == https://* && "$WORKBENCH_URL" == https://* && "$AFNI_VERSION" =~ ^[0-9]{2}\.[0-9]\.[0-9]{2}$ ]] && \
    mkdir -p /opt/afni && \
    curl --fail --location --retry 3 "$AFNI_URL" -o /tmp/afni.tgz && \
    printf '%s  %s\n' "$AFNI_SHA256" /tmp/afni.tgz | sha256sum -c - && \
    tar -xzf /tmp/afni.tgz -C /opt/afni --strip-components=1 && \
    curl --fail --location --retry 3 "$WORKBENCH_URL" -o /tmp/workbench.zip && \
    printf '%s  %s\n' "$WORKBENCH_SHA256" /tmp/workbench.zip | sha256sum -c - && \
    unzip -q /tmp/workbench.zip -d /opt && \
    rm /tmp/afni.tgz /tmp/workbench.zip
ENV PATH=/opt/afni:/opt/workbench/bin_linux64:/opt/micromamba/envs/neuro/bin:${PATH} \
    FS_LICENSE=/opt/freesurfer/license.txt
# Existing launchers use bash -lc. Ubuntu's login profile may reset PATH; restore
# these explicit runtime locations without depending on Docker ENTRYPOINT.
RUN printf '%s\n' \
    'export PATH=/opt/afni:/opt/workbench/bin_linux64:/opt/micromamba/envs/neuro/bin:/opt/ants-2.6.2:/opt/fsl/share/fsl/bin:/opt/fsl/bin:/opt/freesurfer/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/mni/bin:$PATH' \
    > /etc/profile.d/fmriproc.sh
RUN afni -ver | grep -F "AFNI_${AFNI_VERSION}" && wb_command -version | grep -F '2.2.1' && \
    for tool in recon-all fslmaths fslstats mri_binarize antsRegistration antsApplyTransforms \
      N4BiasFieldCorrection CreateJacobianDeterminantImage mri_synthstrip mri_synthseg \
      mri_convert mris_euler_number mcflirt flirt fslmerge fslsplit 3dTshift 3dDespike \
      3dToutcount 3dTstat 3dcalc 3dTproject 3dBlurInMask wb_command bbregister mri_coreg \
      lta_diff lta_convert rmsdiff parallel; do command -v "$tool" || exit 1; done && \
    3dTproject -help >/dev/null && wb_command -list-commands >/dev/null && \
    "$PYTHON_BIN" -c 'import numpy, scipy, pandas, nibabel, nilearn, sklearn, matplotlib, jinja2, templateflow' && \
    mkdir -p /opt/fmriproc /data /out /config /opt/runtime-record && \
    "$PYTHON_BIN" -m pip freeze > /opt/runtime-record/python-freeze.txt && \
    /opt/micromamba/bin/conda list -n neuro --explicit > /opt/runtime-record/conda-explicit.txt && \
    printf 'AFNI=%s\nAFNI_SHA256=%s\nWORKBENCH=2.2.1\nWORKBENCH_SHA256=%s\n' \
      "$AFNI_VERSION" "$AFNI_SHA256" "$WORKBENCH_SHA256" > /opt/runtime-record/binaries.txt
WORKDIR /opt/fmriproc
CMD ["bash", "-l"]
DOCKERFILE
