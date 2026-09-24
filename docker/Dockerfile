# A reproducible compatibility layer over the EXISTING environment, not a
# reconstruction of its undocumented upstream installation. Supply a digest:
# docker build --build-arg BASE_IMAGE=zhaochang07/myubuntu@sha256:<digest> ...
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
SHELL ["/bin/bash", "-lc"]
RUN [[ "$BASE_IMAGE" =~ @sha256:[a-f0-9]{64}$ ]] || { echo 'BASE_IMAGE must be pinned by sha256 digest' >&2; exit 2; }
# Keep the inherited scientific software unchanged. Fail visibly if the selected
# base cannot supply the actual pipeline environment.
RUN for tool in recon-all fslmaths fslstats mri_binarize antsRegistration antsApplyTransforms \
    N4BiasFieldCorrection CreateJacobianDeterminantImage mri_synthstrip mri_synthseg \
    mri_convert mris_euler_number mcflirt flirt fslmerge fslsplit 3dTshift 3dDespike \
    3dToutcount 3dTstat 3dcalc 3dTproject 3dBlurInMask wb_command bbregister mri_coreg \
    lta_diff lta_convert rmsdiff parallel; do command -v "$tool" || exit 1; done && \
    /opt/micromamba/envs/neuro/bin/python -c 'import numpy, scipy, pandas, nibabel, nilearn, sklearn, matplotlib, jinja2, templateflow'
RUN mkdir -p /opt/fmriproc /data /out /config && chmod 755 /opt/fmriproc /data /out /config
# AFNI lives in /root/abin, but /root is drwx------ in the base image: Singularity/
# Apptainer run as the calling (non-root) user, and every 3d* program would be
# "not found". Open /root and prove the toolchain works for an unprivileged user.
RUN chmod 755 /root && \
    su -s /bin/bash nobody -c 'HOME=/tmp bash -lc "command -v 3dTproject 3dTshift 3dDespike recon-all mri_synthseg antsApplyTransforms wb_command >/dev/null && 3dTstat -help >/dev/null"'
ENV PYTHON_BIN=/opt/micromamba/envs/neuro/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    FS_LICENSE=/opt/freesurfer/license.txt
LABEL org.opencontainers.image.title="fMRI_Processing runtime compatibility layer" \
      org.opencontainers.image.description="Pinned upstream neuro environment; repository and data mounted at runtime"
WORKDIR /opt/fmriproc
CMD ["bash", "-l"]
