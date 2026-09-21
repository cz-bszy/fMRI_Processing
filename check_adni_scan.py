#!/usr/bin/env python3
"""Focused numerical checks and descriptive QC for one completed pilot scan."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

def check(root,subject,session):
    scan=root/'output'/subject/session
    func=scan/'func'; reg=func/'reg_dir'; anat=scan/'anat'
    timing=json.loads((func/'preprocessing.json').read_text())
    mask_image=nib.load(func/'rest_mask.nii.gz')
    mask=mask_image.get_fdata()>0
    assert len(mask_image.shape)==3 and mask.any()
    assert np.array_equal(mask,nib.load(func/'rest_pp_mask.nii.gz').get_fdata()>0)
    for name in ['segment_prob_0.nii.gz','segment_prob_1.nii.gz','segment_prob_2.nii.gz']:
        tissue=nib.load(anat/name)
        brain=nib.load(anat/'Stru_Brain.nii.gz')
        assert tissue.shape==brain.shape and np.allclose(tissue.affine,brain.affine)
    prior=np.loadtxt(anat/'std2sub.mat'); forward=np.loadtxt(anat/'sub2std.mat')
    assert np.allclose(prior@forward,np.eye(4),atol=1e-5)
    assert np.array_equal(np.loadtxt(reg/'func2struct.mat'),np.loadtxt(reg/'example_func2highres.mat'))
    warp_quality=json.loads((reg/'warp_quality.json').read_text())
    assert warp_quality['brain_nonpositive']==0
    tissue_counts={}
    for name in ['global','csf','wm']:
        im=nib.load(func/'seg'/f'{name}_mask.nii.gz'); data=im.get_fdata()>0
        assert im.shape==mask_image.shape and np.allclose(im.affine,mask_image.affine)
        assert data.any() and not np.any(data&~mask)
        tissue_counts[name]=int(data.sum())
    # The final retained nuisance directory is the last (no-GSR) model.
    regressors=np.loadtxt(func/'nuisance'/'regressors.1D')
    residual_image=nib.load(func/'rest_res.nii.gz')
    residual=residual_image.get_fdata(dtype=np.float32)[mask].astype(np.float64)
    residual_image.uncache()
    assert residual.shape[1]==timing['retained_volumes']==regressors.shape[0]
    assert np.isfinite(residual).all() and np.isfinite(regressors).all()
    centered=regressors-regressors.mean(0)
    denom=np.linalg.norm(residual,axis=1)[:,None]*np.linalg.norm(centered,axis=0)[None,:]
    normalized_dot=np.divide(residual@centered,denom,out=np.zeros_like(denom),where=denom>0)
    max_projection=float(np.abs(normalized_dot).max())
    # Dividing by a small residual amplifies roundoff after aggressive denoising.
    # Check projection error against the fitted (centered) input scale instead.
    fit_input=nib.load(func/'nuisance'/'centered.nii.gz').get_fdata(dtype=np.float32)[mask]
    input_denominator=np.linalg.norm(fit_input.astype(np.float64),axis=1)[:,None]*np.linalg.norm(centered,axis=0)[None,:]
    scaled_error=np.divide(np.abs(residual@centered),input_denominator,
                           out=np.zeros_like(input_denominator),where=input_denominator>0)
    max_scaled_error=float(scaled_error.max())
    # Two float32 length-N inner products: gamma_N = N*eps/(1-N*eps).
    n_timepoints=residual.shape[1]
    accumulation=n_timepoints*np.finfo(np.float32).eps
    precision_bound=float(2*accumulation/(1-accumulation))
    assert max_scaled_error <= precision_bound, f'Projection error exceeds float32 input-scale bound: {max_scaled_error} > {precision_bound}'
    models={}
    for model in ['NoGRS','Retain_GRS']:
        name=f'{subject}_{session}_{model}'
        path=root/'output'/'results'/model
        im=nib.load(path/f'{name}.nii.gz')
        assert im.shape[3]==timing['retained_volumes']
        ts=np.loadtxt(path/'timeseries'/f'{name}_timeseries.1D')
        cov=np.loadtxt(path/'timeseries'/f'{name}_timeseries.coverage.tsv',skiprows=1)
        assert ts.shape==(im.shape[3],len(cov))
        valid=cov[:,2]>0
        assert np.isfinite(ts[:,valid]).all()
        assert np.isnan(ts[:,~valid]).all()
        models[model]=dict(shape=list(im.shape),roi_count=len(cov),missing_rois=cov[~valid,0].astype(int).tolist(),
                           roi_coverage_median=float(np.median(cov[:,3])),roi_coverage_min=float(cov[:,3].min()))
    old_mask=nib.load(root/'baseline'/subject/session/'rest_mask.nii.gz')
    old_count=int(np.count_nonzero(old_mask.get_fdata()))
    voxel_mm3=float(abs(np.linalg.det(mask_image.affine[:3,:3])))
    motion=np.loadtxt(func/'rest_mc.1D')
    delta=np.diff(motion,axis=0,prepend=motion[:1])
    fd=np.abs(delta[:,3:]).sum(1)+50*np.deg2rad(np.abs(delta[:,:3])).sum(1)
    result=dict(subject=subject,session=session,checks='passed',timing=timing,
                warp_quality=warp_quality,new_mask_voxels=int(mask.sum()),old_mask_voxels=old_count,
                new_mask_volume_ml=int(mask.sum())*voxel_mm3/1000,
                mask_grid_affine_unchanged=np.allclose(old_mask.affine,mask_image.affine),
                nuisance_tissue_voxels=tissue_counts,max_residual_nuisance_normalized_dot=max_projection,
                max_input_scaled_projection_error=max_scaled_error,float32_projection_precision_bound=precision_bound,
                mean_FD_mm=float(fd.mean()),FD_above_0_5mm_count=int((fd>.5).sum()),
                censored_volumes=0,models=models,
                scope='Exploratory processing QC; more covered voxels alone does not establish improved registration.')
    (scan/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('subject');p.add_argument('session')
    a=p.parse_args();check(a.root,a.subject,a.session)
