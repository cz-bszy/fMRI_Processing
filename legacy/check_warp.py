#!/usr/bin/env python3
"""Reject folded/non-finite warps inside the reference brain, report outside too."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

def check(jacobian,reference_mask,output):
    jac=nib.load(jacobian); brain=nib.load(reference_mask)
    if jac.shape!=brain.shape or not np.allclose(jac.affine,brain.affine,atol=1e-4):
        raise ValueError('Jacobian and reference brain must share a grid')
    data=jac.get_fdata(); mask=brain.get_fdata()>0
    if not mask.any() or not np.isfinite(data[mask]).all():
        raise ValueError('Missing brain or non-finite brain Jacobians')
    summary=dict(brain_min=float(data[mask].min()),brain_max=float(data[mask].max()),
                 brain_nonpositive=int(np.count_nonzero(data[mask]<=0)),
                 outside_nonpositive=int(np.count_nonzero(data[~mask]<=0)),
                 definition='Jacobian determinant including affine; nonpositive brain voxels invalidate this warp.')
    output.write_text(json.dumps(summary,indent=2)+'\n')
    if summary['brain_nonpositive']:
        raise ValueError(f"Folded/collapsed brain warp: {summary['brain_nonpositive']} voxels")
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['jacobian','reference_mask','output']:p.add_argument(key,type=Path)
    a=p.parse_args();check(a.jacobian,a.reference_mask,a.output)
