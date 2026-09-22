#!/usr/bin/env python3
"""Extract atlas means only from covered, nonconstant voxels; report coverage."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

def extract(dataset, atlas, mask, output):
    images=[nib.load(p) for p in (dataset,atlas,mask)]
    image,atlas_image,mask_image=images
    if len(image.shape)!=4 or atlas_image.shape!=image.shape[:3] or mask_image.shape!=image.shape[:3]:
        raise ValueError('BOLD, atlas and static 3D coverage mask have incompatible shapes')
    if not all(np.allclose(i.affine,image.affine,atol=1e-4) for i in images[1:]):
        raise ValueError('Atlas and coverage mask must be on the BOLD grid')
    labels=atlas_image.get_fdata().astype(int)
    coverage=mask_image.get_fdata()>0
    data=image.get_fdata(dtype=np.float32)
    ids=np.unique(labels[labels>0])
    series=[]; rows=[]
    for label in ids:
        region=labels==label
        selected=region & coverage
        if not selected.any():
            values=np.empty((0,image.shape[3]),dtype=np.float32)
        else:
            values=data[selected]
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite BOLD in ROI {label}')
            values=values[np.ptp(values,axis=1)>0]
        rows.append((int(label),int(region.sum()),len(values),len(values)/int(region.sum())))
        series.append(values.mean(axis=0,dtype=np.float64) if len(values) else np.full(image.shape[3],np.nan))
    output=Path(output)
    np.savetxt(output,np.asarray(series).T,fmt='%.10g')
    np.savetxt(output.with_suffix('.coverage.tsv'),rows,delimiter='\t',
               header='roi\tatlas_voxels\tvalid_voxels\tcoverage_fraction',comments='',fmt=['%d','%d','%d','%.6f'])
    (output.with_suffix('.json')).write_text(json.dumps(dict(
        source=str(dataset),atlas=str(atlas),mask=str(mask),labels=ids.tolist(),
        missing_rois=[r[0] for r in rows if r[2]==0],
        definition='Mean over covered finite voxels with nonzero temporal range. Missing ROI = NaN; no imputation.'),indent=2)+'\n')
    print(f'{output}: {image.shape[3]} volumes, {len(ids)} ROIs, {sum(r[2]==0 for r in rows)} missing')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['dataset','atlas','mask','output']:p.add_argument(key,type=Path)
    a=p.parse_args();extract(a.dataset,a.atlas,a.mask,a.output)
