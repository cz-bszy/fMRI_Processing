"""Known-answer checks for slice-axis timing and missing/constant ROI support."""
import json
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
from prepare_bold import prepare
from extract_roi_timeseries import extract
from check_warp import check as check_warp
from prepare_regressors import prepare as prepare_regressors

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='helper_check_',dir=root) as folder:
    d=Path(folder).resolve()
    assert d.is_relative_to(root)  # Bound the automatic cleanup to this checkout.
    im=nib.Nifti1Image(np.zeros((2,2,4,10),np.float32),np.eye(4))
    im.header.set_zooms((1,1,1,2)); bold=d/'test_bold.nii.gz'; nib.save(im,bold)
    sidecar=d/'test_bold.json'
    sidecar.write_text(json.dumps(dict(RepetitionTime=2,SliceEncodingDirection='k-',SliceTiming=[0,0,1,1])))
    prepare(bold,d)
    assert np.array_equal(np.loadtxt(d/'slice_timing.1D'),[1,1,0,0])
    sidecar.write_text(json.dumps(dict(RepetitionTime=2)))
    try:
        prepare(bold,d)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing timing was silently accepted')
    values=np.zeros((2,2,2,3),np.float32)
    values[0,0,0]=[1,2,3];values[0,0,1]=[2,2,2]
    atlas=np.zeros((2,2,2),np.int16);atlas[0,0,:]=1;atlas[1,0,0]=2
    mask=np.zeros((2,2,2),np.uint8);mask[0,0,:]=1
    for name,arr in [('data',values),('atlas',atlas),('mask',mask)]:
        nib.save(nib.Nifti1Image(arr,np.eye(4)),d/f'{name}.nii.gz')
    out=d/'series.1D'
    extract(d/'data.nii.gz',d/'atlas.nii.gz',d/'mask.nii.gz',out)
    actual=np.loadtxt(out)
    assert np.array_equal(actual[:,0],[1,2,3])
    assert np.isnan(actual[:,1]).all()
    coverage=np.loadtxt(out.with_suffix('.coverage.tsv'),skiprows=1)
    assert np.array_equal(coverage[:,3],[0.5,0])
    jac=np.ones((2,2,2));jac[1,1,1]=-1
    nib.save(nib.Nifti1Image(jac,np.eye(4)),d/'jac.nii.gz')
    check_warp(d/'jac.nii.gz',d/'mask.nii.gz',d/'warp.json')
    jac[0,0,0]=-1
    nib.save(nib.Nifti1Image(jac,np.eye(4)),d/'jac.nii.gz')
    try:
        check_warp(d/'jac.nii.gz',d/'mask.nii.gz',d/'warp.json')
    except ValueError:
        pass
    else:
        raise AssertionError('Folded brain warp was accepted')
    nuisance = np.column_stack([1e6 + np.arange(10), np.full(10, 7.0)])
    nuisance_path = d / 'nuisance.1D'
    np.savetxt(nuisance_path, nuisance)
    standardized = d / 'standardized.1D'
    prepare_regressors(standardized, [nuisance_path])
    actual = np.loadtxt(standardized)
    assert abs(actual.mean()) < 1e-12
    assert abs(np.linalg.norm(actual) - 1) < 1e-11
    metadata = json.loads(standardized.with_suffix('.json').read_text())
    assert metadata['constant_columns'] == ['nuisance.1D:1']
    assert np.allclose(actual * metadata['centered_l2_norms'][0] + metadata['original_means'][0], nuisance[:, 0])
print('KNOWN_ANSWER_HELPER_CHECKS_OK')
