#!/usr/bin/env python3
"""Validate run timing before processing; never infer a slice order from vendor."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

def prepare(bold, output, drop=0, stc='required'):
    image = nib.load(bold)
    sidecar = Path(str(bold).removesuffix('.gz')).with_suffix('.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8-sig'))
    tr = float(metadata['RepetitionTime'])
    if len(image.shape) != 4 or not np.isfinite(tr) or tr <= 0:
        raise ValueError('Expected 4D BOLD and a positive TR in seconds')
    if not np.isclose(tr, image.header.get_zooms()[3], rtol=1e-4, atol=1e-4):
        raise ValueError('JSON and NIfTI TR disagree')
    if not 0 <= drop < image.shape[3]-5:
        raise ValueError('DROP_VOLUMES must leave at least six volumes')
    offsets = metadata.get('SliceTiming')
    direction = metadata.get('SliceEncodingDirection', 'k')
    if stc != 'off':
        if direction not in ('k', 'k-'):
            raise ValueError('This AFNI entry point requires slice axis k; do not silently reorder metadata')
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (image.shape[2],) or not np.isfinite(offsets).all():
            raise ValueError('A verified SliceTiming value is required for every slice')
        if np.any(offsets < 0) or np.any(offsets >= tr):
            raise ValueError('SliceTiming must lie within [0, TR) seconds')
        if direction == 'k-':
            offsets = offsets[::-1]
        np.savetxt(output/'slice_timing.1D', offsets[None], fmt='%.10g')
    reference = None if stc == 'off' else float((offsets.min()+offsets.max())/2)
    result = dict(source=str(Path(bold).resolve()), TR_s=tr, EchoTime_s=metadata.get('EchoTime'), original_volumes=image.shape[3],
                  retained_volumes=image.shape[3]-drop, dropped_volumes=drop,
                  slice_axis=direction, stc_applied=stc != 'off', reference_time_s=reference,
                  timing_source=metadata.get('SliceTimingSource',str(sidecar)),
                  geometry='Original NIfTI affine retained; no header-only deobliquing')
    (output/'preprocessing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(tr, reference if reference is not None else 0, sep='\t')

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('bold',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--drop',type=int,default=0)
    p.add_argument('--stc',choices=['required','off'],default='required')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    prepare(a.bold,a.output,a.drop,a.stc)
