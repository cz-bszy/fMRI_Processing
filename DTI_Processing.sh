#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <DataFolder> <nThreads> <Atlas_PATH> <Output_Dir>"
    exit 1
fi

data_dir=$1
nthreads=$2
src_dir=$3
subj_dir=$4

schaefer_dir=${src_dir}/schaefer
Glasser_dir=${src_dir}/Glasser_parcellations
HCPpipeline_dir=${src_dir}/HCPpipelines-4.7.0
mrtrixe3_dir=/opt/mrtrix3/share/mrtrix3/labelconvert

subjects=($(find $data_dir -maxdepth 1 -mindepth 1 -type d -exec basename {} \;))

for subj in "${subjects[@]}"
do
    echo "Processing subject $subj"

    sessions=($(find $data_dir/$subj -maxdepth 1 -mindepth 1 -type d -exec basename {} \;))
    if [ ${#sessions[@]} -eq 0 ]; then
        # No sessions, process anat and dwi directly under subject folder
        echo "No sessions found for subject $subj, processing anat and dwi directly"
        diff_dir=$data_dir/$subj/dwi
        anat_dir=$data_dir/$subj/anat

        [[ ! -e $subj_dir/$subj/structural_connectivity ]] && mkdir -p $subj_dir/$subj/structural_connectivity
        output_dir=$subj_dir/$subj/structural_connectivity

        echo -e "Processing subject $subj on $(date)"

        mrconvert $diff_dir/*_dwi.nii.gz $output_dir/DWI.mif -fslgrad $diff_dir/*_dwi.bvec $diff_dir/*_dwi.bval -datatype float32 -strides 0,0,0,1 -quiet

        dwiextract $output_dir/DWI.mif - -bzero | mrmath - mean $output_dir/meanb0.mif -axis 3

        dwidenoise $output_dir/DWI.mif $output_dir/dwi_denoised.mif -nthreads $nthreads

        mrdegibbs $output_dir/dwi_denoised.mif $output_dir/dwi_denoised_unringed.mif

        dwifslpreproc $output_dir/dwi_denoised_unringed.mif $output_dir/dwi_denoised_unringed_preproc.mif -rpe_none -pe_dir AP -eddy_options " --slm=linear --data_is_shelled " -nthreads $nthreads

        dwi2mask $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/dwi_temp_mask.mif

        dwi2response tournier $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/wm_response.txt -voxels $output_dir/voxels.mif -force
        dwi2fod csd -mask $output_dir/dwi_temp_mask.mif $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/wm_response.txt $output_dir/fod.mif -nthreads $nthreads

        [[ ! -e $subj_dir/$subj/recon_all ]] && mkdir -p $subj_dir/$subj/recon_all
        export SUBJECTS_DIR=$subj_dir/$subj/recon_all

        recon-all -s $subj -i $anat_dir/*_T1w.nii.gz -all -openmp $nthreads

        echo -e "Diffusion image processing completed on $(date)"

        mri_convert $SUBJECTS_DIR/$subj/mri/brain.mgz $output_dir/T1w_brain.nii.gz
        5ttgen fsl $output_dir/T1w_brain.nii.gz $output_dir/5TT_in_T1w_space.mif -premasked -force

        tckgen $output_dir/fod.mif $output_dir/tracks_10M.tck -algorithm iFOD2 -act $output_dir/5TT_in_T1w_space.mif -backtrack -crop_at_gmwmi -seed_dynamic $output_dir/fod.mif -maxlength 300 -select 10M -cutoff 0.06 -nthreads $nthreads -force

        schaefer_atlas="Schaefer2018_100Parcels_7Networks_order"
        for hemi in lh rh;
        do
            mri_surf2surf --hemi $hemi --srcsubject ../../../../opt/freesurfer/freesurfer/subjects/fsaverage5 --trgsubject $subj --sval-annot $schaefer_dir/FreeSurfer5.3/fsaverage5/label/${hemi}.${schaefer_atlas}.annot --tval $output_dir/${hemi}.${schaefer_atlas}.annot
        done
        [[ ! -e $subj_dir/$subj/mri ]] && mkdir -p $subj_dir/$subj/mri
        [[ ! -e $subj_dir/$subj/surf ]] && mkdir -p $subj_dir/$subj/surf
        [[ ! -e $subj_dir/$subj/label ]] && mkdir -p $subj_dir/$subj/label

        mv $output_dir/*h.${schaefer_atlas}.annot $subj_dir/$subj/label/

        cp -f $SUBJECTS_DIR/$subj/surf/*h.pial $SUBJECTS_DIR/$subj/surf/*h.white $subj_dir/$subj/surf/
        cp -f $SUBJECTS_DIR/$subj/mri/ribbon.mgz $SUBJECTS_DIR/$subj/mri/aseg.mgz $subj_dir/$subj/mri/

        export SUBJECTS_DIR=$subj_dir

        mri_aparc2aseg --s $subj --o $output_dir/${schaefer_atlas}.mgz --annot $schaefer_atlas

        labelconvert $output_dir/${schaefer_atlas}.mgz $schaefer_dir/project_to_individual/${schaefer_atlas}_LUT.txt $schaefer_dir/freeview_lut/${schaefer_atlas}.txt $output_dir/${schaefer_atlas}_parcels.mif -force

        tck2connectome -symmetric -zero_diagonal -scale_invnodevol $output_dir/tracks_10M.tck $output_dir/${schaefer_atlas}_parcels.mif $output_dir/${schaefer_atlas}_connectome.csv -out_assignment $output_dir/${schaefer_atlas}_connectome_assignments.csv -nthreads $nthreads -force

    else
        # Sessions found, process each session
        for sess in "${sessions[@]}"
        do
            echo "Processing session $sess for subject $subj"

            diff_dir=$data_dir/$subj/$sess/dwi
            anat_dir=$data_dir/$subj/$sess/anat

            [[ ! -e $subj_dir/$subj/$sess/structural_connectivity ]] && mkdir -p $subj_dir/$subj/$sess/structural_connectivity
            output_dir=$subj_dir/$subj/$sess/structural_connectivity

            echo -e "Processing subject $subj, session $sess on $(date)"

            mrconvert $diff_dir/*_dwi.nii.gz $output_dir/DWI.mif -fslgrad $diff_dir/*_dwi.bvec $diff_dir/*_dwi.bval -datatype float32 -strides 0,0,0,1 -quiet

            dwiextract $output_dir/DWI.mif - -bzero | mrmath - mean $output_dir/meanb0.mif -axis 3

            dwidenoise $output_dir/DWI.mif $output_dir/dwi_denoised.mif -nthreads $nthreads

            mrdegibbs $output_dir/dwi_denoised.mif $output_dir/dwi_denoised_unringed.mif

            dwifslpreproc $output_dir/dwi_denoised_unringed.mif $output_dir/dwi_denoised_unringed_preproc.mif -rpe_none -pe_dir AP -eddy_options " --slm=linear --data_is_shelled " -nthreads $nthreads

            dwi2mask $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/dwi_temp_mask.mif

            dwi2response tournier $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/wm_response.txt -voxels $output_dir/voxels.mif -force
            dwi2fod csd -mask $output_dir/dwi_temp_mask.mif $output_dir/dwi_denoised_unringed_preproc.mif $output_dir/wm_response.txt $output_dir/fod.mif -nthreads $nthreads

            [[ ! -e $subj_dir/$subj/$sess/recon_all ]] && mkdir -p $subj_dir/$subj/$sess/recon_all
            export SUBJECTS_DIR=$subj_dir/$subj/$sess/recon_all

            recon-all -s ${subj}_${sess} -i $anat_dir/*_T1w.nii.gz -all -openmp $nthreads

            echo -e "Diffusion image processing completed on $(date)"

            mri_convert $SUBJECTS_DIR/${subj}_${sess}/mri/brain.mgz $output_dir/T1w_brain.nii.gz
            5ttgen fsl $output_dir/T1w_brain.nii.gz $output_dir/5TT_in_T1w_space.mif -premasked -force

            tckgen $output_dir/fod.mif $output_dir/tracks_10M.tck -algorithm iFOD2 -act $output_dir/5TT_in_T1w_space.mif -backtrack -crop_at_gmwmi -seed_dynamic $output_dir/fod.mif -maxlength 300 -select 10M -cutoff 0.06 -nthreads $nthreads -force

            schaefer_atlas="Schaefer2018_100Parcels_7Networks_order"
            for hemi in lh rh;
            do
                mri_surf2surf --hemi $hemi --srcsubject ../../../../opt/freesurfer/freesurfer/subjects/fsaverage5 --trgsubject ${subj}_${sess} --sval-annot $schaefer_dir/FreeSurfer5.3/fsaverage5/label/${hemi}.${schaefer_atlas}.annot --tval $output_dir/${hemi}.${schaefer_atlas}.annot
            done
            [[ ! -e $subj_dir/$subj/$sess/mri ]] && mkdir -p $subj_dir/$subj/$sess/mri
            [[ ! -e $subj_dir/$subj/$sess/surf ]] && mkdir -p $subj_dir/$subj/$sess/surf
            [[ ! -e $subj_dir/$subj/$sess/label ]] && mkdir -p $subj_dir/$subj/$sess/label

            mv $output_dir/*h.${schaefer_atlas}.annot $subj_dir/$subj/$sess/label/

            cp -f $SUBJECTS_DIR/${subj}_${sess}/surf/*h.pial $SUBJECTS_DIR/${subj}_${sess}/surf/*h.white $subj_dir/$subj/$sess/surf/
            cp -f $SUBJECTS_DIR/${subj}_${sess}/mri/ribbon.mgz $SUBJECTS_DIR/${subj}_${sess}/mri/aseg.mgz $subj_dir/$subj/$sess/mri/

            export SUBJECTS_DIR=$subj_dir

            mri_aparc2aseg --s ${subj}_${sess} --o $output_dir/${schaefer_atlas}.mgz --annot $schaefer_atlas

            labelconvert $output_dir/${schaefer_atlas}.mgz $schaefer_dir/project_to_individual/${schaefer_atlas}_LUT.txt $schaefer_dir/freeview_lut/${schaefer_atlas}.txt $output_dir/${schaefer_atlas}_parcels.mif -force

            tck2connectome -symmetric -zero_diagonal -scale_invnodevol $output_dir/tracks_10M.tck $output_dir/${schaefer_atlas}_parcels.mif $output_dir/${schaefer_atlas}_connectome.csv -out_assignment $output_dir/${schaefer_atlas}_connectome_assignments.csv -nthreads $nthreads -force

        done
    fi

done
