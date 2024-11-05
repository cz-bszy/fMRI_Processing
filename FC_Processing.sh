#!/bin/bash

parallel_jobs=8

working_dir="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/ABIDEII-UPSM_Long/" 
tissuepriors_dir="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/tissuepriors"
standard_3mm="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/standard"
template_dir="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/template"
ls -d $working_dir/* | xargs -n 1 basename > ./subjects.txt
subjects_file="./subjects.txt"

num_threads=4
FWHM=6
sigma=2.54798709
highp=0.1
lowp=0.005
TR=1.5
TE=25
vol=200
fsf_type="Retain_GRS"

mkdir -p "./log"

run_steps() {
  subj=$1
  echo "Processing subject: $subj ..."
  
  log_file="./log/${subj}_log.txt"
  
  {
    echo "===== Running FC_step1 for subject: $subj ====="
    ./FC_step1 $subj $num_threads $working_dir
    echo "===== Completed FC_step1 for subject: $subj ====="
    
    echo "===== Running FC_step2 for subject: $subj ====="
    ./FC_step2 $subj $num_threads $working_dir $FWHM $sigma $highp $lowp 
    echo "===== Completed FC_step2 for subject: $subj ====="
    
    echo "===== Running FC_step3 for subject: $subj ====="
    ./FC_step3 $subj $num_threads $working_dir $standard_3mm
    echo "===== Completed FC_step3 for subject: $subj ====="
    
    echo "===== Running FC_step4 for subject: $subj ====="
    ./FC_step4 $subj $num_threads $working_dir $sigma $tissuepriors_dir
    echo "===== Completed FC_step4 for subject: $subj ====="
    
    echo "===== Running FC_step5 for subject: $subj ====="
    ./FC_step5 $subj $num_threads $working_dir
    echo "===== Completed FC_step5 for subject: $subj ====="
    
    echo "===== Running FC_step6 for subject: $subj ====="
    bash ./FC_step6 $subj $num_threads $working_dir $template_dir $TR $TE $vol $fsf_type
    echo "===== Completed FC_step6 for subject: $subj ====="

  } &> $log_file
}

export -f run_steps

export working_dir
export tissuepriors_dir
export template_dir
export num_threads
export FWHM
export sigma
export highp
export lowp
export TR
export TE
export vol
export fsf_type
export standard_3mm

cat $subjects_file | parallel -j $parallel_jobs run_steps

bash FC_step7 $subj $working_dir $fsf_type
