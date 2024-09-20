
echo "执行的脚本$0"
project="pitsc"
GPU=2
mkdir -p logs/$0

dataset="nsynth-100"
python train.py -project $project -dataroot /data/datasets/The_NSynth_Dataset \
-dataset $dataset -config configs/pitsc/pitsc_nsynth-100_stochastic_classifier_t10.yaml -gpu $GPU \
>> logs/$0/${dataset}.log 2>&1
:"
dataset="FMC"
python train.py -project $project -dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
-dataset $dataset -config configs/pitsc/pitsc_FMC_stochastic_classifier_t10.yaml -gpu $GPU \
>> logs/$0/${dataset}.log 2>&1

dataset="librispeech"
python train.py -project $project -dataroot /data/datasets/librispeech_fscil/100spks_segments \
-dataset $dataset -config configs/pitsc/pitsc_LS-100_stochastic_classifier_t10.yaml -gpu $GPU \
>> logs/$0/${dataset}.log 2>&1
"





