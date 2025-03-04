for setting in normal scarcity noise imbalance
do
    for model_name in gemma-2-9b llama-3.1-8b gemma-2-2b
    do
        for i in {1..4}
        do
            OMP_NUM_THREADS=1 python3 train_sae_probes.py --script_type run_normal_baselines --setting $setting --model_name $model_name --device cuda:$i --reg_type l1 &
        done
    done
done
