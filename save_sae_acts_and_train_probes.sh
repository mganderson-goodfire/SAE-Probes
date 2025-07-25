# ---- Generate SAE activations ----

for i in {1..100}
do
    python3 generate_sae_activations.py --model_name gemma-2-9b --setting normal --device mps --randomize_order
done


for i in {1..10}
do
    python3 generate_sae_activations.py --model_name gemma-2-9b --setting scarcity --device mps --randomize_order
done

for i in {1..10}
do
    python3 generate_sae_activations.py --model_name gemma-2-9b --setting imbalance --device mps --randomize_order
done


for i in {1..10}
do
    python3 generate_sae_activations.py --model_name gemma-2-9b --setting OOD --device mps --randomize_order
done



# # ---- Train SAE probes ----

for i in {1..20}
do
    OMP_NUM_THREADS=1 python3 train_sae_probes.py --model_name gemma-2-9b --setting normal --reg_type l1 --randomize_order &
done

wait

for i in {1..20}
do
    OMP_NUM_THREADS=1 python3 train_sae_probes.py --model_name gemma-2-9b --setting scarcity --reg_type l1 --randomize_order &
done

wait


for i in {1..20}
do
    OMP_NUM_THREADS=1 python3 train_sae_probes.py --model_name gemma-2-9b --setting class_imbalance --reg_type l1 --randomize_order &
done

wait


for i in {1..20}
do
    OMP_NUM_THREADS=1 python train_sae_probes.py --model_name gemma-2-9b --setting noise --reg_type l1 --randomize_order &
done

wait


for i in {1..20}
do
    OMP_NUM_THREADS=1 python3 train_sae_probes.py --model_name gemma-2-9b --setting OOD --reg_type l1 --randomize_order &
done

wait
