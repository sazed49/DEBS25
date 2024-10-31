dataset="mnist"
momentum=0.9
weight_decay=1e-4
global_rounds=10
local_rounds=4
max_std_unreliable=30



sign_flip_clienst=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13")

additive_noise_clients=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28,30")
unreliable_clients=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37")
declare -a algorithms=("mudhog" "my_hog"  "fedavg" "median" "gm" "krum" "mkrum" "foolsgold" )
experiment_name="sExp4Ba_40ep"
for ((i=0; i < 1; i++));
do
for ((j=2; j<3; j++));
do
CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --AR ${algorithms[$i]} --loader_type iid --experiment_name ${experiment_name}_${dataset}_${algorithms[$i]}_40C_$((j+3) --epochs $global_rounds --num_clients 40 --inner_epochs $local_rounds --list_unreliable ${unreliable_clients[$j]} --list_uatk_flip_sign ${sign_flip_clienst[$j]} --list_uatk_add_noise ${additive_noise_clients[$j]}   --lr $learning_rate --weight_decay $weight_decay --momentum $momentum --verbose  --max_std_unreliable $max_std_unreliable
done
done