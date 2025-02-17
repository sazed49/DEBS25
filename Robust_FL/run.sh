# dataset="rockburst"
# learning_rate=1e-2
# momentum=0.9
# weight_decay=1e-4
# global_rounds=20
# local_rounds=60

# device="cpu"
# # batch mnist=32. rockburst=8
# #local rounds>- MNIST 4, Rockburst 60

# sign_flip_clienst=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13")

# additive_noise_clients=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28,30")

# unreliable_clients=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37")

# declare -a algorithms=("detect_and_aggregate"  "sign_flip"   "unreliable_clients"  "new_detect_additive_noise"   "mudhog"                       "minedetect"     "mkrum" "krum"  "fedavg" "gm" "foolsgold"                                                                                                 "mkrum" "krum"                                               "SIMPLE"  "median")
# experiment_name="sExp4Ba_40ep"
# # for ((i=0; i < 1; i++));
# # do
# # for ((j=3; j<4; j++));
# # do
# # CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/ICDCS/Writing_All_Fresh/ICDCS25/Robust_FL/main.py --device $device  --dataset $dataset --AR ${algorithms[$i]} --loader_type dirichlet  --experiment_name ${experiment_name}_${dataset}_${algorithms[$i]}_40C_$((j+3)) --epochs $global_rounds --num_clients 40 --inner_epochs $local_rounds --list_unreliable ${unreliable_clients[$j]}  --list_uatk_flip_sign ${sign_flip_clienst[$j]} --list_uatk_add_noise ${additive_noise_clients[$j]}   --lr $learning_rate --weight_decay $weight_decay --momentum $momentum --verbose   
# # done
# # done

# # for ((i=0; i < 1; i++));
# # do
# # for ((j=3; j<4; j++));
# # do
# #     # Default values
# #     max_std_unreliable=50  
# #     std_add_noise_=0.3  

# #     # Set different values based on algorithm
# #     if [[ "${algorithms[$i]}" == "unreliable_clients" ]]; then
# #         max_std_unreliable=30
# #         std_add_noise=0.01
# #     elif [[ "${algorithms[$i]}" == "new_detect_additive_noise" ]]; then
# #         max_std_unreliable=10
# #         std_add_noise=0.3
# #     elif [[ "${algorithms[$i]}" == "sign_flip" ]]; then
# #         max_std_unreliable=10
# #         std_add_noise=0.3
# #     fi

# #     CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/ICDCS/Writing_All_Fresh/ICDCS25/Robust_FL/main.py \
# #         --device $device \
# #         --dataset $dataset \
# #         --AR ${algorithms[$i]} \
# #         --loader_type dirichlet \
# #         --experiment_name ${experiment_name}_${dataset}_${algorithms[$i]}_40C_$((j+3)) \
# #         --epochs $global_rounds \
# #         --num_clients 40 \
# #         --inner_epochs $local_rounds \
# #         --list_unreliable ${unreliable_clients[$j]} \
# #         --list_uatk_flip_sign ${sign_flip_clienst[$j]} \
# #         --list_uatk_add_noise ${additive_noise_clients[$j]} \
# #         --lr $learning_rate \
# #         --weight_decay $weight_decay \
# #         --momentum $momentum \
# #         --max_std_unreliable $max_std_unreliable \
# #         --std_add_noise $std_add_noise \
# #         --verbose   
# # done
# # done




# for ((i=0; i < 1; i++));
# do
# for ((j=3; j<4; j++));
# do
#     # Default values
#     max_std_unreliable=30  
#     std_add_noise=0.01 

#     # Adjust values dynamically for detect_and_aggregate
#     if [[ "${algorithms[$i]}" == "detect_and_aggregate" ]]; then
#         if [[ "$j" == "3" ]]; then
#             max_std_unreliable=20
#             std_add_noise=0.2
#         elif [[ "$j" == "4" ]]; then
#             max_std_unreliable=25
#             std_add_noise=0.15
#         else
#             max_std_unreliable=30
#             std_add_noise=0.1
#         fi
#     fi

#     CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/ICDCS/Writing_All_Fresh/ICDCS25/Robust_FL/main.py \
#         --device $device \
#         --dataset $dataset \
#         --AR ${algorithms[$i]} \
#         --loader_type dirichlet \
#         --experiment_name ${experiment_name}_${dataset}_${algorithms[$i]}_40C_$((j+3)) \
#         --epochs $global_rounds \
#         --num_clients 40 \
#         --inner_epochs $local_rounds \
#         --list_unreliable ${unreliable_clients[$j]} \
#         --list_uatk_flip_sign ${sign_flip_clienst[$j]} \
#         --list_uatk_add_noise ${additive_noise_clients[$j]} \
#         --lr $learning_rate \
#         --weight_decay $weight_decay \
#         --momentum $momentum \
#         --max_std_unreliable $max_std_unreliable \
#         --std_add_noise $std_add_noise \
#         --verbose   
# done
# done


dataset="fashion_mnist"
learning_rate=1e-2
momentum=0.9
weight_decay=1e-4
global_rounds=20
local_rounds=4


device="cuda"

sign_flip_clients=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13")
additive_noise_clients=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28")
unreliable_clients=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37,15")

declare -a algorithms=("minedetect" "sign_flip" "unreliable_clients" "new_detect_additive_noise" "mudhog" "mkrum" "krum" "fedavg" "gm" "foolsgold" "median")

experiment_name="sExp4Ba_40ep"

# Display available algorithms with numbers
echo "Select an algorithm by entering the corresponding number:"
for i in "${!algorithms[@]}"; do
    echo "$((i+1)). ${algorithms[$i]}"
done

# Get user input
read -p "Enter the number of the algorithm you want to run: " selected_number

# Validate user input (must be a number within range)
if ! [[ "$selected_number" =~ ^[0-9]+$ ]] || (( selected_number < 1 || selected_number > ${#algorithms[@]} )); then
    echo "Error: Invalid selection!"
    exit 1
fi

# Get the selected algorithm name
selected_algorithm="${algorithms[$((selected_number-1))]}"

echo "You selected: $selected_algorithm"

for ((j=5; j<6; j++)); do
    CUDA_VISIBLE_DEVICES=0 python /content/drive/MyDrive/ICDCS/Writing_All_Fresh/ICDCS25/Robust_FL/main.py \
        --device $device \
        --dataset $dataset \
        --AR ${selected_algorithm} \
        --loader_type dirichlet \
        --experiment_name ${experiment_name}_${dataset}_${selected_algorithm}_40C_$((j+3)) \
        --epochs $global_rounds \
        --num_clients 40 \
        --inner_epochs $local_rounds \
        --list_unreliable ${unreliable_clients[$j]} \
        --list_uatk_flip_sign ${sign_flip_clients[$j]} \
        --list_uatk_add_noise ${additive_noise_clients[$j]} \
        --lr $learning_rate \
        --weight_decay $weight_decay \
        --momentum $momentum \
        --verbose   
done





