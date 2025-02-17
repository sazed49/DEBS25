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





