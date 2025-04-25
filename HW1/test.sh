#!/bin/bash

# 資料集與對應的資料夾
declare -A datasets=(
    ["5000"]="../DrBC_Data/Synthetic"
    ["10000"]="../DrBC_Data/Synthetic"
    ["20000"]="../DrBC_Data/Synthetic"
    ["50000"]="../DrBC_Data/Synthetic"
    ["100000"]="../DrBC_Data/Synthetic"
    ["com-youtube"]="../DrBC_Data/Real"
    ['amazon']="../DrBC_Data/Real"
    ["dblp"]="../DrBC_Data/Real"
    ["com-lj"]="../DrBC_Data/Real"
)

dims=(64 128 256)
layers=(3 4 5)
aggregate_types=("sum" "GCN")

for dataset in "${!datasets[@]}"; do
    path=${datasets[$dataset]}
    for dim in "${dims[@]}"; do
        for layer in "${layers[@]}"; do
            for agg in "${aggregate_types[@]}"; do
                echo "Running: $dataset | dim=$dim | layers=$layer | aggregate_type=$agg"
                python src/test.py --test_data_path "$path" --test_dataset "$dataset" --dim "$dim" --layers "$layer" --aggregate_type "$agg"
            done
        done
    done
done
