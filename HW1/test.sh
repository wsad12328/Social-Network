#!/bin/bash

# 檢查是否有輸入兩個參數
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <test_graphs_min> <test_graphs_max>"
  exit 1
fi

MIN=$1
MAX=$2

# python src/test.py --num_min 100 --num_max 200
# python src/test.py --num_min 200 --num_max 300
# python src/test.py --num_min 1000 --num_max 1200
# python src/test.py --num_min 2000 --num_max 3000
# python src/test.py --num_min 4000 --num_max 5000 --test_graphs_min $MIN --test_graphs_max $MAX 

# python src/test.py --num_min 4000 --num_max 5000 --test_graphs_min $MIN --test_graphs_max $MAX --test_data_path ../DrBC_Data/Synthetic

python src/test.py --num_min 4000 --num_max 5000 --test_data_path ../DrBC_Data/Real/ --test_graphs_min $MIN --test_graphs_max $MAX
