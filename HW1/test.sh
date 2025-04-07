#!/bin/bash

# 檢查是否有輸入兩個參數
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <test_graphs_min> <test_graphs_max>"
  exit 1
fi

MIN=$1
MAX=$2

epoch=(50 100 150 200)

for i in "${epoch[@]}"
do
  python src/test.py --epochs $i --test_graphs_min $MIN --test_graphs_max $MAX 
done
