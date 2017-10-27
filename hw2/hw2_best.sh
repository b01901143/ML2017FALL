#!/bin/bash 
# bash hw2_best.sh datad/train.csv datad/test.csv featured/X_train featured/Y_train featured/X_test test_dir/ans_nn.csv
python best.py --train --train_data_path $3 --train_label_path $4 --test_data_path $5 
python best.py --infer --train_data_path $3 --train_label_path $4 --test_data_path $5 --output_path $6