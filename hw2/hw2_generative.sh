#!/bin/bash 
# bash hw2_generative.sh datad/train.csv datad/test.csv featured/X_train featured/Y_train featured/X_test test_dir/ans_gen.csv
python generative.py --train --train_data_path $3 --train_label_path $4 --test_data_path $5 
python generative.py --infer --train_data_path $3 --train_label_path $4 --test_data_path $5 --output_path $6