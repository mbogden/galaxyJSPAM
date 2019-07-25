#python3 Comparison_Methods/pixel_comparison/python/exec_pixel_comp.py -thisSdssDir ~/testData/587722984435351614/ -writeLoc here.txt -zooFile Input_Data/zoo_models/587722984435351614.txt

#python3 Useful_Bin/batch_execution.py -i temp.txt -pp 1
#python3 Useful_Bin/batch_execution.py -i Input_Data/comparison_methods/exec_laptop.txt -pp 3


# Try 2
# Testing if compare works for 1
'''
python3 Comparison_Methods/compare.py \
  -runDir ~/testData/587722984435351614/run_00_0001/ \
  -argFile Input_Data/comparison_methods/arg_compare.txt
'''

# Create list of executable commands
python3 Comparison_Methods/cmd_compare_maker.py \
  -writeLoc all_cmd.txt \
  -sdssDir ~/testData/587722984435351614/ \
  -zooFile Input_Data/zoo_models/587722984435351614.txt \

# Test executable commands
python3 Useful_Bin/batch_execution.py -i all_cmd.txt -pp 3

python3 Score_Analysis/python/score_analysis.py -plotDir temp_plot_dir -sdssDir ~/testData/587722984435351614/ -basicPlots
