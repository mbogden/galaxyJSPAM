
# Create list of executable commands
python3 Comparison_Methods/cmd_compare_maker.py \
  -writeLoc Input_Data/comparison_methods/cmd_compare_test.txt \
  -sdssDir ~/testData/587722984435351614/ \
  -zooFile Input_Data/zoo_models/587722984435351614.txt \
  -n 1

# Test executable commands
python3 Useful_Bin/batch_execution.py \
  -i Input_Data/comparison_methods/cmd_compare_test.txt \
  -pp 1

#python3 Score_Analysis/python/score_analysis.py -plotDir temp_plot_dir -sdssDir ~/testData/587722984435351614/ -basicPlots
