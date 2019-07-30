
# Create list of executable commands
python3 SPAM/python/cmd_zooRun.py \
  -o Input_Data/spam_input/cmd_spam_test.txt \
  -zooLoc Input_Data/spam_input/zoo_all_models.txt \
  -nPart 10000 \
  -n 5

# Test executable commands
python3 Useful_Bin/batch_execution.py \
  -i Input_Data/spam_input/cmd_spam_test.txt \
  -uID \
  -pp 6

