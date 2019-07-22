#python3 Image_Creator/python/image_creator_v2.py \
#  -runDir ~/testData/588017702948962343/run_00_0001/

python3 Image_Creator/python/cmd_image_creator_v2.py \
  -sdssDir ~/testData/588017702948962343/ \
  -writeLoc Input_Data/image_creator/cmd_v2_test.txt \
  -n 6

python3 Useful_Bin/batch_execution.py \
  -i Input_Data/image_creator/cmd_v2_test.txt \
  -pp 3
