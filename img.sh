
python3 Image_Creator/python/cmd_image_creator_v2.py \
  -sdssDir ~/testData/588017702948962343/ \
  -writeLoc Input_Data/image_creator/cmd_v2_test.txt \
  -n 1

python3 Useful_Bin/batch_execution.py \
  -i Input_Data/image_creator/cmd_v2_test.txt \
  -pp 1
