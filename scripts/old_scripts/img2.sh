
python3 Image_Creator/python/cmd_image_creator_v2.py \
  -sdssDir ~/localstorage/587722984435351614/ \
  -writeLoc Input_Data/image_creator/cmd_v2_qual1.txt \
  -paramLoc Input_Data/image_parameters/param_1.txt \
  -noprint


python3 Useful_Bin/batch_execution.py \
  -i Input_Data/image_creator/cmd_v2_qual1.txt \
  -pp 7


#  -sdssDir ~/testData/587722984435351614/ \

