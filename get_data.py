import os

if not os.path.exists('data'):
    !mkdir data
    
!cd data
# Images:
# traincrop.zip
!gdown 1jsCzT4uDkLHel8ov1CfbvLXiQVy6KExT
# testcrop.zip
!gdown 1NwyJjbhEATg-5x8-hQqet3KuiIZOhXZu
# valcrop.zip
!gdown 1DboG3vmTjbZyzrUQSyqToZw_xCOaeNoE
# Download .txt files describing datasets:
# train.txt
!gdown 1f6kco4VRf47bBp0qtb57sePvoqA_xFZa
# test.txt
!gdown 1He8W5Zf32rrA-pCAopfV-MrX94jKU4nR
# val.txt
!gdown 1f4FG4pmB7bcRLeSGtNVCpjgAaMD5mrdm

!unzip -q traincrop.zip
!unzip -q testcrop.zip
!unzip -q valcrop.zip
!ls