import os 
os.system('wget https://github.com/June103310110/Image_Segmentation/releases/download/datasets/requirements_chaos.txt')
os.system('wget https://raw.githubusercontent.com/June103310110/Image_Segmentation/main/03_Models/loss.py')
os.system('wget https://raw.githubusercontent.com/June103310110/Image_Segmentation/main/03_Models/unet.py')
os.system('pip3 install -q -r ./requirements_chaos.txt')
os.system('pip3 install --upgrade gdown')

import gdown 
'''
If dataset is unavailable, plz contact me: june103310110@gmail.com.
'''
# download dataset 
url = "https://drive.google.com/u/0/uc?id=1r1gRIZt7V4rNMzGvZXjQ1mLrJ0j5l-7i&export=download"
output = "full_data.zip"
gdown.download(url, output)

os.system('mkdir data')
os.system(f'unzip -o {output} -d ./data')
os.system(f'rm -f {output}')

