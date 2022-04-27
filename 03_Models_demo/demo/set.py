import os 
os.system('pip3 install -q -r ./requirements.txt')
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

