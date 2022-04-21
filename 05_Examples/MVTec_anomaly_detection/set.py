import os 
os.system('pip3 install -r ./requirements.txt')
os.system('pip3 install -U gdown')
import gdown 
'''
If dataset is unavailable, plz contact me: june103310110@gmail.com.
'''
# download dataset 
url = "https://drive.google.com/u/0/uc?id=1LwVWWpNRmSZAXD3M3bVCwrIfSEKe4zol&export=download"
output = "MVtech-capsule.tar.xz"
gdown.download(url, output)

os.system('mkdir data')
os.system(f'unzip -o {output} -d ./data')
os.system(f'rm -f {output}')