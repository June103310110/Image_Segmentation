import os 
os.system('pip3 install -r ./requirements.txt')



import gdown 
'''
If dataset is unavailable, plz contact me: june103310110@gmail.com.
'''
# download dataset 
os.system('pip install --upgrade gdown')
url = "https://drive.google.com/u/1/uc?id=15EVEIB2o-sJJd0qOjoFbZrz4briIYoeF&export=download"
output = "full_data.zip"
os.remove('full_data.zip')
gdown.download(url, output)
os.system('unzip -o full_data.zip -d ./')

