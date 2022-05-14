pause(){
  read -p "Press [Enter] key to continue..." fackEnterKey
}

Imagenet(){
 echo "Imagenet Init"
#  imagenet-camera --camera /dev/video0 --width=640 --height=480
 pause
}

DownloadData(){
echo "download CHAOS dataset"
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/set.py
python set.py
rm set.py
pause
}

EnvSetting(){
echo "Enviroment setting, download utils"
pip install -q -U albumentations
pip install -q -U pydicom
pip install -q -U pytorch-lightning
mkdir utils
echo "download utils..."
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/loss.py -O utils/loss.py
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/train_unet.py -O utils/train_unet.py
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/unet.py -O utils/unet.py
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/dataset.py -O utils/dataset.py
wget -q https://github.com/June103310110/Image_Segmentation/releases/download/AIA_course/DG_baseline.py -O utils/DG_baseline.py
echo "finished"
pause
}


# function to display menus
show_menus() {
	clear
	echo "~~~~~~~~~~~~~~~~~~~~~"	
	echo "影像分割專題實作"
	echo "~~~~~~~~~~~~~~~~~~~~~"
	echo "a. 實作開始"
	echo "=======前置作業====="
	echo "1. 環境建置並下載關聯文件       2. 下載並解壓資料集"
	echo "q. Exit離開"
}

# read input from the keyboard and take a action
# Exit when user the user select 3 form the menu option.
read_options(){
	local choice
	read -p "Enter choice [ ] " choice
	case $choice in
		1) EnvSetting ;;
		2) DownloadData ;;
		q) exit 0;;
		*) echo -e "${RED}Error...${STD}" && sleep 2
	esac
}

# ----------------------------------------------
# Step #3: Trap CTRL+C, CTRL+Z and quit singles
# ----------------------------------------------
trap '' SIGINT SIGQUIT SIGTSTP
 
# -----------------------------------
# Step #4: Main logic - infinite loop
# ------------------------------------
while true
do
	show_menus
	read_options
done