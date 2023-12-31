python main.py --net ResNet --set_seed 777 ./results  --imgs 500,5000,1600 --resolution 500  --epochs 10 --lr 0.01

python main.py --net <model_name> --set_seed <init_seed> --save_net <model_save_path> --imgs 500,5000,1600 --resolution 500 --active_log --epochs <number_epochs> --lr <suitable_learningrate>


python main.py --net resnet --set_seed 997 --save_net saves --imgs 500,5000,1600 --resolution 500 --active_log --epochs 2 --lr 0.01

python main.py --load_net /home/juanma/TRDP/OODrepo/dbViz/pretrained_models/resnet18-f37072fd.pth --net resnet --set_seed 666 --save_net saves --imgs 500,5000,1600 --resolution 500 --active_log --epochs 2 --lr 0.01



/media/jpenatrapero/TAU/TRDP/OODrepo/dbViz/pretrained_models


python main.py --load_net /media/jpenatrapero/TAU/TRDP/OODrepo/dbViz/pretrained_models
/resnet18-f37072fd.pth --net resnet --set_seed 666 --save_net saves --imgs 500,5000,1600 --resolution 500 --active_log --epochs 2 --lr 0.01
