from utils import *
from TripletExtractor import Triplet
from PlanesetExtractor import Planeset
from InfoDBExtractor import PlanesetInfoExtractor
import numpy as np

import argparse
import sys
sys.path.append("./")

import utils2 
import models.builer as builder



def load_latent_example(config, idx):
    # load a planest object 
    planeset_obj = load_planeset_example(config)
    input_tensor = planeset_obj.planeset[idx]
    upsample_layer = nn.Upsample(size=(7, 7), mode='nearest')
    return upsample_layer(input_tensor)

def load_planeset_example(config):
    filenamesCombis =  getCombiFromDBoptimalGoogleDrive(config)
    triplet_obj= Triplet(filenamesCombis[7], config)
    planeset_obj = Planeset(triplet_obj, config)
    return planeset_obj 

def load_example():
    model = models.vgg16(pretrained=True)
    #model = models.resnet18(weights='IMAGENET1K_V1')


    # Base model as feature extractor: remove classification head
    #base_model = nn.Sequential(*list(model.children())[:-2])
    base_model = nn.Sequential(*list(model.features.children())[:-1])
    base_model.eval()
    
    path_example = "C:/Users/Blanca/Documents/GitHub/OODbench/smallDatasets/ImageNetVal_small/n01498041/ILSVRC2012_val_00001935_n01498041.jpeg"
    img = Image.open(path_example)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(img)
    #print(image.size())
    
    # Expand dimensions to simulate batch size of 1
    input_batch= image.unsqueeze(0)
    with torch.no_grad():
        example = base_model(input_batch)
    
    print(example.size())
    return example


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', 
                        default='resnet18',
                        type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume',
                        type=str,
                        default = "caltech256-resnet18.pth" , 
                        #default = "C:/Users/Blanca/Documents/GitHub/OODbench/imagenet-vgg16.pth"
                        )      
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args


def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    utils2.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils2.load_dict(args.resume, model)
    
    
    # ["stingray", "junco", "bullfrog"]:
    class_selection = ["n01498041", "n01534433","n01641577"]

    config = Configuration(model= "ResNet18", 
                            N = 2, 
                            dataset = "ImageNetVal_small" ,
                            id_classes= class_selection,
                            resolution= 20,
                            useFilteredPaths = "False", 
                            )

    
    trans = transforms.ToPILImage()
    model.eval()
    print('=> Genarating ...')
    with torch.no_grad():
      input = load_latent_example(config,0).cuda()
      #input = load_example().cuda()
      output = model.module.decoder(input)
      output = trans(output.squeeze().cpu())
      output.show()
      print(output)
      #plt.savefig('figs/generation.jpg')

if __name__ == '__main__':

    args = get_args()

    main(args)


