import torchvision
import torchvision.transforms as transforms
import torch
import random
import os
from PIL import Image

def _dataset_picker(args, clean_trainset):
    trainset = clean_trainset
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    return trainset, trainloader

def _baseset_picker(args):
    if args.net in ["ViT_pt",'mlpmixer_pt','MLPMixer_pt']:
        size = 224
    else:
        size = 32
    if args.baseset == 'CIFAR10':
        ''' best transforms - figure out later (LF 06/11/21)
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''

        clean_trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=transform_train)
        #
        # clean_trainset, _ = torch.utils.data.random_split(clean_trainset,
        #                                             [100, int(len(clean_trainset) - 100)],
        #                                             generator=torch.Generator().manual_seed(42), )

        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=args.bs, shuffle=False, num_workers=4)

    elif args.baseset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071598291397095, 0.4866936206817627,
                0.44120192527770996), (0.2673342823982239, 0.2564384639263153,
                0.2761504650115967)),
        ])
        clean_trainset = torchvision.datasets.CIFAR100(root='~/data', train=True,
            download=True, transform=transform_train)
        # LIAM CHANGED TO SHUFFLE=FALSE
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=128, shuffle=False, num_workers=2)

    elif args.baseset == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        base_trainset = torchvision.datasets.SVHN(root='~/data', split='train',
            download=True, transform=transform_train)
        # LIAM CHANGED TO SHUFFLE=FALSE
        clean_trainset = _CIFAR100_label_noise(base_trainset, args.label_path)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=128, shuffle=False, num_workers=2)

    elif args.baseset == 'CIFAR_load':
        old_clean_trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=None)

        class _CIFAR_load(torch.utils.data.Dataset):
            def __init__(self, root, baseset, dummy_root='~/data', split='train', download=False, **kwargs):

                self.baseset = baseset
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))])
                self.transform = transform_train
                self.samples = os.listdir(root)
                self.root = root

            def __len__(self):
                return len(self.baseset)

            def __getitem__(self, idx):
                true_index = int(self.samples[idx].split('.')[0])
                true_img, label = self.baseset[true_index]
                return self.transform(Image.open(os.path.join(self.root,
                                                    self.samples[idx]))), label

        clean_trainset = _CIFAR_load(args.load_data, old_clean_trainset)
        clean_trainloader = torch.utils.data.DataLoader(
                    clean_trainset, batch_size=128, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return clean_trainset, clean_trainloader, testset, testloader

def get_data(args):
    print('==> Preparing data..')
    clean_trainset, clean_trainloader, testset, testloader = _baseset_picker(args)

    trainset, trainloader = _dataset_picker(args, clean_trainset)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''

    from myFunctions import imgToTensor


    # Step 1: Verify tensor sizes
    if img1.size() != img2.size() or img1.size() != img3.size():
        min_size = min(img1.size(), img2.size(), img3.size())
        min_height, min_width = min_size[1], min_size[2]  # Extract minimum height and width
        aspect_ratio = min_height / min_width  # Calculate aspect ratio
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((min_height, min_width)),  # Resize to the minimum common size
            transforms.Pad((0, 0, 0, int((min_width - min_height * aspect_ratio) / 2)), fill=0)  # Pad to achieve 1:1 aspect ratio
        ])
        img1 = transform(img1)
        img2 = transform(img2)
        img3 = transform(img3)
        img1= imgToTensor(img1)
        img2= imgToTensor(img2)
        img3= imgToTensor(img3)


    # Step 2: Subtract img1 from img2 and img3
    a = img2 - img1
    b = img3 - img1

    # Step 3: Check and resize a and b if necessary
    if a.size() != b.size():
        min_size = min(a.size(), b.size())
        min_height, min_width = min_size[1], min_size[2]  # Extract minimum height and width
        aspect_ratio = min_height / min_width  # Calculate aspect ratio
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((min_height, min_width)),  # Resize to the minimum common size
            transforms.Pad((0, 0, 0, int((min_width - min_height * aspect_ratio) / 2)), fill=0)  # Pad to achieve 1:1 aspect ratio
        ])
        a = transform(a)
        b = transform(b)

    # Step 4: Use a and b in the ResNet model for image classification
    # ... (code for using a and b in the ResNet model)
    #####################################################33
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]


    #Understanding the output - JMPT
    


    return a, b_orthog, b, coords

#plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r)
class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        
        range_l = 0.2
        range_r = 0.2
        
        
        self.base_img = base_img
        self.vec1 = vec1 #a
        self.vec2 = vec2 #b_orthogonal
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1,list2])

        print('Init plane_dataset')
        print('x_bounds: ',x_bounds)
        print('y_bounds: ',y_bounds)

        """"
        # Assuming you have the coordinates of the three images as coords
        original_img1_coords = coords[0]  # Coordinates of img1
        original_img2_coords = coords[1]  # Coordinates of img2
        original_img3_coords = coords[2]  # Coordinates of img3

        # Next, we need to find the indices of these coordinates in the grid
        # Let's assume grid is a 2D grid you've created as mentioned in your code

        # Calculate the closest indices for img1
        img1_indices = [int((original_img1_coords[0] - list1[0]) / (list1[1] - list1[0])),
                        int((original_img1_coords[1] - list2[0]) / (list2[1] - list2[0]))]

        # Calculate the closest indices for img2
        img2_indices = [int((original_img2_coords[0] - list1[0]) / (list1[1] - list1[0])),
                        int((original_img2_coords[1] - list2[0]) / (list2[1] - list2[0]))]

        # Calculate the closest indices for img3
        img3_indices = [int((original_img3_coords[0] - list1[0]) / (list1[1] - list1[0])),
                        int((original_img3_coords[1] - list2[0]) / (list2[1] - list2[0]))]

        print("Indices of img1 in the grid:", img1_indices)
        print("Indices of img2 in the grid:", img2_indices)
        print("Indices of img3 in the grid:", img3_indices)

        Indices of img1 in the grid: [4, 4]
        Indices of img2 in the grid: [44, 4]
        Indices of img3 in the grid: [20, 44]



        """
        

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, args):
    print('--->make_planeloader')
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])

    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r)


    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=args.batch_size_planeloader, shuffle=False, num_workers=0)
    return planeloader
