import pprint
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch
import os
import numpy as np
from PIL import Image
import json



def pretty_print(obj):
	"""Simple pretty printer"""
	pprint.pprint (obj)


def output_print(probs, label):
	"""Simple output printer"""
	print ('\nLabels and probabilities')
	print ('************************')
	i = 0
	while i<len(probs):
		print(label[i], '------->' ,probs[i])
		i = i + 1


def valid_input_train(obj):
	"""Check training input parameter"""
	if not hasattr(models, obj.get('arch')):
		print ('Error: Unknown architecture. Check torchvision documentation for available pretrained models.')
		return False
	if not (os.path.exists(obj.get('data_directory'))):
		print ('Error: Data path not found. Check if folder exists.')
		return False
	if not (torch.cuda.is_available()):
		if (obj.get('gpu')):
			print ('Error: GPU was requested but is not supported by the system.')
			return False
	if (obj.get('save_dir') != None):
		if not (os.path.exists(obj.get('save_dir'))):
			print ('Error: Checkpoint save path not found. Check if folder exists.')
			return False
	return True



def valid_input_predict(obj):
	"""Check predict input parameter"""
	if not (os.path.exists(obj.get('image_path'))):
		print ('Error: Input image does not exist.')
		return False
	if not (os.path.exists(os.path.join(obj.get('checkpoint')))):
		print ('Error: Could not find model checkpoint. Make sure not to miss the extension.')
		return False
	if (obj.get('category_names') != None):
		if not (os.path.exists(os.path.join(obj.get('category_names')))):
			print ('Error: Request for category names was passed but path does not exist.')
			return False
	if not (torch.cuda.is_available()):
		if (obj.get('gpu')):
			print ('Error: GPU was requested but is not supported by the system.')
			return False
	return True



def read_dataset(data_dir):
	"""
	Torchvision transforms used to augment the training data with random scaling, rotations, mirroring and cropping
	The training, validation and testing data appropriately cropped and normalized
	The data for each set is loaded with torchvision's DataLoader
	The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
	"""
	normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	data_transforms = {
	    'train': transforms.Compose([
	        transforms.RandomRotation(30),
	        transforms.RandomResizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        normalize
	    ]),
	    'test': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        normalize
	    ]),
	    'valid': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        normalize
	    ]),
	}

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test', 'valid']}

	dataset_sizes = {x: len(image_datasets[x]) 
	                              for x in ['train', 'test', 'valid']}

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
	                                             shuffle=True)
	              for x in ['train', 'test', 'valid']}
	return image_datasets, dataset_sizes, dataloaders




def pretrained_model(arch):
	"""gets the required model with pretrained=True"""
	return getattr(models, arch)(True)



def get_input_feature(classifier):
	"""the models are linear or sequential"""
	try:
		return classifier[0].in_features
	except:
		return classifier.in_features



def classifier_dict(input_size, output_size, hidden_units):
	"""create a ordered dict with input size, output size, hidden units, activation functions, and dropout"""
	temp_dict = OrderedDict({})
	i=0
	temp_input = input_size
	while (i<len(hidden_units)):
		temp_dict.update({'fc'+str(i+1): nn.Linear(temp_input, hidden_units[i])})
		temp_dict.update({'relu'+str(i+1):nn.ReLU()})
		temp_dict.update({'dropout'+str(i+1):nn.Dropout(p=0.2)})
		temp_input = hidden_units[i]
		i = i + 1 
	temp_dict.update({'fc'+str(len(hidden_units)+1):nn.Linear(hidden_units[len(hidden_units)-1], output_size)})
	temp_dict.update({'output':nn.LogSoftmax(dim=1)})
	return temp_dict



def image_preprocessing(image_path):
    """Image preprocessing to convert pil image into an object tht can be used as input to a trained model"""
    im = Image.open(image_path)
    if (im.size[0] < im.size[1]):
        im.thumbnail((256, 10000), Image.ANTIALIAS)
    else:
        im.thumbnail((10000, 256), Image.ANTIALIAS)
    
    left =  (im.width-224)/2
    lower = (im.height-224)/2
    right = left+224
    upper = lower+224
    box = (left, lower, right, upper)
    im = im.crop(box)
    
    np_im = np.array(im)/225
    np_im = (np_im - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    np_im = np_im.transpose((2,0,1))
    return np_im



def import_labels(label_path):
    """import labels from json file"""
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name



def predict(image_path, model, topk, cuda, category_names):
    """predict label to top k probabilities associated with input image path""" 
    if (cuda):
        model = model.cuda()
    else:
        model = model.cpu()
    processed_image = image_preprocessing(image_path)

    if (cuda):
        processed_image_float = torch.from_numpy(processed_image).type(torch.cuda.FloatTensor)
    else:
   	    processed_image_float = torch.from_numpy(processed_image).type(torch.FloatTensor)
    
    processed_image_float.unsqueeze_(0)
    output = model.forward(processed_image_float)
    probability = torch.exp(output)
    probs, classes = probability.topk(topk)
    probs = probs.detach().numpy().tolist()[0] 
    classes = classes.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in    
                        model.class_to_idx.items()}
    if (category_names == None):
        labels = [idx_to_class[label] for label in classes]
    else:
        cat_to_name = import_labels(category_names)
        labels = [cat_to_name[idx_to_class[label]] for label in classes]
    return probs, labels