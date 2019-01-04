import torch
import utility
from torch import nn


def load_model(checkpoint_path):
    """load saved model from model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model = utility.pretrained_model(checkpoint['arch'])
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = nn.Sequential(checkpoint['classifier_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_and_predict(args_dict):
	"""predict the input"""
	print ('\nLOADING MODEL ...')
	model = load_model(args_dict.get('checkpoint'))
	print ('\nMODEL LOADED ...')
	print ('\nPREDICTING INPUT ...')
	probs, labels = utility.predict(args_dict.get('image_path'), model, args_dict.get('top_k'), args_dict.get('gpu'), args_dict.get('category_names')) 
	utility.output_print(probs, labels)
	print ('\nINPUT PREDICTED... DONE')