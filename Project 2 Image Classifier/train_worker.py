import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import copy

import utility

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs, cuda):
    """train the model, print validation and training loss, and accuracy"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if (cuda):
        model.cuda()
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if (cuda):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_model(arch, classifier_dict, class_to_idx, save_dir, trained_model):
	"""save the model to a checkpoint"""
	if (save_dir == None):
		pth = 'checkpoint.pth'
	else:
		pth = os.path.join(save_dir, 'checkpoint.pth')
	torch.save({'arch': arch,
            'classifier_dict': classifier_dict, 
            'state_dict': trained_model.state_dict(), 
            'class_to_idx': class_to_idx}, 
             pth) 
	
def train_and_save(args_dict):
	"""train and save the model to a checkpoint"""
	print('\nLOAD DATASETS...')
	image_datasets, dataset_sizes, dataloaders = utility.read_dataset(args_dict.get('data_directory'))
	print('\nDATASETS LOADED...')
	print('\nLOAD PRETRAINED MODEL...')
	model = utility.pretrained_model(args_dict.get('arch'))
	print('\nPRETRAINED MODEL LOADED...')
	# freezing the model parameters
	for param in model.parameters():
		param.requires_grad = False

	input_size = utility.get_input_feature(model.classifier)
	output_size = len(image_datasets['train'].classes)
	model_classifier_dict = utility.classifier_dict(input_size, output_size, args_dict.get('hidden_units'))
	classifier = nn.Sequential(model_classifier_dict)
	print (classifier)
	model.classifier = classifier
	# NLLLoss because our output is LogSoftmax
	criterion = nn.NLLLoss()
	# Adam optimizer with a learning rate
	optimizer = optim.Adam(model.classifier.parameters(), lr = args_dict.get('learning_rate'))
	# Decay LR by a factor of 0.1 every 5 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
	cuda = args_dict.get('gpu')
	print('\nTRAIN THE MODEL...')
	trained_model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, args_dict.get('epochs'), cuda)
	print('\nMODEL TRAINED...')
	print('\nSAVE TRAINED MODEL...')
	save_model(args_dict.get('arch'), model_classifier_dict, image_datasets['train'].class_to_idx, args_dict.get('save_dir'), trained_model)
	print('\nTRAINED MODEL SAVED...')