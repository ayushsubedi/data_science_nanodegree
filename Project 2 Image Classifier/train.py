import argparse
import utility
import train_worker

parser = argparse.ArgumentParser(description='Arguements to train the network')
parser.add_argument('data_directory', help='location of dataset directory', type = str)
parser.add_argument('--save_dir', help='location of directory used to save model checkpoint', type = str, action='store')
parser.add_argument('--arch', help='pretrained model architecture', type=str, default = "vgg13", action = 'store')
parser.add_argument('--learning_rate', help='learning rate of the model', type = float, default = 0.01, action = 'store')
parser.add_argument('--hidden_units', help='hidden units of the model', default = [512], action = 'store', type = lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--epochs', help='training epochs', type = int, default = 10, action = 'store')
parser.add_argument('--gpu', help='use GPU to train', action='store_true', default = False)

# getting params from argparse
args = parser.parse_args()
args_dict = vars(args)

# pretty printing input arguments
print('\nINPUT ARGUEMENTS ...')
utility.pretty_print(args_dict)

# validate input
print('\nVALIDATING INPUT ARGUEMENTS ...')
if not (utility.valid_input_train(args_dict)):exit()
print('\nINPUT ARGUEMENTS VALID...')
# train the model
train_worker.train_and_save(args_dict)
