import argparse
import utility
import predict_worker

parser = argparse.ArgumentParser(description='Arguements for prediction')
parser.add_argument('image_path', help='path to image', type = str)
parser.add_argument('checkpoint', help='model checkpoint', type =str)
parser.add_argument('--gpu', help='use GPU for inference', action='store_true', default = False)
parser.add_argument('--top_k', help='tok k probabilities', action='store', type = int, default = 2)
parser.add_argument('--category_names', help='labelling for category names', action='store', type=str)

# getting params from argparse
args = parser.parse_args()
args_dict = vars(args)

# pretty printing input arguments
print('\nINPUT ARGUEMENTS ...')
utility.pretty_print(args_dict)

# validate input
print('\nVALIDATING INPUT ARGUEMENTS ...')
if not (utility.valid_input_predict(args_dict)):exit()
print('\nINPUT ARGUEMENTS VALID...')
predict_worker.load_and_predict(args_dict)