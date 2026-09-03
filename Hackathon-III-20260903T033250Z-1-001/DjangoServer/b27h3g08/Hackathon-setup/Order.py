from . import Order_model
import warnings
import os
import torch
# Add more imports if required
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

list_task = [["Smart Phones", "Laptops", "Speakers", "Hard Disks", "Power Banks", "Headphones"], ['Amazon', 'Flipkart', 'Paytmmall', 'RelianceDigital', 'Croma', 'Tata']]
menu_prompts = ["menu_list.wav", "quantity_list.wav"]
# list_task[0] is list of product items
# list_task[1] is list of stores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################################################################################
#		 Caution: Don't change any of the filenames, function names and definitions                                                       #
#		Always use the BASE_DIR + "/Hackathon-setup/<filename>" for refering any files, without it we cannot access files on the server   #
###########################################################################################################################################

class order:
    # Classify based on the given features and model
    # It should return the predicted label of the model
    def classify_input(self, features, model):
        # YOUR CODE HERE to Unsqueeze the features
        predicted_label = # YOUR CODE HERE to Pass the unsqeezed features through the model and get the predicted label
        return predicted_label

    def confirm_input(self, digit, flag):
        if digit < len(list_task[flag]):
            return digit, list_task[flag][digit]
        return -99, -99

    def take_user_input(self, flag):
        # Extracting features from the user input
        features = Order_model.get_features(BASE_DIR + "/Hackathon-setup/1_userinput_1.wav")

        ############################################################################################
        ## Example for loading a model using weight state dictionary:                             ##
        ## feature_net = Order_model.Net()#Example network                                       ##
        ## ckpt = torch.load(BASE_DIR + "/Hackathon-setup/speech_model.t7", map_location=device)  ##
        ## feature_net.load_state_dict(ckpt['net_dict'])                                          ##
        ############################################################################################

        # YOUR CODE HERE to Load the model from server (refer the above example)
        # model = 

        digit = self.classify_input(features, model)
        digit, choice = self.confirm_input(digit, flag)
        return digit, choice
