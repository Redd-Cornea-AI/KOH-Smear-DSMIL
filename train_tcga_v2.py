# In this version, we taylored the training to our needs. 
# First, we select only slides with 0 and 1 decisions and exclude other classes
# Second, we merge the same patients WSIs together according to patient name
# Then, we create a way to include test set from an external source in '5-fold-cv-standalone-test' mode

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
import shutil
import random
import torch.backends.cudnn as cudnn

# add seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
cudnn.deterministic = True
cudnn.benchmark = False

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    print()
    print(cm)
    
def compute_class_weights(train_df, args):
    class_counts = np.zeros(args.num_classes)
    for i in range(len(train_df)):
        label, _, _ = get_bag_feats(train_df.iloc[i], args)
        class_counts += label
    print('Class counts of 0: ', class_counts)
    print('Class counts of 1: ', len(train_df)-class_counts)
    class_frequencies = class_counts / len(train_df)
    class_weights = 1.0 / (class_frequencies + 1e-10)  # Adding a small value to avoid division by zero
    return torch.tensor(class_weights, dtype=torch.float).cuda()
    
def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
                
    return label, feats, feats_csv_path

def generate_pt_files(args, df):
    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    print('Creating intermediate training files.')
    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
        # Save the stacked data into a .pt file
        pt_file_path = os.path.join(temp_train_dir, os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)

def generate_pt_files_test(args, df):
    temp_train_dir = "temp_test"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    print('Creating intermediate testing files.')
    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
        # Save the stacked data into a .pt file
        pt_file_path = os.path.join(temp_train_dir, os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)


def train(args, train_df, milnet, criterion, optimizer):
    milnet.train()
    dirs = shuffle(train_df)
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for i, item in enumerate(dirs):
        optimizer.zero_grad()
        stacked_data = torch.load(item, map_location='cuda:0')
        bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
        bag_feats = Tensor(stacked_data[:, :args.feats_size])
        bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))        
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False, return_files=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    test_files = []  # Added line to store the file names
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            test_files.append(item)  # Store the file path
            stacked_data = torch.load(item, map_location='cuda:0')
            bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
            bag_feats = Tensor(stacked_data[:, :args.feats_size])
            bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_predictions)
    
    if return_predictions and return_files:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels, test_files
    elif return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def test_nodropout(args, test_df, milnet, criterion, thresholds=None, return_predictions=False, return_files=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    test_files = []  # Added line to store the file names
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            test_files.append(item)  # Store the file path
            stacked_data = torch.load(item, map_location='cuda:0')
            bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
            bag_feats = Tensor(stacked_data[:, :args.feats_size])
            # bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_predictions)
    
    if return_predictions and return_files:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels, test_files
    elif return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs):
    if args.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score

def save_model(args, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

def print_save_message(args, save_name, thresholds_optimal):
    if args.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

# Function to create the new directory and copy the structure
def clone_directory_structure(src_dir, dst_dir):
    # print(f"Cloning directory structure from {src_dir} to {dst_dir}")
    for root, dirs, files in os.walk(src_dir):
        # Construct the path to the destination directory
        dest_path = root.replace(src_dir, dst_dir, 1)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

def combine_csv_files(src_dir, dst_dir, args):
    print(f"Combining CSV files in {src_dir} and saving to {dst_dir}")
    for root, dirs, files in os.walk(src_dir):
        # Skip processing files in the main directory to keep main csv intact
        if root == src_dir:            
            continue
        patient_data = {}
        for file in files:
            # print(f"Processing {file}")
            if file.endswith('.csv'):
                # Extract patient ID from the filename
                patient_id = file.split('_')[0]
                full_path = os.path.join(root, file)
                if patient_id not in patient_data:
                    patient_data[patient_id] = []
                patient_data[patient_id].append(pd.read_csv(full_path))
        
        # Combine data for each patient and save to new directory
        for patient_id, dfs in patient_data.items():
            combined_df = pd.concat(dfs, ignore_index=True)
            # Ensure the destination directory exists
            dst_folder = root.replace(src_dir, dst_dir, 1)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            # Construct new file path
            new_file_path = os.path.join(dst_folder, f'{patient_id}.csv')
            combined_df.to_csv(new_file_path, index=False)
    

# With this code we are merging the features for the same patient
def create_combined_path(file_path):
    # Split the path to get all components
    parts = file_path.split('/')
    if len(parts) > 1:  # Check to ensure there is at least one directory in the path
        parts[1] += '_combined'
    # Replace the filename part with patient_id and append .csv
    new_filename = parts[-1].split('_')[0] + '.csv'
    # Combine everything back except the original filename
    combined_path = '/'.join(parts[:-1]) + '/' + new_filename
    return combined_path

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 10x and 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512*2, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=50, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,1), help='GPU ID(s) [0, 1]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='KOH_Dataset_train_lambda', type=str, help='Dataset folder name')
    parser.add_argument('--dataset_test', default='KOH_Dataset_test_lambda', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0.2, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='5-fold-cv-standalone-test', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')

    
    args = parser.parse_args()
    print(args.eval_scheme)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        return milnet, criterion, optimizer, scheduler
    
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        bags_csv_test = os.path.join('datasets', args.dataset_test, args.dataset_test+'.csv')

    #######################
    # This code modifies the bags_csv to only include the slides that are originally 0 or 1
    df_annotated = pd.read_csv('Consolidated_annotations_KOHsmear.csv')
    # keep only "File name" and "Merged_numerical" columns
    df_annotated = df_annotated[['File name', 'Merged_numerical']]
    # remove "_cropped.jpeg" from "File name" end
    df_annotated['File name'] = df_annotated['File name'].str.replace('_cropped.jpeg', '')
    # print(df_annotated.head())
    
    # create new col for bags_csv 'File name' with the basename only
    df_current = pd.read_csv(bags_csv)
    df_current['File name'] = df_current['0'].str.split('/').str[-1]
    # remove ".csv" from 'File name'
    df_current['File name'] = df_current['File name'].str.replace('.csv', '')
    # print(df_current.head())
    
    # Merge the two dataframes on 'File name'
    df_merged = pd.merge(df_current, df_annotated, on='File name', how='inner')
    # remove 'File name' and 'label' col
    df_merged = df_merged.drop(columns=['File name', 'label'])
    # rename 'Merged_numerical' to 'label'
    df_merged = df_merged.rename(columns={'Merged_numerical': 'label'})
    # drop 'Merged_numerical' values of 2 and 3
    df_merged = df_merged[df_merged['label'] < 2]
    print(df_merged.head())
    
    # save new df as csv
    df_merged.to_csv(os.path.join('datasets', args.dataset, args.dataset+'_edited.csv'), index=False)
    print('CSV length after editing: ', len(df_merged))
    # define new bags_csv
    bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_edited.csv')
    #######################
    
    #######################
    # This code merges the same patients together by file name
    bags_path = pd.read_csv(bags_csv)
    print("CSV length before merging: ", len(bags_path))
     
    # Apply the function to each row in the DataFrame to create a new column for the combined path
    bags_path['0'] = bags_path['0'].apply(create_combined_path)
    print('CSV length after merging: ', len(bags_path))
    # Now consolidate the rows with the same combined path
    # always have 1 win over 0 because 1 is fungus and if there 1 patch of fungus, the whole slide is.
    bags_path = bags_path.groupby('0').agg({'label': 'max'}).reset_index()
    print('CSV length after consolidation: ', len(bags_path))
    # drop nan
    bags_path = bags_path.dropna()
    print('CSV length after dropping nans: ', len(bags_path))
    print('Merged same entries: ')
    print(bags_path.head())
    # print value counts without excluding nans
    print('Value counts: ')
    print(bags_path['label'].value_counts(dropna=False))
    # save the new csv for logging
    bags_path.to_csv('merged_patients_v5.csv', index=False)
    
    # Create the new directory name by appending '_combined'
    new_dataset_dir = 'datasets/' + args.dataset + '_combined'
    # Clone the directory structure of the original dataset
    clone_directory_structure('datasets/'+ args.dataset, new_dataset_dir)
    # Combine CSV files for the same patient and save them in the new dataset directory
    combine_csv_files('datasets/'+ args.dataset, new_dataset_dir, args)
    print(f"Dataset cloned and combined at: {new_dataset_dir}")
    
    bags_csv = os.path.join(new_dataset_dir,args.dataset+'_combined.csv')
    bags_path.to_csv(bags_csv, index=False)
    #######################
    
    class_weights = compute_class_weights(bags_path, args)
    
    ##### Test Set #####
    
    #######################
    # This code modifies the bags_csv to only include the slides that are originally 0 or 1
    df_annotated = pd.read_csv('Consolidated_annotations_KOHsmear.csv')
    # keep only "File name" and "Merged_numerical" columns
    df_annotated = df_annotated[['File name', 'Merged_numerical']]
    # remove "_cropped.jpeg" from "File name" end
    df_annotated['File name'] = df_annotated['File name'].str.replace('_cropped.jpeg', '')
    # print(df_annotated.head())
    
    # create new col for bags_csv 'File name' with the basename only
    df_current = pd.read_csv(bags_csv_test)
    df_current['File name'] = df_current['0'].str.split('/').str[-1]
    # remove ".csv" from 'File name'
    df_current['File name'] = df_current['File name'].str.replace('.csv', '')
    # print(df_current.head())
    
    # Merge the two dataframes on 'File name'
    df_merged = pd.merge(df_current, df_annotated, on='File name', how='inner')
    # remove 'File name' and 'label' col
    df_merged = df_merged.drop(columns=['File name', 'label'])
    # rename 'Merged_numerical' to 'label'
    df_merged = df_merged.rename(columns={'Merged_numerical': 'label'})
    # drop 'Merged_numerical' values of 2 and 3
    df_merged = df_merged[df_merged['label'] < 2]
    print(df_merged.head())
    
    # save new df as csv
    df_merged.to_csv(os.path.join('datasets', args.dataset_test, args.dataset_test+'_edited.csv'), index=False)
    print('CSV length after editing: ', len(df_merged))
    # define new bags_csv
    bags_csv_test = os.path.join('datasets', args.dataset_test, args.dataset_test+'_edited.csv')
    #######################
    
    #######################
    # This code merges the same patients together by file name
    bags_path_test = pd.read_csv(bags_csv_test)
    print("CSV length before merging: ", len(bags_path_test))
     
    # Apply the function to each row in the DataFrame to create a new column for the combined path
    bags_path_test['0'] = bags_path_test['0'].apply(create_combined_path)
    print('CSV length after merging: ', len(bags_path_test))
    # Now consolidate the rows with the same combined path
    # always have 1 win over 0 because 1 is fungus and if there 1 patch of fungus, the whole slide is.
    bags_path_test = bags_path_test.groupby('0').agg({'label': 'max'}).reset_index()
    print('CSV length after consolidation: ', len(bags_path_test))
    # drop nan
    bags_path_test = bags_path_test.dropna()
    print('CSV length after dropping nans: ', len(bags_path_test))
    print('Merged same entries: ')
    print(bags_path_test.head())
    # print value counts without excluding nans
    print('Value counts: ')
    print(bags_path_test['label'].value_counts(dropna=False))
    # save the new csv for logging
    bags_path_test.to_csv('merged_patients_v5_test.csv', index=False)
    
    # Create the new directory name by appending '_combined'
    new_dataset_dir = 'datasets/' + args.dataset_test + '_combined'
    # Clone the directory structure of the original dataset
    clone_directory_structure('datasets/'+ args.dataset_test, new_dataset_dir)
    # Combine CSV files for the same patient and save them in the new dataset directory
    combine_csv_files('datasets/'+ args.dataset_test, new_dataset_dir, args)
    print(f"Dataset cloned and combined at: {new_dataset_dir}")
    
    bags_csv_test = os.path.join(new_dataset_dir,args.dataset_test+'_combined.csv')
    bags_path_test.to_csv(bags_csv_test, index=False)
    
    #######################
    
    all_predictions = [] # to store predictions for all folds
    all_labels = [] # to store labels for all folds
    all_files = [] # to store file paths for all folds
    
    generate_pt_files(args, pd.read_csv(bags_csv))
    generate_pt_files_test(args, pd.read_csv(bags_csv_test))

    if args.eval_scheme == '5-fold-cv-standalone-test':
        bags_path = glob.glob('temp_train/*.pt')
        bags_path = shuffle(bags_path)
        bags_path_test = glob.glob('temp_test/*.pt')
        bags_path_test = shuffle(bags_path_test)
        # edit here for the test set split
        reserved_testing_bags = bags_path_test
        # bags_path = bags_path[int(args.split*len(bags_path)):]
        
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_results = []
        fold_models = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            test_path = [bags_path[i] for i in test_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            best_model = []

            for epoch in range(1, args.num_epochs+1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
                
                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                    best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
                    milnet.cuda()
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
            fold_models.append(best_model)

        print("Starting testing phase.")
        fold_predictions = []
        fold_predictions_df = pd.DataFrame()
        auc_predictions = []
        for item in fold_models:
            best_model = item[0]
            optimal_thresh = item[1]
            test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test_nodropout(args, reserved_testing_bags, best_model.cuda(), criterion, thresholds=optimal_thresh, return_predictions=True)
            fold_predictions.append(test_predictions)
            # create new column for fold_predictions_df
            fold_predictions_df[f'Fold_{len(fold_predictions)}'] = test_predictions
            print("Fold predictions shape:", np.array(test_predictions).shape)
            auc_predictions.append(aucs)
        predictions_stack = np.stack(fold_predictions, axis=0)
        mode_result = mode(predictions_stack, axis=0)
        combined_predictions = mode_result[0]
        # combined_predictions = mode_result.mode[0]
        combined_predictions = combined_predictions.squeeze()
        auc_stack = np.stack(auc_predictions, axis=0)
        mean_auc = np.mean(auc_stack, axis=0)
        print("Mean AUC:", mean_auc)
        print("Fold results on CV (accuracy + auc):", fold_results)
        
        print("Fold predictions shape:", np.array(fold_predictions).shape)
        print("Mode result:", mode_result)
        print("Combined predictions before squeeze:", mode_result.mode[0])
        print("Combined predictions:", combined_predictions)
        
        # Create test directory if it doesn't exist
        os.makedirs('test', exist_ok=True)
        
        fold_predictions_df['Mode'] = combined_predictions
        fold_predictions_df['Labels'] = test_labels
        fold_predictions_df['Files'] = reserved_testing_bags
        # save fold_predictions_df to a csv file in /test/
        fold_predictions_df.to_csv(f'test/test_predictions_per_fold_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
        
        # create a df to save the predictions
        results_df = pd.DataFrame({
            'Predictions': combined_predictions,
            'Labels': test_labels,
            'Files': reserved_testing_bags
        })

        # save with current date and time
        results_df.to_csv(f'test/test_predictions_mode_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
        
        # # save combined predictions to a csv file
        # os.makedirs('test', exist_ok=True)
        # with open("test/test_predictions.csv", "w") as file:
        #     np.savetxt(file, combined_predictions, delimiter=',', fmt='%d')

        if args.num_classes > 1:
            # Compute Hamming Loss
            hammingloss = hamming_loss(test_labels, combined_predictions)
            print("Hamming Loss:", hammingloss)
            # Compute Subset Accuracy
            subset_accuracy = accuracy_score(test_labels, combined_predictions)
            print("Subset Accuracy (Exact Match Ratio):", subset_accuracy)
        else:
            accuracy = accuracy_score(test_labels, combined_predictions)
            print("Accuracy:", accuracy)
            balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
            print("Balanced Accuracy:", balanced_accuracy)

        os.makedirs('test', exist_ok=True)
        with open("test/test_list.json", "w") as file:
            json.dump(reserved_testing_bags, file)

        for i, item in enumerate(fold_models):
            best_model = item[0]
            optimal_thresh = item[1]
            torch.save(best_model.state_dict(), f"test/mil_weights_fold_{i}.pth")
            with open(f"test/mil_threshold_fold_{i}.json", "w") as file:
                optimal_thresh = [float(i) for i in optimal_thresh]
                json.dump(optimal_thresh, file)
                

if __name__ == '__main__':
    main()