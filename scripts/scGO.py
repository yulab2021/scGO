
from models import scGO
from utils import MyDataset,accuracy_score,select_and_pad_features,label_decoding,label_encoding
from utils import data_normalization
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable





class Configuration:
    """
    Configuration class to store parameters for the scGO model.
    """

    def __init__(self):
        """
        Initialize Configuration with default values.
        # Example usage
        config = Configuration()

        # Accessing attributes
        print(config.data_file)
        print(config.epoch)
        print(config.criteria_cross_entropy)
        """

        # File paths
        self.data_dir = "data"  # Directory to strore data
        self.model_dir = "models"  # Directory to store models
        self.data_file = None  # Path to the data file
        self.pretrained_model = None  # Path to a pre-trained model
        self.new_model = None  # Path to save a new model
        self.train_data = None  # Path to the training data
        self.predict_result = None  # Path to store prediction results

        # Training parameters
        self.epoch = 50  # Number of training epochs
        self.task = 'classification'  # Task type ('classification' or 'regression')
        self.learning_rate = 0.001  # Learning rate for optimization
        self.batch_size = 20  # Batch size for training

        # Loss functions
        self.criteria_cross_entropy = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
        self.criteria_mes = nn.MSELoss()  # Mean Squared Error loss for regression






def train(args):

    config = Configuration()
    config.data_dir = "/".join(args.gene_expression_matrix.split("/")[:-1])

    #load predefined model masks and connections
    gene_to_TF_transform_matrix=pickle.load(open("%s/gene_to_TF_transform_matrix" %config.data_dir,"rb"))
    TF_mask=pickle.load(open("%s/TF_mask" %config.data_dir,"rb"))
    GO_mask=pickle.load(open("%s/GO_mask" %config.data_dir,"rb"))
    GO_TF_mask=pickle.load(open("%s/GO_TF_mask" %config.data_dir,"rb"))

    #load gene expression matrix
    gene_expression_matrix=pd.read_csv(args.gene_expression_matrix,sep=",",index_col=0)

    #load meta_data
    meta_data = pd.read_csv(args.meta_data,sep=",",index_col=0)
    labels = meta_data[args.label].tolist()
    
    classes, encoded_labels = label_encoding(labels)

    num_gene = GO_mask.shape[0]
    num_GO = GO_mask.shape[1]
    num_TF = TF_mask.shape[1]
    num_class=len(set(encoded_labels))

    input_size=num_gene
    output_size=num_class


    model = scGO(input_size, output_size, num_GO, num_TF, gene_to_TF_transform_matrix, GO_mask, TF_mask, GO_TF_mask, ratio=[0,0,0])

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


    train_x,test_x,train_y,test_y=train_test_split(gene_expression_matrix,encoded_labels,test_size=0.2,random_state=0,shuffle=True)


    train_data=MyDataset(train_x.to_numpy(),train_y)
    test_data=MyDataset(test_x.to_numpy(),test_y)

    train_loader=DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader=DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epoch):
        running_loss = 0.0

        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            #print(labels)
            inputs=Variable(inputs).to(torch.float32)
            labels=Variable(labels).to(torch.long)
            # 将梯度缓存清零
            optimizer.zero_grad()

            # 前向传播、计算损失和反向传播
            outputs,_,_,_ = model(inputs)

            loss = config.criteria_cross_entropy(outputs, labels)

            loss.backward()
        
            optimizer.step()

            running_loss += loss.item()

            
            if i>400:
                break

        result=[]        
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            
            inputs=Variable(inputs).to(torch.float32)

            labels=Variable(labels).to(torch.long)
            
            outputs,_,_,_ = model(inputs)
            pred = list(torch.max(outputs, 1)[1].numpy())
            result.extend(pred)
            #print(pred,labels)
            if i>100:
                break
        accuracy = accuracy_score(test_y[0:len(result)],result)
        #f1_score = calculate_multiclass_f1_score(y_test_relabeled[0:len(result)],result)

        print("epoch %s" %(epoch),"\taccuracy:\t",accuracy,"\tloss:\t",running_loss / len(train_loader))

        #save model
        data_to_dump = (classes, model)
        pickle.dump(data_to_dump,open("%s" %args.model,"wb"))

def predict(args):
    
    config = Configuration()
    config.data_dir = "/".join(args.gene_expression_matrix.split("/")[:-1])


    feature=pickle.load(open("%s/feature" %config.data_dir,"rb"))

    #load predefined model masks and connections
    gene_to_TF_transform_matrix=pickle.load(open("%s/gene_to_TF_transform_matrix" %config.data_dir,"rb"))
    TF_mask=pickle.load(open("%s/TF_mask" %config.data_dir,"rb"))
    GO_mask=pickle.load(open("%s/GO_mask" %config.data_dir,"rb"))
    GO_TF_mask=pickle.load(open("%s/GO_TF_mask" %config.data_dir,"rb"))

    #load gene expression matrix
    gene_expression_matrix=pd.read_csv(args.gene_expression_matrix,sep=",",index_col=0)
    

    data_normalized = data_normalization(gene_expression_matrix)

    test_data = data_normalized[feature]



    #load model
    classes, model = pickle.load(open("%s" %args.model,"rb"))

    num_gene = GO_mask.shape[0]
    num_GO = GO_mask.shape[1]
    num_TF = TF_mask.shape[1]
    num_class=len(classes)

    input_size=num_gene
    output_size=num_class

    test_data=MyDataset(test_data.to_numpy(),np.zeros(gene_expression_matrix.shape[0]))

    test_loader=DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    result=[]
    probability=[]
    for i, batch in enumerate(test_loader):
        inputs, labels = batch
        
        inputs=Variable(inputs).to(torch.float32)

        outputs,_,_,_ = model(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())
        #result.extend(pred)
        #print(pred,labels)
        #print(nn.Softmax(dim=1)(outputs).tolist())
        for j,p in enumerate(pred):
            
            probability.append(nn.Softmax(dim=1)(outputs).tolist()[j][p])

            #report novel class
            if args.indicate_novel_cell_type:
                if max(nn.Softmax(dim=1)(outputs).tolist()[j])<0.8:
                    result.append("novel cell type")
                else:
                    result.append(classes[p])

            else:
                result.append(classes[p])
        #probability.append(nn.Softmax(dim=1)(outputs).tolist()[0][pred])

    #result=label_decoding(classes,result)
    #print(result)

    #write predictions to csv file
    prediction = pd.DataFrame({"cell_id":gene_expression_matrix.index.tolist(),"predicted cell_type":result,"probability":probability})#.to_csv(args.output,sep=",",header=True,index=False)
    print(prediction)
    prediction.to_csv(args.output,sep=",",header=True,index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scGO, biologically informed deep learning model for single cell annotation.')

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument('--gene_expression_matrix', required=True, default='', help='Train data file.')
    train_parser.add_argument('--meta_data', required=False, default='', help='Meta data, including cell type labels.')
    train_parser.add_argument('--model', required=True, default='', help='Output model file.')
    train_parser.add_argument('--task', required=False, default='classification', help='Task type, classification or regression.')
    train_parser.add_argument('--epoch', required=False, default=50, help='Number of training epochs.')
    train_parser.add_argument('--batch_size', required=False, default=12, help='Batch size.')
    train_parser.add_argument('--learning_rate', required=False, default=0.001, help='Learning rate.')
    train_parser.add_argument('--label', required=False, default="cell_type", help='Column in meta_data to be predicted.')
    #train_parser.add_argument('--indicate_novel_class', required=False, default=False, help='Whether to report novel class.')

    # Subparser for the 'predict' command
    predict_parser = subparsers.add_parser("predict", help="Predict using a trained model")
    predict_parser.add_argument('--gene_expression_matrix', required=True, default='',help='Test data file.')
    predict_parser.add_argument('--model', required=True, default='',help='Trained model file.')
    predict_parser.add_argument('--output', required=True, default='',help='Output file.')
    predict_parser.add_argument('--indicate_novel_cell_type', required=False, default=False, help='Whether to report novel class.')



    args = parser.parse_args()

    if args.subcommand == "train":
        train(args)

    elif args.subcommand == "predict":
        predict(args)

    else:
        print("Error: Need to specify a subcommand. Use --help for more information.")
        exit(-1)

    


























