import argparse
import pickle
import numpy as np
import pandas as pd


#read data, return a pandas dataframe
def read_data(data_file):
    data=pd.read_csv(data_file,sep=',',index_col=0)

    return data

#split data
def split_data(data):
    pass


#data nomalization
def data_normalization(data):
    """
    Normalize the rows of a DataFrame by dividing each row by its sum and multiplying by 1e4.

    Parameters:
    - data: Input DataFrame with rows to be normalized.

    Returns:
    - Normalized DataFrame.

    Example usage:
    Assuming you have your data DataFrame,
    normalized_data = data_normalization(data)
    """

    # Calculate the sum of each row
    row_sums = data.sum(axis=1)

    # Normalize each row by dividing by its sum and multiplying by 1e4
    data_normalized = data.div(row_sums, axis=0) * 1e4

    return data_normalized



def filter_genes_by_cells(data, min_cells=1000):
    """
    Filter genes based on the number of cells expressing each gene.

    Parameters:
    - data: Input DataFrame with cells as columns and genes as rows.
    - min_cells: Minimum number of cells expressing a gene to be retained.

    Returns:
    - Filtered DataFrame with genes expressed in at least min_cells.

    Example usage:
    Assuming you have your data_sparse DataFrame
    data_filtered = filter_genes_by_expression(data_sparse, min_cells=1000)
    print(data_filtered.shape)

    """
    # Statistics of cells expressing each gene
    gene_expressed_cell_number = data.astype(bool).sum(axis=0)

    print(f"Total Genes: {len(gene_expressed_cell_number)}")

    # Filter genes expressed in fewer than min_cells
    gene_expressed_cell_number = gene_expressed_cell_number[gene_expressed_cell_number > min_cells]

    print(f"Genes after filtering: {len(gene_expressed_cell_number)}")

    # Filter the original data based on selected genes
    data_filtered = data[gene_expressed_cell_number.index.tolist()]

    return data_filtered




def filter_genes_by_expression(data, num_genes=2000):
    """
    Filter the top genes based on the number of cells in which they are expressed.

    Parameters:
    - data: Input DataFrame with cells as columns and genes as rows.
    - num_genes: Number of top genes to keep.

    Returns:
    - Filtered DataFrame with the top num_genes genes expressed in the most cells.

    Example usage:
    Assuming you have your data_sparse DataFrame
    data_filtered = filter_top_genes_by_expression(data_sparse, num_genes=2000)
    print(data_filtered.shape)

    """
    # Statistics of cells expressing each gene (total expression)

    gene_total_expression = data.astype(bool).sum(axis=0)

    print(f"Total Genes: {len(gene_total_expression)}")

    # Sort genes based on total expression in descending order
    sorted_genes = gene_total_expression.sort_values(ascending=False)
    
    # Select the top num_genes genes
    top_genes = sorted_genes.head(num_genes).index.tolist()

    print(f"Top Genes after filtering: {len(top_genes)}")

    # Filter the original data based on selected top genes
    data_filtered = data[top_genes]

    return data_filtered



def norm_and_filter_command(args):
    print(f"Running norm_and_filter with parameters: {args}")

    # Read data
    data = read_data(args.gene_expression_matrix)

    # Normalize data
    data_normalized = data_normalization(data)

    # Filter genes by expression
    data_filtered = filter_genes_by_expression(data_normalized, num_genes=int(args.num_genes))

    # Save data
    data_filtered.to_csv(args.output, sep=",", header=True, index=True)


def build_network_command(args):
    print(f"Running build_network with parameters: {args}")
    
    base_dir = "/".join(args.GO_annotation.split("/")[:-1])

    #preprare TF 

    TF_gene_dict={}          #{TF:[gene]}
    gene_TF_dict={}          #{gene:[TF]}

    with open(args.TF_annotation) as f:
        for line in f:
            if "PeakID" in line:
                continue
            if "promoter-TSS" in line:
                TF=line.split(":")[0]
                gene_name=line.split("\t")[15]
                #print(TF,gene_name)
                if TF not in TF_gene_dict:
                    TF_gene_dict[TF]=[]

                #if gene_name not in TF_gene_dict[TF]:
                TF_gene_dict[TF].append(gene_name)

                if gene_name not in gene_TF_dict:
                    gene_TF_dict[gene_name]=[]
                #if TF not in gene_TF_dict[gene_name]:
                gene_TF_dict[gene_name].append(TF)
                
    print("Number of TFs: ",len(TF_gene_dict))

    pickle.dump(TF_gene_dict,open("%s/TF_gene_dict" %base_dir,"wb"))
    pickle.dump(gene_TF_dict,open("%s/gene_TF_dict" %base_dir,"wb"))

    
    #generate gene_to_TF_transform_matrix
    with open(args.gene_expression_matrix) as f:
        gene_list=f.readline().strip().split(",")[1:]
        num_gene=len(gene_list)

    num_TF=len(TF_gene_dict)

    gene_to_TF_transform_matrix=np.zeros((num_gene,num_TF))

    TF_list=list(TF_gene_dict.keys())

    for i,gene in enumerate(gene_list):
        try:
            j=TF_list.index(gene)
            gene_to_TF_transform_matrix[i][j]=1
        except:
            pass
            
    gene_to_TF_transform_matrix


    pickle.dump(gene_list,open("%s/feature" %base_dir,"wb"))
    pickle.dump(gene_to_TF_transform_matrix,open("%s/gene_to_TF_transform_matrix" %base_dir,"wb"))

    
    TF_number = len(TF_gene_dict)

    TF_mask = np.zeros((num_gene,TF_number))
    error_count=0

    for i,gene_id in enumerate(gene_list):

        for j,TF in enumerate(TF_gene_dict):
            if TF in gene_TF_dict.get(gene_id,[]):
                TF_mask[i][j]=1
            else:
                error_count+=1
            


    pickle.dump(TF_mask,open("%s/TF_mask" %base_dir,"wb"))

    #generate GO_mask
    GO_dict={}
    with open(args.GO_annotation) as f:
        for line in f:
            if line[0] == "!":
                continue
            
            gene_id=line.split("\t")[2]
            GO_term=line.split("\t")[4]
            if GO_term not in GO_dict:
                GO_dict[GO_term]=[]
            GO_dict[GO_term].append(gene_id)


    GO_list=[]
    count=0
    for item in GO_dict:
        if len(GO_dict[item])>=30:
            count+=1
            GO_list.append(item)



    gene_dict={}
    with open(args.GO_annotation) as f:
        for line in f:
            if line[0]=="!":
                continue
            gene_id=line.split("\t")[2].upper()
            GO_term=line.split("\t")[4]
            if gene_id not in gene_dict:
                gene_dict[gene_id]=[]
            gene_dict[gene_id].append(GO_term)



    
    num_GO=len(GO_list)  

    GO_mask=np.zeros((num_gene,num_GO))


    for i,gene_id in enumerate(gene_list):

        for j,GO_term in enumerate(GO_list):
            if GO_term in gene_dict.get(gene_id,"GO:default"):

                GO_mask[i][j]=1
            else:
                pass
        

    pickle.dump(GO_mask,open("%s/GO_mask" %base_dir,"wb"))


    #generate GO_to_TF_mask
    GO_TF_mask=np.zeros((num_TF,num_GO))


    for i,TF in enumerate(TF_gene_dict):
        for j,GO in enumerate(GO_list):
            if GO in gene_dict.get(TF,"GO:default"):
                GO_TF_mask[i][j]=1
            else:
                pass

    pickle.dump(GO_TF_mask,open("%s/GO_TF_mask" %base_dir,"wb"))




def classification_command(args):
    print(f"Running classification with parameters: {args}")

def regression_command(args):
    print(f"Running regression with parameters: {args}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="My Machine Learning Script")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Subparser for the 'norm and filter' command
    norm_and_filter_parser = subparsers.add_parser("norm_and_filter", help="Normalize and filter data")
    norm_and_filter_parser.add_argument("--gene_expression_matrix", help="Path to the input data")
    norm_and_filter_parser.add_argument("--num_genes", help="Number of top genes to keep")
    norm_and_filter_parser.add_argument("--output", help="Path to the output data")


    # Subparser for the 'buid_betwork' command
    build_network_parser = subparsers.add_parser("build_network", help="Build network")
    build_network_parser.add_argument("--gene_expression_matrix", help="Path to the input data")
    build_network_parser.add_argument("--GO_annotation", help="Path to the GO annotation file")
    build_network_parser.add_argument("--TF_annotation", help="Path to the TF annotation file")



    args = parser.parse_args()

    if args.subcommand == "norm_and_filter":
        norm_and_filter_command(args)
    elif args.subcommand == "build_network":
        build_network_command(args)

    else:
        print("Please provide a valid subcommand. Use the -h flag to see help about the available subcommands.")


    


























