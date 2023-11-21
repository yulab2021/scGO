# read_and_write.R

# Check if the correct number of arguments are provided
if (length(commandArgs(trailingOnly = TRUE)) != 2) {
  stop("Usage: Rscript read_and_write.R <input_file.rds> <output_file.csv>")
}

# Get input and output file names from command line arguments
in_file <- commandArgs(trailingOnly = TRUE)[1]
out_file <- commandArgs(trailingOnly = TRUE)[2]

# Load data from the RDS file
data <- readRDS(in_file)
sparse_matrix = data@assays$RNA@counts

dense_matrix <- as.matrix(sparse_matrix)

# Write data to a CSV file
write.csv(t(dense_matrix) , file = out_file, row.names = FALSE)

cat("Data successfully written to", out_file, "\n")
