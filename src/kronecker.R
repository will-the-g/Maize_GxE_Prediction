library(data.table)
library(arrow)  # for write_feather

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cv <- 0
  debug <- TRUE
  kinship_type <- "additive"
} else {
  cv <- args[1]
  debug <- as.logical(args[2])
  kinship_type <- args[3]
}

# paths now in main directory
kinship_path <- paste0("kinship_", kinship_type, ".txt")
outfile <- paste0("cv", cv, "_kronecker_", kinship_type, ".arrow")

cat("Debug mode:", debug, "\n")
cat("Using", kinship_type, "matrix\n")

# read training/validation features from main folder
xtrain <- data.frame()
for (file in list.files('.', pattern = paste0('cv', cv, '_xtrain_fold.*\\.csv$'))) {
  xtrain <- rbind(xtrain, fread(file, data.table = FALSE))
}
xval <- data.frame()
for (file in list.files('.', pattern = paste0('cv', cv, 'xval_fold.*\\.csv$'))) {
  xval <- rbind(xval, fread(file, data.table = FALSE))
}

# bind files and aggregate
x <- rbind(xtrain, xval)
rm(xtrain); rm(xval); gc()
x <- x[, !grepl("yield_lag", colnames(x))]
x$Hybrid <- NULL
x <- aggregate(x[, -1], by = list(x$Env), FUN = mean)
rownames(x) <- x$Group.1
x$Group.1 <- NULL
x$Env <- NULL
x <- as.matrix(x)

# read phenotypes from main folder
ytrain <- data.frame()
for (file in list.files('.', pattern = paste0('cv', cv, 'ytrain_fold.*\\.csv$'))) {
  ytrain <- rbind(ytrain, fread(file, data.table = FALSE))
}
yval <- data.frame()
for (file in list.files('.', pattern = paste0('cv', cv, 'yval_fold.*\\.csv$'))) {
  yval <- rbind(yval, fread(file, data.table = FALSE))
}

# get unique combinations
y <- rbind(ytrain, yval)
y$Hybrid <- gsub("^Hybrid", "", y$Hybrid)
y <- y[y$Hybrid != "(Intercept)", ]
hybrids <- unique(y$Hybrid)
env_hybrid <- unique(interaction(y$Env, y$Hybrid, sep = ':', drop = TRUE))
rm(y); rm(ytrain); rm(yval); gc()

# load kinship
if (!debug) {
  kinship <- fread(kinship_path, data.table = FALSE)
} else {
  kinship <- fread(kinship_path, data.table = FALSE, nrows = 100)
}

# ============= MINIMAL DIAGNOSTIC OUTPUT =============
cat("\n==================== DIAGNOSTIC CHECK ====================\n")
cat("\n[1] KINSHIP FILE - First 5 column names (RAW, before any processing):\n")
print(colnames(kinship)[1:5])

colnames(kinship) <- substr(colnames(kinship), 1, nchar(colnames(kinship)) / 2)
kinship <- as.matrix(kinship)
rownames(kinship) <- colnames(kinship)[1:nrow(kinship)]

cat("\n[2] KINSHIP MATRIX - First 10 row/column names (AFTER processing):\n")
print(head(rownames(kinship), 10))
cat("\n[3] KINSHIP MATRIX - Last 10 row/column names:\n")
print(tail(rownames(kinship), 10))
cat("\n[4] PHENOTYPE FILE - First 10 unique hybrid IDs:\n")
print(head(hybrids, 10))
cat("\n[5] PHENOTYPE FILE - Last 10 unique hybrid IDs:\n")
print(tail(hybrids, 10))

overlap <- sum(hybrids %in% rownames(kinship))
cat("\n[6] OVERLAP CHECK:\n")
cat("    Total unique hybrids in phenotypes:", length(hybrids), "\n")
cat("    Total IDs in kinship matrix:", nrow(kinship), "\n")
cat("    Number of hybrids found in kinship: ", overlap, "\n")
cat("    Percentage overlap: ", round(100 * overlap / length(hybrids), 1), "%\n")

if (overlap > 0) {
  matching_hybrids <- hybrids[hybrids %in% rownames(kinship)]
  cat("\n[7] EXAMPLES OF MATCHING IDs (first 5):\n")
  print(head(matching_hybrids, 5))
}

if (overlap < length(hybrids)) {
  non_matching <- hybrids[!(hybrids %in% rownames(kinship))]
  cat("\n[8] EXAMPLES OF NON-MATCHING HYBRID IDs (first 10):\n")
  print(head(non_matching, 10))
}

cat("\n[9] QUICK FORMAT CHECKS:\n")
cat("    Kinship IDs contain underscore '_':", sum(grepl("_", rownames(kinship))) > 0, "\n")
cat("    Kinship IDs contain 'x':", sum(grepl("x", rownames(kinship), fixed = TRUE)) > 0, "\n")
cat("    Kinship IDs contain dash '-':", sum(grepl("-", rownames(kinship), fixed = TRUE)) > 0, "\n")
cat("    Hybrid IDs contain underscore '_':", sum(grepl("_", hybrids)) > 0, "\n")
cat("    Hybrid IDs contain 'x':", sum(grepl("x", hybrids, fixed = TRUE)) > 0, "\n")
cat("    Hybrid IDs contain dash '-':", sum(grepl("-", hybrids, fixed = TRUE)) > 0, "\n")
cat("\n==========================================================\n\n")
# ============= END DIAGNOSTIC OUTPUT =============

# continue original code
kinship <- kinship[rownames(kinship) %in% hybrids, colnames(kinship) %in% hybrids]
cat("kinship dim:", dim(kinship), "\n")

x <- x[, colSums(is.na(x)) < nrow(x)]
x <- x[complete.cases(x), ]

K <- kronecker(x, kinship, make.dimnames = TRUE)
rm(x); rm(kinship); gc()
cat("K dim:", dim(K), "\n")

cat("env_hybrid length:", length(env_hybrid), "\n")
cat("rownames(K) length:", length(rownames(K)), "\n")
print(head(env_hybrid))
print(head(rownames(K)))

K <- K[rownames(K) %in% env_hybrid, ]
cat("K dim:", dim(K), "\n")
cat("K size:", format(object.size(K), units = "MB"), "\n")
cat("Number of rows in K:", nrow(K), "\n")

cat("[STEP 1] Creating data frame...\n")
flush.console()
K_df <- data.frame(id = rownames(K), K)

cat("[STEP 2] Data frame created. Writing to feather...\n")
flush.console()
arrow::write_feather(K_df, outfile)

cat("[STEP 3] Feather write complete!\n")
flush.console()
rm(K_df); rm(K); gc()
cat("Writing file:", outfile, "\n\n")
Sys.sleep(5)
