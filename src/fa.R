options(warn = 1)

library(data.table)
library(sommer)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cv <- 0
  fold <- 0
  seed <- 1
  debug <- FALSE
  invert <- FALSE
} else {
  cv <- args[1]  # 0, 1, or 2
  fold <- args[2]  # 0, 1, 2, 3, or 4
  seed <- args[3]  # 1, ..., 10
  debug <- as.logical(args[4])  # TRUE or FALSE
  invert <- as.logical(args[5]) # TRUE or FALSE
}
cat('debug:', debug, '\n')
cat('invert:', invert, '\n')

# datasets
ytrain <- fread(paste0('output/cv', cv, '/ytrain_fold', fold, '_seed', seed, '.csv'), data.table = F)
ytrain <- transform(ytrain, Env = factor(Env), Hybrid = factor(Hybrid))
cat('ytrain shape:', dim(ytrain), '\n')
yval <- fread(paste0('output/cv', cv, '/yval_fold', fold, '_seed', seed, '.csv'), data.table = F)
yval <- transform(yval, Env = factor(Env), Hybrid = factor(Hybrid))

# additive matrix
kmatrix <- fread('output/kinship_additive.txt', data.table = F)
kmatrix <- as.matrix(kmatrix)
colnames(kmatrix) <- substr(colnames(kmatrix), 1, nchar(colnames(kmatrix)) / 2)  # fix column names
rownames(kmatrix) <- colnames(kmatrix)
print(kmatrix[1:5, 1:5])

# keep only phenotyped individuals
ind_idxs <- which(rownames(kmatrix) %in% c(ytrain$Hybrid, yval$Hybrid) == TRUE)
kmatrix <- kmatrix[ind_idxs, ind_idxs]
if (debug == TRUE) {
  set.seed(2023)
  sampled_idx <- sample(1:nrow(kmatrix), 100)
  kmatrix <- kmatrix[sampled_idx, sampled_idx]
  ytrain <- subset(ytrain, Hybrid %in% rownames(kmatrix))
}
cat('Number of individuals being used:', nrow(kmatrix), '\n')
cat('dim:', dim(kmatrix), '\n')

# prepare relationship matrix for sommer
if (invert == TRUE) {
  A <- MASS::ginv(kmatrix)
  print(A[1:5, 1:5])
  A_list <- list(Hybrid = A)
} else {
  A_list <- list(Hybrid = kmatrix)
}

# modeling
set.seed(2023)
gc()

converged <- FALSE
mod <- NULL

tryCatch({
  mod <- mmer(
    Yield_Mg_ha ~ Env,
    random = ~ vsr(usr(Env), Hybrid, Gu = A_list),
    data = ytrain,
    verbose = FALSE
  )
  
  # check convergence
  if (!is.null(mod$convergence) && mod$convergence == TRUE) {
    converged <- TRUE
    cat('Model converged successfully\n')
  } else {
    cat('WARNING: Model may not have converged properly\n')
    converged <- FALSE
  }
  
}, error = function(e) {
  cat('ERROR: Model fitting failed:', conditionMessage(e), '\n')
  converged <<- FALSE
})

gc()

if (!is.null(mod)) {
  varcomp <- as.data.frame(summary(mod)$varcomp)
  varcomp <- transform(varcomp, VarComp = round(VarComp, 8))
  print(varcomp)
  
  # Count variance components (excluding residual)
  fa_comps <- sum(grepl('Env.*Hybrid', rownames(varcomp)))
  cat('Number of variance components estimated:', fa_comps, '\n')
  
  # FA number of estimated components is E(k+1) - k(k-1)/2, where E is the number of environments and k is the FA order
  E <- length(unique(ytrain$Env))
  k <- 1
  exp_fa_comps <- E * (k + 1) - 0.5 * k * (k - 1)
  cat('Number of componentes expected from formula (E(k+1) - k(k-1)/2):', exp_fa_comps, '\n')
  
  evaluate <- function(df) {
    df$error <- df$Yield_Mg_ha - df$predicted.value
    rmses <- with(df, aggregate(error, by = list(Env), FUN = function(x) sqrt(mean(x ^ 2))))
    colnames(rmses) <- c('Env', 'RMSE')
    print(rmses)
    cat('RMSE:', mean(rmses$RMSE), '\n')
  }
  
  # get predictions
  pred <- predict(mod, classify = c('Env', 'Hybrid'))
  pred <- as.data.frame(pred$pvals)
  pred <- pred[, c('Env', 'Hybrid', 'predicted.value')]
  
  pred_train_env_hybrid <- merge(ytrain, pred, by = c('Env', 'Hybrid'))
  
  # average between years
  val_year <- sub('(.*)_', '', yval$Env[1])
  pred$Field_Location <- as.factor(sub('_(.*)', '', pred$Env))
  pred <- with(pred, aggregate(predicted.value, list(Field_Location, Hybrid), mean))
  colnames(pred) <- c('Field_Location', 'Hybrid', 'predicted.value')
  pred$Env <- paste0(pred$Field_Location, '_', val_year)
  
  # merge on val
  pred_env_hybrid <- merge(yval, pred, by = c('Env', 'Hybrid'))
  evaluate(pred_env_hybrid)
  
  # write predictions
  cols <- c('Env', 'Hybrid', 'Yield_Mg_ha', 'predicted.value')
  pred_env_hybrid <- pred_env_hybrid[, cols]
  colnames(pred_env_hybrid) <- c('Env', 'Hybrid', 'ytrue', 'ypred')
  
  # Add convergence flag to output
  pred_env_hybrid$converged <- converged
  
  if (debug == FALSE) {
    fwrite(pred_env_hybrid, paste0('output/cv', cv, '/oof_fa_model_fold', fold, '_seed', seed, '.csv'))
  }
  
  # write predictions for train
  pred_train_env_hybrid <- pred_train_env_hybrid[, cols]
  colnames(pred_train_env_hybrid) <- c('Env', 'Hybrid', 'ytrue', 'ypred')
  pred_train_env_hybrid$converged <- converged
  
  if (debug == FALSE) {
    fwrite(pred_train_env_hybrid, paste0('output/cv', cv, '/pred_train_fa_model_fold', fold, '_seed', seed, '.csv'))
  }
  
  cat('Correlation:', cor(pred_env_hybrid$ytrue, pred_env_hybrid$ypred), '\n')
  # plot(pred_env_hybrid$ytrue, pred_env_hybrid$ypred)
  
} else {
  cat('ERROR: Could not fit model, no predictions generated\n')
  if (debug == FALSE) {
    # Write empty files to indicate failure
    empty_df <- data.frame(
      Env = character(0),
      Hybrid = character(0),
      ytrue = numeric(0),
      ypred = numeric(0),
      converged = logical(0)
    )
    fwrite(empty_df, paste0('output/cv', cv, '/oof_fa_model_fold', fold, '_seed', seed, '.csv'))
    fwrite(empty_df, paste0('output/cv', cv, '/pred_train_fa_model_fold', fold, '_seed', seed, '.csv'))
  }
}