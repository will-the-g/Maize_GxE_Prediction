# Load required packages
library(lme4)
library(dplyr)

# Read dataset
data <- read.csv("data/Training_Data/1_Training_Trait_Data_2014_2021.csv")

# Define environments
envs2019 <- c('DEH1_2019', 'TXH2_2019', 'NCH1_2019', 'SCH1_2019', 'IAH3_2019', 'MNH1_2019', 'IAH2_2019', 'TXH3_2019', 'NYH3_2019', 'ILH1_2019',
              'WIH1_2019', 'GAH1_2019', 'WIH2_2019', 'TXH1_2019', 'IAH4_2019', 'MIH1_2019', 'INH1_2019', 'GEH1_2019', 'IAH1_2019', 'NYH2_2019', 
              'GAH2_2019', 'NEH2_2019', 'NEH1_2019')

envs2020 <- c('DEH1_2020', 'GAH1_2020', 'GAH2_2020', 'GEH1_2020', 'IAH1_2020', 'INH1_2020', 'MIH1_2020', 'MNH1_2020', 'NCH1_2020', 'NEH1_2020', 'NEH2_2020',
              'NEH3_2020', 'NYH2_2020', 'NYH3_2020', 'NYS1_2020', 'SCH1_2020','TXH1_2020', 'TXH2_2020', 'TXH3_2020', 'WIH1_2020', 'WIH2_2020', 'WIH3_2020')

envs2021 <- c('COH1_2021', 'DEH1_2021', 'GAH1_2021', 'GAH2_2021', 'GEH1_2021', 'IAH1_2021', 'IAH2_2021', 'IAH3_2021', 'IAH4_2021', 'ILH1_2021', 'INH1_2021', 'MIH1_2021',
              'MNH1_2021', 'NCH1_2021', 'NEH1_2021', 'NEH2_2021', 'NEH3_2021', 'NYH2_2021', 'NYH3_2021', 'NYS1_2021', 'SCH1_2021', 'TXH1_2021', 'TXH2_2021', 'TXH3_2021',
              'WIH1_2021', 'WIH2_2021', 'WIH3_2021')

envs <- c(envs2019) # , envs2020, envs2021

# Initialize storage for results
all_blues <- data.frame()
cvs_h2s <- data.frame()

# Loop through environments
for (env in envs) {
  cat("Processing environment:", env, "\n")
  
  # Filter for the environment
  env_data <- data %>% filter(Env == env)
  
  # Convert categorical variables to factors
  env_data$Hybrid <- as.factor(env_data$Hybrid)
  env_data$Replicate <- as.factor(env_data$Replicate)
  
  # Check for sufficient data
  if (nlevels(env_data$Hybrid) < 2 || nlevels(env_data$Replicate) < 1) {
    cat("Skipping environment due to insufficient levels for Hybrid or Replicate.\n")
    next
  }
  
  # Fit linear mixed-effects model
  model <- lmer(Yield_Mg_ha ~ Hybrid + (1 | Replicate), data = env_data)
  
  # Extract fixed effects (BLUEs)
  blues <- as.data.frame(fixef(model))
  blues <- data.frame(Hybrid = rownames(blues), predicted.value = blues[, 1])
  blues$predicted.value <- blues$predicted.value + fixef(model)["(Intercept)"]
  blues$Env <- env
  
  all_blues <- rbind(all_blues, blues)
  
  # CV and Heritability
  residual_variance <- summary(model)$sigma^2
  mean_yield <- mean(env_data$Yield_Mg_ha, na.rm = TRUE)
  cv <- sqrt(residual_variance) / mean_yield
  cat("CV for", env, ":", cv, "\n")
  
  var_components <- as.data.frame(VarCorr(model))
  print("Variance components:")
  print(var_components)
  
  if ("Hybrid" %in% var_components$grp) {
    var_hybrid <- var_components$vcov[var_components$grp == "Hybrid"]
  } else {
    var_hybrid <- 0
  }
  
  heritability <- var_hybrid / (var_hybrid + residual_variance)
  cat("Heritability (H²) for", env, ":", heritability, "\n\n")
  
  cvs_h2s <- rbind(cvs_h2s, data.frame(Env = env, CV = cv, H2 = heritability))
}

# Convert Env to factor
all_blues$Env <- as.factor(all_blues$Env)
cvs_h2s$Env <- as.factor(cvs_h2s$Env)

# Write results (no subdirectories)
write.csv(all_blues[, c('Env', 'Hybrid', 'predicted.value')], 'blues.csv', row.names = FALSE)
write.csv(cvs_h2s, 'cvs_h2s.csv', row.names = FALSE)
