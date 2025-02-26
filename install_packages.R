options(repos = c(CRAN = "https://cran.rstudio.com"))
.libPaths("C:/Users/wgatl/R/library")

install.packages("arrow", lib = "C:/Users/wgatl/R/library")
install.packages("data.table", lib = "C:/Users/wgatl/R/library")
install.packages("AGHmatrix", lib = "C:/Users/wgatl/R/library")
install.packages("devtools", lib = "C:/Users/wgatl/R/library")
install.packages("asreml", lib = "C:/Users/wgatl/R/library")  # for BLUEs and FA

# Install package from GitHub
setRepositories(ind = 1:2)  # Ensure GitHub dependencies are accessible
devtools::install_github("samuelbfernandes/simplePHENOTYPES")

library(arrow) 
installed.packages()[, "Package"]