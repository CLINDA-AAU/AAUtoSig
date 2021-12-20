library(MutationalPatterns)
library("NMF")
library(xlsx)


train_data <- read.csv("Q:/AUH-HAEM-FORSK-MutSigDLBCL222/article_1/generated_data/DLBCL1001_trainset1_80p.csv")
validation_data <- read.csv("Q:/AUH-HAEM-FORSK-MutSigDLBCL222/article_1/generated_data/DLBCL1001_testset1_20p.csv")

n_mut <- nrow(train_data)

estimate <- nmf(train_data[,2:802], rank = 2:12, method = "brunet", 
                nrun = 10, seed = 123456, .opt = "v-p")
plot(estimate)



out_error <- function(train_data ,val_data){
  nmf_res <- NMF::nmf(train_data[,2:802], rank = 5, method = "brunet", nrun = 1)
  
  n_pat_val <- ncol(validation_data)-1
  
  fit_res <- fit_to_signatures(validation_data[,2:201], basis(nmf_res))

  return(sum((fit_res$reconstructed - validation_data[,2:201])^2)/(n_pat_val*96))
}

asd <- replicate(1000, out_error(train_data, validation_data))

write.table(asd_df,"Q:/AUH-HAEM-FORSK-MutSigDLBCL222/article_1/generated_data/out_error_NMF1000.csv", row.names = F)
