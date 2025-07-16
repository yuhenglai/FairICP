gamma_likeloglik_eval = function(y, Aactive, w, sigma2){
 mu = apply(Aactive, 1, sum) *sqrt(w)
 sigma2_y = sigma2+(1-w) *ncol(Aactive)
 return( sum((y - mu)^2/(2*sigma2_y)))
}

dat_gen = function(n,K0,K, w, sigma2, u, D){
  Ktotal = K0+K
  tmp1 = matrix(rgamma(n = n*(Ktotal),1,10), ncol = Ktotal)
  #tmp2 = matrix(rnorm(n = n*(Ktotal)), ncol = Ktotal)
  tmp2 = matrix(rgamma(n = n*(Ktotal),1,1), ncol = Ktotal)
  I = matrix(rbinom (n = n*(Ktotal),size = 1, prob = 0.5), ncol = Ktotal)
  A = matrix(0, ncol = Ktotal, nrow = n)
  for(i in 1:n){
    for(j in 1:Ktotal){
      if(I[i,j] == 0){
        A[i,j] = tmp1[i,j]
      }else{
        A[i,j] = tmp2[i,j]
      }

    }
  }
  if(Ktotal > 1){
    A = A%*%D%*%u
  }
  # if(K>0){
  #   A[,(K0+1):min(K+K0,2*K0)] = A[,(K0+1):min(K+K0,2*K0)]/2+A[,1:min(K0,K)]/2
  # }
  
  y = sqrt(w) *apply(A[,1:K0,drop = F],1,sum)+rnorm(n, mean = 0,sd = sqrt(sigma2+(1-w) *K0))
  return(list(y = y, A=A))
}

library(reticulate)
library(latex2exp)
source_python("./MAF_demo/MAF_demo_pytorch.py")
set.seed(2024)
K0s = c(1, 5, 10)
Ks = c(0, 5, 10, 20, 50, 100)
n = 200; w = 0.6; nte = 200; sigma2.truth = 1;B = 10000
Iteration = 20;eigen_ratio=5
result_collection = list()
for(k in 1:length(K0s)){
  result_collection[[k]] = list()
  for(l in 1:length(Ks)){
    result_collection[[k]][[l]] = array(NA, dim = c(B,2,4,Iteration)) 
  }
}
pairwise_swappings = matrix(0, nrow = n*(n-1)/2, ncol = 2)
ll=0
for(i in 1:(n-1)){
  for(j in (i+1):n){
    ll = ll+1
    pairwise_swappings[ll,1] = i; pairwise_swappings[ll,2] = j
    
  }
}
for(k in (length(K0s)):(length(K0s))){
  K0 = K0s[k] 
  for(l in (length(Ks)-1):(length(Ks)-1)){
    K = Ks[l]
    for(it in 1:Iteration){
      print(paste0("K0=",K0, ",K=",K,",it=", it))
      Ktotal = K0 +K
      u = matrix(rnorm(Ktotal*Ktotal), ncol =Ktotal)
      D = sqrt(diag(seq(1, eigen_ratio, length.out = Ktotal)))
      dat.tr = dat_gen(n = n, K0 = K0, K = K, w = w, sigma2 = sigma2.truth,u = u, D =D)
      dat.te = dat_gen(n = nte, K0 = K0, K = K, w = w, sigma2 = sigma2.truth,u = u, D =D)
      y = dat.tr$y; A = dat.tr$A
      y.te = dat.te$y; A.te = dat.te$A 
      ###lasso
      if(ncol(A)>1){
        tmp1 = glmnet::cv.glmnet(x=A, y=y)
        idx = predict(tmp1, type = "nonzero", s = "lambda.1se")
        tmp1 = lm(y~A[,idx[,1],drop = F])
        theta = rep(0, ncol(A)+1)
        theta[c(1,idx[,1]+1)] = tmp1$coefficients
        sigma2 = sum(tmp1$residuals^2)/(n-length(idx)-1)
      }else{
        tmp1 = lm(y~A)
        theta = tmp1$coefficients
        sigma2 = sum(tmp1$residuals^2)/(n-1-1)
      }

      ###MAF
      log_lik_mat = MAF_density_estimation_pytorch(np_array(y), np_array(A), np_array(y.te), np_array(A.te))
      
      #graphical lasso
      require(CVglasso)
      if(Ktotal > 1){
        tmp2 = CVglasso(X =cbind(y, A),  nlam = 100, crit.cv ="BIC")
        Omega = tmp2$Omega
        Sigmahat = tmp2$Sigma
        Asigma2 = Sigmahat[-1,-1]
        beta = matrix(0, ncol = ncol(A), nrow = 2)
        beta[2,] = Sigmahat[-1,1,drop = F]%*%solve(Sigmahat[1,1])
        
        for(j in 1:ncol(A)){
          beta[1,j] = mean(A[,j]) - beta[2,j] * mean(y)
        }
        Asigma2= Sigmahat[-1,-1] - Sigmahat[-1,1,drop = F]%*%solve(Sigmahat[1,1])%*%Sigmahat[1,-1,drop = F]
        
      }else{
        tmp1 = lm(A~y)
        beta = matrix(tmp1$coefficients,ncol=1)
        Asigma2 = sum(tmp1$residuals^2)/(n-1-1)
      }

      ###MAF
      log_lik_mat2 = MAF_density_estimation_pytorch(np_array(A), np_array(y), np_array(A.te), np_array(y.te))

      Asigma2.inv = solve(Asigma2)
      loglik_diff_ICP = matrix(0, ncol = 2, nrow = B)
      loglik_diff_CP =  matrix(0, ncol = 2, nrow = B)
      
      loglik_diff_MAF_CP =  matrix(0, ncol = 2, nrow = B)
      loglik_diff_MAF_ICP =  matrix(0, ncol = 2, nrow = B)
      
      for(b in 1:B){
        ii0 = pairwise_swappings[b,]
        ii = ii0[c(2,1)]
        loglik_diff_ICP[b,1]=gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_ICP[b,1]=loglik_diff_ICP[b,1]-gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii0,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        r.0 = (y.te[ii0] -theta[1]- A.te[ii0,,drop = F]%*%theta[-1])
        r =(y.te[ii0] -theta[1]- A.te[ii,,drop = F]%*%theta[-1])
        loglik_diff_ICP[b,2] = (sum(r^2)-sum(r.0^2))/(2*sigma2)
        
        loglik_diff_MAF_ICP[b,1]=gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_MAF_ICP[b,1]=loglik_diff_MAF_ICP[b,1]-gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii0,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_MAF_ICP[b,2] = sum(diag(log_lik_mat[ii0, ii0])) - sum(diag(log_lik_mat[ii0, ii]))
                
        loglik_diff_CP[b,1]=gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_CP[b,1]=loglik_diff_CP[b,1]-gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii0,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        r =A.te[ii,] - y.te[ii0]%*%beta[2,,drop = F]; r = sapply(1:ncol(A.te), function(j) r[,j] - beta[1,j])
        r.0 = A.te[ii0,] - y.te[ii0]%*%beta[2,,drop = F]; r.0 = sapply(1:ncol(A.te), function(j) r.0[,j] - beta[1,j])
        loglik_diff_CP[b,2] = sum(sapply(1:nrow(r), function(i) {0.5*r[i,,drop = F]%*%Asigma2.inv%*%t(r[i,,drop = F])}))
        loglik_diff_CP[b,2] = loglik_diff_CP[b,2]-sum(sapply(1:nrow(r), function(i) {0.5*r.0[i,,drop = F]%*%Asigma2.inv%*%t(r.0[i,,drop = F])}))
        
        loglik_diff_MAF_CP[b,1]=gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_MAF_CP[b,1]=loglik_diff_MAF_CP[b,1]-gamma_likeloglik_eval(y=y.te[ii0],Aactive =  A.te[ii0,1:K0, drop = F], w=w, sigma2=sigma2.truth)
        loglik_diff_MAF_CP[b,2] = sum(diag(log_lik_mat2[ii0, ii0])) - sum(diag(log_lik_mat2[ii, ii0]))
      
      }
      
      result_collection[[k]][[l]][,,1,it] = loglik_diff_CP
      result_collection[[k]][[l]][,,2,it] =  loglik_diff_ICP
      result_collection[[k]][[l]][,,3,it] = loglik_diff_MAF_CP
      result_collection[[k]][[l]][,,4,it] =  loglik_diff_MAF_ICP
      
      r0 = c(1,exp(-loglik_diff_CP[,1])); r0 = r0/sum(r0)
      r1 = c(1,exp(-loglik_diff_CP[,2])); r1 = r1/sum(r1)
      print(mean(abs(r0-r1)))
      
      
      r0 = c(1,exp(-loglik_diff_ICP[,1])); r0 = r0/sum(r0)
      r1 = c(1,exp(-loglik_diff_ICP[,2])); r1 = r1/sum(r1)
      print(mean(abs(r0-r1)))
      
      r0 = c(1,exp(-loglik_diff_MAF_CP[,1])); r0 = r0/sum(r0)
      r1 = c(1,exp(-loglik_diff_MAF_CP[,2])); r1 = r1/sum(r1)
      print(mean(abs(r0-r1)))


      r0 = c(1,exp(-loglik_diff_MAF_ICP[,1])); r0 = r0/sum(r0)
      r1 = c(1,exp(-loglik_diff_MAF_ICP[,2])); r1 = r1/sum(r1)
      print(mean(abs(r0-r1)))
    
    }

    
  }
}

result_summaryI = array(NA, dim = c(Iteration, 4, length(K0s), length(Ks)))  

for(k in 1:length(K0s)){
  for(l in 1:length(Ks)){
    for(it in 1:Iteration){
      tmp1 = result_collection[[k]][[l]][,,1,it]
      r0 = c(1,exp(-tmp1[,1])); r0 = r0/sum(r0)
      r1 =c(1,exp(-tmp1[,2])); r1 = r1/sum(r1)
      result_summaryI[it,1,k, l] = mean(abs(r0-r1))
      
      tmp1 = result_collection[[k]][[l]][,,2,it]
      r0 = c(1,exp(-tmp1[,1])); r0 = r0/sum(r0)
      r1 =c(1,exp(-tmp1[,2])); r1 = r1/sum(r1)
      result_summaryI[it,2,k, l] = mean(abs(r0-r1))
      
      tmp1 = result_collection[[k]][[l]][,,3,it]
      r0 = c(1,exp(-tmp1[,1])); r0 = r0/sum(r0)
      r1 =c(1,exp(-tmp1[,2])); r1 = r1/sum(r1)
      result_summaryI[it,3,k, l] = mean(abs(r0-r1))

      tmp1 = result_collection[[k]][[l]][,,4,it]
      r0 = c(1,exp(-tmp1[,1])); r0 = r0/sum(r0)
      r1 =c(1,exp(-tmp1[,2])); r1 = r1/sum(r1)
      result_summaryI[it,4,k, l] = mean(abs(r0-r1))
    }
  }
}

plotDF = NULL
for(k in 1:length(K0s)){
  for(l in 1:length(Ks)){
    tmp = result_summaryI[,c(1, 2, 3, 4),k,l]
    tmp = data.frame(tmp)
    colnames(tmp) = c("CP_graphicalLasso", "ICP_lasso", "CP_MAF", "ICP_MAF") 
    tmp$it = c(1:Iteration)
    tmp$K0 = K0s[k]
    tmp$K = Ks[l]
    plotDF =rbind(plotDF, tmp)
  }
}
plotDF$CP = log(plotDF$CP, base = 10)
plotDF$ICP = log(plotDF$ICP, base = 10)
plotDF$CP_MAF = log(plotDF$CP_MAF, base = 10)
plotDF$ICP_MAF = log(plotDF$ICP_MAF, base = 10)
require(ggplot2)
require(dplyr)
require(tidyverse)

plotDF = gather(plotDF, key = "method", value = "logTV", -it, -K0, -K)

plt1 = ggplot(plotDF, aes(x = factor(K), y = logTV, color = method)) +
  geom_boxplot(outlier.shape = NA) +  # Add boxplot
  geom_jitter(aes(group = method), width = 0.2) +  # Add jitter to show individual points
  facet_wrap(~ K0, labeller = labeller(K0 = c(`1` = "K0=1", `5` = "K0 = 5", `10` = "K0 = 10"))) +  # Separate panels for CP and ICP
  theme_minimal() +  # Minimal theme
  labs(title="",x = "K", y = "Restricted TV") +  # Labels
  theme(axis.text.x = element_text(angle = 0, hjust = 1))  # Improve x-axis labels visibility