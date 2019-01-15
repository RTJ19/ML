df <- read.csv("F:/ML/Zenrays/AI_ML_Feb25_Kormangla/hands on/credit.csv" )
df[df == "unknown"] <- NA
replacementVals<-c(checking_balance=mode(df$checking_balance))
df[is.na(df[,names(replacementVals)])]<-replacementVals

df$checking_balance <- t(apply(df$checking_balance, 2, 
                               function(x){x[is.na(x)] <- Mode(x); x}))
a <- Mode(df$checking_balance)

df$checking_balance = ifelse(is.na(df$checking_balance),
                             (function(x) {return(mode(df$checking_balance))}),
                             df$checking_balance)
ratio <- table(df$checking_balance, useNA = "no")
val <- as.data.frame (ratio/sum(ratio))
val <- val[-4,]
val
valc <- as.list(val[,1])
valc
