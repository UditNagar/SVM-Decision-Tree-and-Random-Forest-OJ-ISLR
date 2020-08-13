
# # OJ DATA INTRODUCTION
# 
# ### The OJ data from the ISLR package. This data contain 1070 purchases 
# ### information to study which orange juice a customer would buy. The 
# ### Purchase variable is a factor with levels CH and MM indicating whether 
# ### the customer purchased Citrus Hill or Minute Maid Orange Juice. 
# ### 17 features of the customers and products are recorded. The details 
# ### of this dataset can be found in https://rdrr.io/cran/ISLR/man/OJ.html.

# In[ ]:


# This R environment comes with many helpful analytics packages installed
# It is defined by the kaggle/rstats Docker image: https://github.com/kaggle/docker-rstats
# For example, here's a helpful package to load

library(tidyverse) # metapackage of all tidyverse packages

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

list.files(path = "../input")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

library(ISLR)
library(caret)
library(e1071)
library(tree)
library(pROC)
library(randomForest)
library(ROCR)


# ## FUNCTION TO CREATE ROC PLOTS

# In[ ]:


rocplot =function (pred , truth , order,...){
  predob = prediction (pred , truth,label.ordering = order)
  perf = performance (predob,"tpr", "fpr")
  plot(perf,...)}


# # PRELIMINARY DATA ANALYSIS
# 
# Loading the OJ Data
# 

# In[ ]:



data(OJ)
attach(OJ)
head(OJ) #The first 6 rows


# # Preliminary - Creating train and test variables
# 
# For our analysis we are creating a train-test split of 70/30 from the OJ data.
# 
# Also, the costs involved to create SVM models in the analysis below are 0.01,0.1,1,10.
# 
# The Gamma value is considered to be default for Radial SVM i.e. 0.25.
# 
# The degree value is considered to be 2 for Polynomial SVM.

# In[ ]:


set.seed(451)
trainIndex <- createDataPartition(OJ$Purchase, p=0.7,list=FALSE, times = 1)

#Setting Cost for Tuning in a variable
cost<-c(0.01,0.1,1,10) 
#DEGREE FOR POLYNOMIAL KERNEL
degree<-2 
#DEFAULT GAMMA VALUE
gamma<-0.25 

train<-OJ[trainIndex,] #TRAINING SET
test<-OJ[-trainIndex,] #TEST SET


# # Support Vector Machine

# ### SVM Classifier with Linear Kernel
# 
# Caret Library is used to compute the support vector classifier for the Linear Kernel with repeated 
# cross validation.

# In[ ]:


#***************************SETTING FIT CONTROL FOR SVM*************************

#USING CARET TO COMPUTE SVC FOR LINEAR KERNEL

fitControl = trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          classProbs = TRUE,
                          savePredictions = TRUE)


# In[ ]:


#***************FITTING SVC AND COMPUTATION OF TEST ERROR RATE**************

grid1 = expand.grid(C=cost) #Changing the cost into a grid

set.seed(452)

LinearSVM = train(Purchase ~ .,
                  data = train,
                  method = "svmLinear",
                  trControl = fitControl,
                  tuneGrid = grid1,
                  preProcess = c("center","scale"))

summary(LinearSVM) #Summary
LinearSVM #Displaying the results to show Accuracy


# In[ ]:


#COMPUTING THE TEST ERROR RATE FOR Q1 AFTER TUNING
LinPredResult = predict(LinearSVM,test,decison.values=TRUE)
TestErrorRate1 = mean(LinPredResult != OJ[-trainIndex,1])

cat("The Error rate of this model is: ",  TestErrorRate1)


# The test error rate was found to be <b>0.15</b> for the support vector classifier using the tuned value of 
# the cost. The most optimal cost was found to be <b>1</b>.

# ### SVM Classifier with Radial Kernel
# 

# In[ ]:


#******FITTING SVM RADIAL KERNEL & COMPUTATION OF TEST ERROR RATE*********

grid2 = expand.grid(sigma=gamma,C=cost) #Changing the cost into a grid

#Default Gamma Value = 0.25

set.seed(453)

RadialSVM = train(Purchase ~ .,
                  data = train,
                  method = "svmRadial",
                  trControl = fitControl,
                  tuneGrid = grid2,
                  preProcess = c("center","scale"))

summary(RadialSVM) #Summary
RadialSVM #Displaying the results to show Accuracy


# In[ ]:


#COMPUTING THE TEST ERROR RATE FOR Q2 AFTER TUNING
RadPredResult = predict(RadialSVM,test,decison.values=TRUE)
TestErrorRate2 = mean(RadPredResult != OJ[-trainIndex,1])

cat("The Error rate of this model is: ",  TestErrorRate2)


# The test error rate was found to be <b>0.1625</b> for the support vector classifier using the tuned value 
# of the cost. The most optimal cost was found to be <b>1</b>.

# ### SVM Classifier with Polynomial Kernel
# 

# In[ ]:


grid3 = expand.grid(degree=degree,C=cost,scale=1) #Changing the cost into a grid

#Default Gamma Value = 0.25

set.seed(453)

PolynomialSVM = train(Purchase ~ .,
                      data = train,
                      method = "svmPoly",
                      trControl = fitControl,
                      tuneGrid = grid3,
                      preProcess = c("center","scale"))

summary(PolynomialSVM) #Summary
PolynomialSVM #Displaying the results to show Accuracy


# In[ ]:


#COMPUTING THE TEST ERROR RATE FOR Q3 AFTER TUNING
PolPredResult = predict(PolynomialSVM,test,decison.values=TRUE)
TestErrorRate3 = mean(PolPredResult != OJ[-trainIndex,1])
cat("The Error rate of this model is: ",  TestErrorRate3)


# The test error rate was found to be <b>0.15625</b> for the support vector classifier using the tuned value 
# of the cost. The most optimal cost was found to be <b>0.01</b>.

# #### ROC PLOTS FOR THE SVM MODELS

# In[ ]:


#TEST PREDICTIONS : PREDICTING PROBABILITY VALUES
testPred1 = predict(LinearSVM,test,type="prob")
testPred2 = predict(RadialSVM,test,type="prob")
testPred3 = predict(PolynomialSVM,test,type="prob")

#USING ROC FUNCTION FROM ROCR PACKAGE.
LinearRoc <- roc(response = test$Purchase, predictor = testPred1[,2])
RadialRoc <- roc(response = test$Purchase, predictor = testPred2[,2])
PolynomialRoc <- roc(response = test$Purchase, predictor = testPred3[,2])

#Area Under the curve for ROC curves
LinearAuc <- LinearRoc$auc
RadialAuc <- RadialRoc$auc
PolynomialAuc <- PolynomialRoc$auc

plot(LinearRoc,col = c("red"), main="SVM ROC")
plot(RadialRoc,col = c("blue"), add=TRUE)
plot(PolynomialRoc,col = c("green"), add=TRUE)

legend(0.2, 0.1, legend = c("Linear","Radial","Polynomial"), 
       lty = c(1), col = c("red","blue","green"),cex=0.75)


# The lowest misclassification test error comes from a model with a Linear Kernel i.e. 0.15. Polynomial model 
# comes very close to the linear model with a slightly higher test error rate of 0.15625.

# In[ ]:


cat("The Linear Model's Area under the Curve is: ", LinearAuc, "\n")
cat("The Radial Model's Area under the Curve is: ", RadialAuc, "\n")
cat("The Polynomial Model's Area under the Curve is: ", PolynomialAuc)


# As we know, the higher the value for the AUC, the better the classifier. As suggested by the previous analysis,
# linear kernel had the lowest misclassification test error and a better AUC than the other kernels adds to the 
# statement.

# # Decision Tree
# 
# For the following decision tree, we will use Cross Validation to determine the optimal size of the tree. From
# the CV, if we can prune the tree, then we will continue our analysis withe the pruned tree. But if we can't 
# prune the tree, then we will create a pruned tree with 5 terminal nodes.

# In[ ]:


set.seed(454)

#SETTING UP THE TREE PRIOR TO ITS CROSS VALIDATION
OJtree = tree(Purchase ~ ., train)
summary(OJtree)

#PLOTTING THE TREE
plot(OJtree)
text(OJtree,pretty=1,cex=0.85)


# The Decision Tree's model for OJ's dataset was found to have 8 terminal nodes before it's pruning with a 
# misclassification rate of <b>0.1827</b>.

# In[ ]:


#Decision Tree Error Prior to Cross Validation
DTError = mean(predict(OJtree, test, type='class')!=test$Purchase)

#Applying Cross Validation to proceed for Pruning
OJcv = cv.tree(OJtree,FUN=prune.misclass)
OJcv


# In[ ]:


#Plotting Cross Validation Error Rate vs Number of Leaves

plot(OJcv$size, OJcv$dev, type = "b", xlab = "No. of Leaves", ylab = "CV Error Rate %", main="CV Error rate vs the Number of leaves")


# In[ ]:


#Plotting Cross Validation Error Rate vs Alpha (k)

plot(OJcv$k, OJcv$dev, type = "b", xlab = expression(alpha), ylab = "CV Error Rate %" , main= "CV Error rate vs the Alpha(k)")


# The cross validation results of the tree showed the lowest CV Error Rate for the tree sizes of 4 and 8 
# respectively. Hence, the best size for the pruned tree was chosen to be 4.
# 
# Since, the cross validation results suggest the pruning of the tree, we can prune the tree to a subtree of 
# size 4.

# In[ ]:


#Taking the best value as 4 since the min Dev is 153 for Sizes 4 and 8
#Hence, taking the best value as 4 seems the viable option.
prunedTree = prune.misclass(OJtree,best = 4)

#Plotting the Pruned Tree
plot(prunedTree)
text(prunedTree,pretty=1,cex=0.8)


# In[ ]:


#COMPUTING THE TEST ERROR RATE FOR THE PRUNED TREE
predDTPrune = predict(prunedTree, test)
testErrDT = mean(predict(prunedTree, test, type='class')!=test$Purchase)
cat("The Error rate for this model is: ", testErrDT)


# <b><i>Hence, we see the decision tree pruned from 8 terminal nodes to 4 terminal nodes. The Test Error Rate 
# for this Pruned Tree was found to be 0.16875.</i></b>

# # Random Forest
# 
# For our analysis, we will be fitting a random forest to the training data with the following mtry values: 
# 1, 2, 3, 4, 5 and 6. Also, we will be creating a plot showing variable importance for the model with the 
# best test error.

# In[ ]:


#Adding MTRY values from 1:6 in a grid for tuning
grid6 = expand.grid(.mtry=c(1:6))
set.seed(455)

#Fitting the Model
OJrf=train(Purchase ~ .,
           data=train,
           method="rf",
           metric="Accuracy",
           trControl=fitControl,
           tuneGrid=grid6)

#Random Forest Results
OJrf


# Random Forest methodology was applied on the OJ Data with mtry values ranging from 1 to 6. After the fit, 
# the best mtry value which was used for the model was <b>6</b> with the highest accuracy rate of <b>0.788</b>, 
# hence the lowest error rate of <b>0.211</b>. Since Error rate = 1 - Accuracy.
# 
# The plot of the model which compares the accuracy after cross-validation (on the y-axis) with the randomly 
# selected predictors (on the x-axis) is:

# In[ ]:


#Plotting Random Forest Results
plot(OJrf)


# In[ ]:


#Final Model for the Random Forest
OJrf$finalModel


# The variable importance plot showed that the most important variable in the random forest model 
# of the OJ data is the column "LoyalCH" with an overall of 100, while the least important 
# column is "DiscCH" with an overall of 0. The plot is:

# In[ ]:


#Plotting Variable Importance
varImp(OJrf)
plot(varImp(OJrf), main="Variable Importance of OJ Data")


# In[ ]:


#COMPUTING THE TEST ERROR RATE FOR Q6 AFTER TUNING
RfPredResult = predict(OJrf,test)
RfPredProbs = predict(OJrf,test,type="prob")
testErrRF = mean(RfPredResult != test$Purchase)
cat("The Error rate for the model is: ",testErrRF)


# In[ ]:


#*******************************ROC PLOTS FOR DT AND RF*************************

#Area Under the Curves for ROC

RFROC <- roc(response = test$Purchase, predictor = RfPredProbs[,2])
RFDT <- roc(response = test$Purchase, predictor = predDTPrune[,2])


#RANDOM FOREST ROC
rocplot(RfPredProbs[,2],test[,1],order=c("CH","MM"),col="black",lwd=2,cex.lab=1.5,cex.axis=1.5,main="Test set")
#DECISION TREE ROC
rocplot(predDTPrune[,2],test[,1],order=c("CH","MM"),col="blue",lwd=2,cex.lab=1.5,cex.axis=1.5,add=TRUE)

legend("bottomright",
       c(expression(paste("Decision Tree")),
         expression(paste("Random Forest"))),
       cex=0.75,
       lty=1,
       lwd=2)


# The area under the curve for the random forest approach on the OJ data was found to be 0.903 
# and the area under the curve for the decision tree analysis on the OJ data was found to be 0.858. 
# As we know, the higher the area under the curve parameter, the better the classifier. Here we can see 
# that the Random Forest Classifier gave us better results than the Decision Tree classifier which also is 
# established by the fact that Random Forest models have a lower variance than the Decision Tree models. 
# The ROC plots for the Decision Tree Classifier and the Random Forest Classifier at MTRY of 6 are plotted
# above.

# In[ ]:


#AUC Values

cat("The AUC for Decision Tree: ", RFDT$auc,"\n")
cat("The AUC for Random Forest: ", RFROC$auc, "\n")

