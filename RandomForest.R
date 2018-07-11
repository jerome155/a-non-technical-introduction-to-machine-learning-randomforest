install.packages('data.tree')

IsPure <- function(data) {
  length(unique(data[,1])) == 1
}

InformationGain_Numeric <- function(trainData) {
  EntropyChildNodes <- data.frame()
  EntropyChildNodes <- matrix(nrow=nrow(trainData)-1, ncol=ncol(trainData)-1)
  colnames(EntropyChildNodes) <- colnames(trainData[,2:ncol(trainData)])
  
  ParentNodeProbability <- sum(x=trainData[,1])/nrow(trainData)
  ParentNodeEntropy <- -1*(ParentNodeProbability * log2(ParentNodeProbability) + 
                             (1-ParentNodeProbability) * log2(1-ParentNodeProbability))
  
  for (i in 2 : ncol(trainData)) {
    trainData <- trainData[order(trainData[,i]),]
    for (j in 1:(nrow(trainData)-1)) {
      if (trainData[j, i] == trainData[j+1, i]) {
        EntropyChildNodes[j, i-1] <- NA
      } else {
        subsetProbabilityAbove <- sum(x=trainData[1:j, 1])/j
        subsetProbabilityBelow <- sum(x=trainData[(j+1):(nrow(trainData)), 1])/(nrow(trainData)-j)
        EntropyChildNode1 <- -1*(subsetProbabilityAbove * log2(subsetProbabilityAbove) + 
                                   (1-subsetProbabilityAbove) * log2(1-subsetProbabilityAbove))
        if (is.nan(EntropyChildNode1)) {
          EntropyChildNode1 <- 0
        }
        EntropyChildNode2 <- -1*(subsetProbabilityBelow * log2(subsetProbabilityBelow) + 
                                   (1-subsetProbabilityBelow) * log2(1-subsetProbabilityBelow))
        if (is.nan(EntropyChildNode2)) {
          EntropyChildNode2 <- 0
        }
        EntropyChildNodes[j, i-1] <- ParentNodeEntropy - (j/nrow(trainData)*EntropyChildNode1 + 
                                                            (nrow(trainData)-j)/nrow(trainData)*EntropyChildNode2)
      }
    }
  }
  EntropyChildNodes[is.na(EntropyChildNodes)] <- 0
  EntropyChildNodes
}

TrainID3 <- function(node, data) {
  
  node$obsCount <- nrow(data)
  
  if (IsPure(data)) {
    child <- node$AddChild(unique(data[,1]))
    node$feature <- tail(names(data), 1)
    child$obsCount <- nrow(data)
    child$feature <- ''
    child$value <- 0
    
  } else {
    ig <- InformationGain_Numeric(data)
    igFeatureColumn <- which.max(apply(ig, 2, which.max))
    
    if (length(which(ig[,igFeatureColumn]==max(ig[,igFeatureColumn])))>1) {
      igFeatureColumnSplitPosition <- sample(which(ig[,igFeatureColumn]==max(ig[,igFeatureColumn])),1)
    } else {
      igFeatureColumnSplitPosition <- which.max(ig[,igFeatureColumn])
    }
    
    data <- data[order(data[,igFeatureColumn+1]),]
    
    node$feature <- colnames(data)[igFeatureColumn+1]
    node$value <- (data[igFeatureColumnSplitPosition, igFeatureColumn+1] + 
                     data[igFeatureColumnSplitPosition+1, igFeatureColumn+1])/2
    
    childObs1 <- data[1:igFeatureColumnSplitPosition,];
    childObs2 <- data[(igFeatureColumnSplitPosition+1):nrow(data),]
    childObs <- list(childObs1, childObs2)
    
    names(childObs) <- c(paste0(node$feature, "<=", (data[igFeatureColumnSplitPosition, igFeatureColumn+1] + 
                                                       data[igFeatureColumnSplitPosition+1, igFeatureColumn+1])/2),
                         paste0(node$feature, ">", (data[igFeatureColumnSplitPosition, igFeatureColumn+1] + 
                                                      data[igFeatureColumnSplitPosition+1, igFeatureColumn+1])/2))
    
    for(i in 1:length(childObs)) {
      child <- node$AddChild(names(childObs)[i])
      
      TrainID3(child, childObs[[i]])
    }
  }
}

PredictID3 <- function(tree, features) {
  if (tree$children[[1]]$isLeaf) return (tree$children[[1]]$name)
  if (features[,tree$feature]<=tree$value) {
    child <- tree$children[[1]]
  } 
  else {
    child <- tree$children[[2]]
  }
  return(PredictID3(child, features))
}

TrainRandomForest <- function(data, numberOfTrees, numberOfFeatures, numberOfObservations, seed) {
  #Create empty variables we require later to store the data into
  treeData <- list()
  treesOut <- list()
  
  #Repeat this step for the number of trees required.
  for(i in 1:numberOfTrees) {
    #Set the random seed + 1. With this we ensure that we get different trees 
    #for every iteration of the algorithm.
    seed <- seed + 1
    set.seed(seed)
    
    #Prepare the data variable for the newly shuffled data
    tempData <- data.frame()
    
    #Step 2: From the original data, sample from BOTH features new datapoints, with replacement.
    tempData <- data[sample(nrow(data), numberOfObservations, TRUE),]
    #Store the column with the output variable (Y) separately, we loose it in the next step.
    tempDataYCol <- tempData[, "Y"]
    #Step 3: Select at random one feature of the available ones.
    tempData <- sample(tempData[,2:ncol(tempData)], numberOfFeatures, TRUE)
    #Add back the Y column that we just lost.
    tempData <- cbind("Y"=tempDataYCol,tempData)
    
    #Store this data into the prepared lists containing data for the trees / the actual trees.
    treeData[[i]] <- tempData
    treesOut[[i]] <- Node$new(paste0("MachineLearningExample", i))
  }
  #Train every tree based on the sampled data.
  for(i in 1:length(treesOut)) {
    TrainID3(treesOut[[i]], treeData[[i]])
  }
  #Return the finished trees.
  treesOut
}

PredictRandomForest <- function(treesIn, features) {
  #Prepare an empty vector into which the prediction of every tree is stored.
  resultVectorOut <- vector()
  #For every tree available, predict the result and store it.
  for(i in 1:length(treesIn)) {
    resultVectorOut[i] <- PredictID3(treesIn[[i]], features)
  }
  resultVectorOut
}

MajorityVote <- function(resultVector) {
  #Sum up the number of Zero's and One's into a small table.
  table(resultVector)
}

library(data.tree)

#Input data from the slides
trainData <- data.frame(index = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
                        Y = c(0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L, 0L),
                        X1 = c(2, 2.8, 1.5, 2.1, 5.5, 8, 6.9, 8.5, 2.5, 7.7),
                        X2 = c(1.5, 1.2, 1, 1, 4, 4.8, 4.5, 5.5, 2, 3.5))

#The index is not necessary for this computation, therefore remove it.
trainData <- trainData[,c('Y', 'X1', 'X2')]
#Create a new, empty tree.
treesOut <- TrainRandomForest(trainData, 99, 1, 10, 12345)
treesOut

#Trying to predict two new sets of points.
predictionValue <- data.frame(X1=3.19, X2=5)
resultVector <- PredictRandomForest(treesOut, predictionValue)
MajorityVote(resultVector)

predictionValue2 <- data.frame(X1=0, X2=1.6)
resultVector2 <- PredictRandomForest(treesOut, predictionValue2)
MajorityVote(resultVector2)