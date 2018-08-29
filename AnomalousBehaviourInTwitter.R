##############################################################
# TWITTER CASE STUDY  ###
##############################################################

#Business Understanding
#Data Preparation & EDA
#Model Building 
#Model Evaluation
################################################################

### Business Understanding:

# Based on the tweet data information,  we have to predict whether a twitter account is compromised or not and to identify bots

## Goal:

# Automate the process and predict whether twitter account is compromised or not, identify bots and find the factors affecting 

################################################################
library(dplyr)
library(ggplot2)
library(neuralnet)
library(MASS)
library(ggplot2)
library(cowplot)
library(ggthemes)
library(caTools)
library(e1071)
library(ROCR)
library(dplyr)
library(GGally)
library(prettyunits)
library(chron)
library(lubridate)
library(tidyr)
library(cluster)
library(plotly)
library(caTools)

#####################################################################
#       I. EDA                                                      #
#####################################################################
###### DATA UNDERSTANDING, PREPARATION & DATA ANALYSIS ##############
startTime <- Sys.time()

tweets <- read.csv("tweets.csv",stringsAsFactors = FALSE)
finalResult1 <- tweets[!duplicated(tweets$user_id), ]

# structure of tweets dataframes
str(tweets)        # 140640 obs. of  14 variables
# summary of tweets dataframes
summary(tweets)

sum(duplicated(tweets$id))  # id is unique in the whole data

#removing the rows which contains NA's and empty values
tweets <- tweets[rowSums(is.na(tweets)) != ncol(tweets),]
tweets <- tweets[!apply(tweets == "", 1, all),]

####### Missing values (NA's) or Empty data handling 

sum(is.na(tweets))  # 0 NA values so no need to do anything
sum(tweets=="") # 0 empty data in columns

# Removing the Redundant Columns
#tweets <- tweets[,-7]

#Data Preprocessing - which will look for any null/Empty values for mandatory columns and removes the corresponding row

tweets <- tweets[rowSums(is.na(tweets)) != ncol(tweets),]
tweets <- tweets[!apply(tweets == "", 1, all),]

if(any(is.na(tweets[, "user_created_at"]))){
  tweets <- tweets[!(is.na(tweets$created_at) | tweets$created_at == ""), ]
}

if(any(is.na(tweets[, "favorite_count"]))){
  tweets <- tweets[!(is.na(tweets$favorite_count) | tweets$favorite_count == ""), ]
}

if(any(is.na(tweets[, "followers_count"]))){
  tweets <- tweets[!(is.na(tweets$followers_count) | tweets$followers_count==""), ]
}

if(any(is.na(tweets[, "friends_count"]))){
  tweets <- tweets[!(is.na(tweets$friends_count) | tweets$friends_count=="" | tweets$friends_count == 0), ]
}

if(any(is.na(tweets[, "user_created_at"]))){
  tweets <- tweets[!(is.na(tweets$user_created_at) | tweets$user_created_at==""), ]
}

if(any(is.na(tweets[, "lang"]))){
  tweets <- tweets[!(is.na(tweets$lang) | tweets$lang==""), ]
}

if(any(is.na(tweets[, "source"]))){
  tweets <- tweets[!(is.na(tweets$lang) | tweets$lang==""), ]
}

#########################Data Preprocessing Ends###############################


#########################Social Bots Detection#################################

#Utility function returns the difference in days between two date objects
# @param1 - date object one
# @param2 - date object two
# @return - differnce between two date objects in terms of days
dateDifference <- function(date1, date2=Sys.Date()){
  as.numeric(difftime(date2,date1,units="days"))
}

##### convert continous independent variables values into numeric
tweets$favorite_count <- as.numeric(tweets$favorite_count)
tweets$followers_count <- as.numeric(tweets$followers_count)
tweets$friends_count <- as.numeric(tweets$friends_count)

# Derived columns
processeduserLongevity <- mutate(tweets, userLongevity = round(dateDifference(ymd_hms(user_created_at))))
tweets <- processeduserLongevity
finalResult1 <- processeduserLongevity[!duplicated(processeduserLongevity$user_id), ]

for(i in 1:nrow(tweets)){
  tweets[i, 'langPreference'] <- sum(as.numeric(charToRaw(tweets[i, 'lang'])))
}

for(i in 1:nrow(tweets)){
  tweets[i, 'devicePreference'] <- sum(as.numeric(charToRaw(tweets[i, 'source'])))
}

#tweets$tweetCount <- as.numeric(tweets$tweetCount)
processedfriendShipRatio <- mutate(tweets, friendShipRatio = (tweets$friends_count/ tweets$userLongevity)*10)
tweets <- processedfriendShipRatio
finalResult1 <- processedfriendShipRatio[!duplicated(processedfriendShipRatio$user_id), ]

processedfavoriteCountRatio <- mutate(tweets, favoriteCountRatio = tweets$favorite_count / tweets$userLongevity)
tweets <- processedfavoriteCountRatio
finalResult1 <- processedfavoriteCountRatio[!duplicated(processedfavoriteCountRatio$user_id), ]

processedtweetsRatio <- mutate(tweets, tweetsRatio = tweets$tweets_count / tweets$userLongevity)
tweets <- processedtweetsRatio
finalResult1 <- processedtweetsRatio[!duplicated(processedtweetsRatio$user_id), ]

#Compute Twitting Period - Which duration of the day does the user sends tweet
Q1 <- 0
Q2 <- 0
Q3 <- 0
Q4 <- 0

tweetingPeriod <- sapply(tweets$user_created_at, function(tweetTime){
  sentTime <- hour(ymd_hms(tweetTime))
  
  if(as.numeric(floor(sentTime/6)) == 0){
    Q1 <- Q1+1
    return (0)
  } else if(as.numeric(floor(sentTime/6)) == 1){
    Q2 <- Q2+1
    return (1)
  } else if(as.numeric(floor(sentTime/6)) == 2){
    Q3 <- Q3+1
    return (2)
  } else{
    Q4 <- Q4+1
    return (3)
  }
})

tweets <- transform(tweets, tweetingPeriod = tweetingPeriod)
finalResult1 <- tweets[!duplicated(tweets$user_id), ]

#segregate Users who are evey much active from the rest based on followers count
#ggplot(finalResult1, aes(finalResult1$user_id, finalResult1$followers_count),xlab("User ID"), ylab("Followers Count")) + geom_point()

#Look for NA's introduced due to corresion and remove them.
sum(is.na(finalResult1))
finalResult1 <- finalResult1[complete.cases(finalResult1), ]

#Use Clustering here k-means
for(i in 1:nrow(tweets)){
  if(tweets[i, 'followers_count'] > 5000){
    tweets[i, 'active_user'] <- 1
  } else{
    tweets[i, 'active_user'] <- 0
  }
}

# Derived column Normal to check whether record is a valid or invalid, currently i am considering friendshipRatio and tweetRatio > 14 is invalid and all other is valid
# Value 0 corresponds to Normal and 1 corresponds to Anamalous behaviour
for(i in 1:nrow(tweets)){
  if((tweets[i,14]>1.5) || (tweets[i,15]>=2) || (tweets[i,16] >= 1.5)){
    tweets[i,"socialBots"] <- 0
    finalResult1[i, "socialBots"] <- 0
  }else{
    tweets[i,"socialBots"] <- 1
    finalResult1[i, "socialBots"] <- 1
  }
}

socialBots <- sum(tweets$socialBots)
normalUserCount <- nrow(tweets) - socialBots

users <- c(normalUserCount,socialBots)
temp <- data.frame(users)

#users <- data.frame("Categorie"=rownames(users), users)
userData <- temp[,c('users')]

p <- plot_ly(as.data.frame(userData), labels = ~c('Normal Users', 'Social Bots'), values = ~users, type = 'pie') %>%
  layout(title = 'Normal Users Vs Social Bots in Twitter Dataset.',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

print(p)

#########################Social Bots Detection Ends#####################################



################################Compromised Account Detection using ANN#################
neuralNetData <- tweets[, c('user_id','devicePreference','langPreference','tweetingPeriod')]

for(i in 1:nrow(neuralNetData)){
  neuralNetData[i,"featureMean"] <- floor(neuralNetData[i, 'devicePreference'] + neuralNetData[i, 'langPreference'] + neuralNetData[i, 'tweetingPeriod'] / 3)
}

avg <- mean(neuralNetData$featureMean)

for(i in 1:nrow(neuralNetData)) {
  if(avg - neuralNetData[i,"featureMean"] > 0) {
    neuralNetData[i,"normal"] <- 1
  } else {
    neuralNetData[i,"normal"] <- 0
  }
}

maxs <- apply(neuralNetData, 2, max) 
mins <- apply(neuralNetData, 2, min)

scaled <- as.data.frame(scale(neuralNetData, center = mins, scale = maxs - mins))

split = sample.split(scaled$user_id, SplitRatio = 0.70)

train = subset(scaled, split == TRUE)
test = subset(scaled, split == FALSE)

#Get the column names
n <- names(train)
f <- as.formula(paste("normal ~", paste(n[!n %in% "normal"], collapse = " + ")))

nn <- neuralnet(f,data=train,hidden=c(5,3),linear.output=TRUE)

plot(nn)

predicted.nn.values <- compute(nn,test[1:5])

str(predicted.nn.values)

# Convert back to non-scaled predictions
true.predictions <- predicted.nn.values$net.result*(max(neuralNetData$normal)-min(neuralNetData$normal))+min(neuralNetData$normal)

# Convert the test data
test.r <- (test$normal)*(max(test$normal)-min(test$normal))+min(test$normal)

# Check the Mean Squared Error
MSE.nn <- sum((test.r - true.predictions)^2)/nrow(test)

predictedValues <- floor(true.predictions)

error.df <- data.frame(test.r,predictedValues)

ggplot(error.df,aes(x=test.r,y=predictedValues)) + geom_point() + stat_smooth()

################################Compromised Account Detection using ANN Ends#################

finalResult1 <- finalResult1[complete.cases(finalResult1), ]
write.csv(tweets, "finalResult1.csv")

##########Create Cluster based on number of friends and followers####################

#pl <- ggplot(tweets, aes(tweets$user_id, tweets$followers_count, color=tweets$user_id))
#print(pl+geom_point(size=4))
# set.seed(101)
# 
# clusteredTweets <- kmeans(tweets[, 8:9], 2, nstart = 20)
# clusplot(tweets, clusteredTweets$cluster, color=TRUE, shade=TRUE, labels=0,lines=0 )
# 
# 
# ##########Univariate Analysis for categorical features ##################
# 
# bar_theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
#                    legend.position="none")
# ggplot(tweets, aes(x=tweets$source,fill=tweets$normal))+ geom_bar()  
# ggplot(tweets, aes(x=tweets$name,fill=tweets$normal))+ geom_bar()+bar_theme1
# ggplot(tweets, aes(x=tweets$screen_name,fill=tweets$normal))+ geom_bar()+bar_theme1
# ggplot(tweets, aes(x=tweets$tweet_language,fill=tweets$normal))+ geom_bar()+bar_theme1
# 
# plot_grid(ggplot(tweets, aes(x=tweets$source,fill=tweets$normal))+ geom_bar()  , 
#           ggplot(tweets, aes(x=tweets$name,fill=tweets$normal))+ geom_bar()+bar_theme1,
#           ggplot(tweets, aes(x=tweets$screen_name,fill=tweets$normal))+ geom_bar()+bar_theme1,
#           ggplot(tweets, aes(x=tweets$tweet_language,fill=tweets$normal))+ geom_bar()+bar_theme1,
#           align = "h")   
# 
# # Histogram and Boxplots for numeric variables 
# 
# box_theme<- theme(axis.line=element_blank(),axis.title=element_blank(), 
#                   axis.ticks=element_blank(), axis.text=element_blank())
# 
# box_theme_y<- theme(axis.line.y=element_blank(),axis.title.y=element_blank(), 
#                     axis.ticks.y=element_blank(), axis.text.y=element_blank(),
#                     legend.position="none")
# 
# plot_grid(ggplot(tweets, aes(tweets$favorite_count))+ geom_histogram(binwidth = 1),
#           ggplot(tweets, aes(x="",y=tweets$favorite_count))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
#           align = "v",ncol = 1)
# 
# plot_grid(ggplot(tweets, aes(tweets$tweetCount))+ geom_histogram(binwidth = 100),
#           ggplot(tweets, aes(x="",y=tweets$followers_count))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
#           align = "v",ncol = 1)
# 
# plot_grid(ggplot(tweets, aes(tweets$userLongevity))+ geom_histogram(binwidth = 100),
#           ggplot(tweets, aes(x="",y=tweets$userLongevity))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
#           align = "v",ncol = 1)
# 
# plot_grid(ggplot(tweets, aes(tweets$retweet_count))+ geom_histogram(binwidth = 100),
#           ggplot(tweets, aes(x="",y=tweets$retweet_count))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
#           align = "v",ncol = 1)
# 
# # Boxplots of numeric variables to detect outliers
# plot_grid(ggplot(tweets, aes(x=tweets$normal,y=retweet_count, fill=tweets$normal))+ geom_boxplot(width=0.2)+ 
#             coord_flip() +theme(legend.position="none"),
#           ggplot(tweets, aes(x=tweets$normal,y=userLongevity, fill=tweets$normal))+ geom_boxplot(width=0.2)+
#             coord_flip() + box_theme_y,
#           ggplot(tweets, aes(x=tweets$normal,y=tweetCount, fill=tweets$normal))+ geom_boxplot(width=0.2)+
#             coord_flip() + box_theme_y,
#           ggplot(tweets, aes(x=tweets$normal,y=friends_count, fill=tweets$normal))+ geom_boxplot(width=0.2)+
#             coord_flip() + box_theme_y,
#           ggplot(tweets, aes(x=tweets$normal,y=favorite_count, fill=tweets$normal))+ geom_boxplot(width=0.2)+
#             coord_flip() + box_theme_y,
#           ggplot(tweets, aes(x=tweets$normal,y=followers_count, fill=tweets$normal))+ geom_boxplot(width=0.2)+
#             coord_flip() + box_theme_y,
#           align = "v",nrow = 1)
# 
# ##### Outliers Check ####################
# # check for outliers in all continuous independent variables, different outlier detection methods used
# # based on business data, however following method is used to detect and remove outliers
# 
# Outlier_Retweet_count <- boxplot(tweets$retweet_count,plot = FALSE)$out
# Outlier_Retweet_count
# quantile(tweets$retweet_count,seq(0,1,.01),na.rm = T) # we have outliers here, big jump between 99% and 100%
# tweets$retweet_count[which(tweets$retweet_count > 2681.32)] <- 2681.32 # removing outlier and setting to max value 8620.96
# 
# Outlier_userLongevity <- boxplot(tweets$userLongevity,plot = FALSE)$out
# Outlier_userLongevity
# quantile(tweets$userLongevity,seq(0,1,.01),na.rm = T) # no jump, no need to handle outlier here
# 
# Outlier_tweetCount <- boxplot(tweets$tweetCount,plot = FALSE)$out
# Outlier_tweetCount
# quantile(tweets$tweetCount,seq(0,1,.01),na.rm = T) # no jump, no need to handle outlier here
# 
# Outlier_friends_count <- boxplot(tweets$friends_count,plot = FALSE)$out
# Outlier_friends_count
# quantile(tweets$friends_count,seq(0,1,.01),na.rm = T) # no jump, no need to handle outlier here
# 
# Outlier_favorite_count <- boxplot(tweets$favorite_count,plot = FALSE)$out
# Outlier_favorite_count
# quantile(tweets$favorite_count,seq(0,1,.01),na.rm = T) # we have outliers here, big jump after 98%
# tweets$favorite_count[which(tweets$favorite_count > 8620.96)] <- 8620.96 # removing outlier and setting to max value 8620.96
# 
# Outlier_followers_count <- boxplot(tweets$followers_count,plot = FALSE)$out
# Outlier_followers_count
# quantile(tweets$followers_count,seq(0,1,.01),na.rm = T) # no jump, no need to handle outlier here
# 
# 
# ######### Continuous Variables standardisation ############################################
# # Normalising continuous variables 
# 
# tweets$retweet_count<- scale(tweets$retweet_count) 
# tweets$userLongevity<- scale(tweets$userLongevity) 
# tweets$tweetCount<- scale(tweets$tweetCount) 
# tweets$friends_count<- scale(tweets$friends_count) 
# tweets$favorite_count<- scale(tweets$favorite_count) 
# tweets$followers_count<- scale(tweets$followers_count) 
# tweets$user_id<- scale(tweets$user_id) 
# 
# original_tweets <- tweets
# 
# tweets <- original_tweets
# ##################################################################
# #        MODEL BUILDING                                       #
# ##################################################################
# 
# tweets <- tweets[,c("user_id","retweet_count","followers_count","friends_count","userLongevity","tweetCount","friendShipRatio","favoriteCountRatio","tweetsRatio","normal")]
# 
# 
# ###################################################################
# ## Prediction using Logistic Regression Model
# ###################################################################
# # splitting the data between train and test
# set.seed(100)
# indices = sample.split(tweets$normal, SplitRatio = 0.5)
# train = tweets[indices,]
# test = tweets[!(indices),]
# 
# intialModel <- glm(normal ~ ., data = tweets)
# summary(intialModel)
# 
# stepAIC(intialModel,direction = "both")
# glm_1 <- glm(formula = normal ~ user_id + retweet_count + followers_count + 
#                friends_count + userLongevity + tweetCount + friendShipRatio, 
#              data = tweets)
# summary(glm_1)
# 
# #removing followers_count becasue p value is >.05
# glm_2 <- glm(formula = normal ~ user_id + retweet_count  + 
#                friends_count + userLongevity + tweetCount + friendShipRatio, 
#              data = tweets)
# summary(glm_2)
# 
# 
# #removing retweet_count becasue p value is >.05
# glm_2 <- glm(formula = normal ~ user_id   + 
#                friends_count + userLongevity + tweetCount + friendShipRatio, 
#              data = tweets)
# summary(glm_2)
# 
# 
# # so user_id, friends_count , userLongevity , tweetCount  and friendShipRatio are significant variable we can not remove further
# final_model <- glm_2
# 
# 
# ###################################################################
# #   MODEL EVALUATION FOR Logistic Model                         ##
# ###################################################################
# 
# test_pred = predict(final_model, type = "response", test)
# # Let's see the summary 
# summary(test_pred)
# 
# test$normal <- test_pred
# #View(test)
# # Let's use the probability cutoff of 50%.
# test_pred <- factor(ifelse(test_pred >= 0.50, "Yes", "No"))
# test_actual <- factor(ifelse(test$normal==1,"Yes","No"))
# 
# table(test_actual,test_pred)
# # Let's Choose the cutoff value. 
# # Let's find out the optimal probalility cutoff 
# 
# perform_fn <- function(cutoff) 
# {
#   predicted_Attrition <- factor(ifelse(test_pred >= cutoff, "Yes", "No"))
#   conf <- confusionMatrix(predicted_Attrition, test_actual_Attrition, positive = "Yes")
#   acc <- conf$overall[1]
#   sens <- conf$byClass[1]
#   spec <- conf$byClass[2]
#   out <- t(as.matrix(c(sens, spec, acc))) 
#   colnames(out) <- c("sensitivity", "specificity", "accuracy")
#   return(out)
# }
# 
# # Summary of test probability
# summary(test_pred)
# 
# s = seq(.01,.80,length=100)
# 
# OUT = matrix(0,100,3)
# 
# plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
# axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
# axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
# lines(s,OUT[,2],col="darkgreen",lwd=2)
# lines(s,OUT[,3],col=4,lwd=2)
# box()
# legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
# 
# #the closest difference between sensitivity and specificity approximately 0.015, where both lines cut each other
# cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.015)]
# cutoff
# 
# ###################################################################
# ## Prediction using Neural Network Model
# ###################################################################
# tweets <- original_tweets
# tweets <- finalResult1[,c("user_id","retweet_count","followers_count","friends_count","userLongevity","tweetCount","friendShipRatio","favoriteCountRatio","tweetsRatio","tweetingPeriod","normal")]
# # splitting the data between train and test
# set.seed(100)
# indices = sample.split(tweets$normal, SplitRatio = 0.5)
# train = tweets[indices,]
# test = tweets[!(indices),]
# 
# column_names <- names(train)
# column_names
# 
# combined_col_names <- as.formula(paste("normal ~ ",paste(column_names[!column_names %in% "normal"],collapse = " + ")))
# combined_col_names
# 
# neuralNet <- neuralnet(combined_col_names,data=train,hidden=c(5,3),linear.output=TRUE)
# plot(neuralNet)
# 
# predicted.nn.values <- compute(neuralNet,test[1:10])
# predicted.nn.values


endTime <- Sys.time()

cat(paste('Total Execution Time::', (endTime - startTime)/60, 'min'))

