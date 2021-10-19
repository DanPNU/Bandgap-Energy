# PREDICT 454 Advanced Modeling Techniques

# Midterm Project - Nomad 2018 Predicting Transparent Semiconductors - R Script

# OBJECTIVE: Predict formation energy and bandgap energy of transparent conductors.
# Data driven models offer less expensive alternatives to density-function theory
# quantum mechanical methods

#Clear RStudio Environment
rm(list=ls())

# Load Libraries
library(readr)
library(GGally)
library(ggplot2)
library(ISLR)
library(MASS)
library(class)
library(tree)
library(randomForest)
library(leaps)
library(glmnet)
library(pls)
library(xgboost)
library(caret)
library(car)
library(stringr)
library(xgboost)

# load the data
train <- read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/Northwestern/Course Materials/454DL/Nomad2018 Predicting Transparent Conductors/train.csv")
test <- read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/Northwestern/Course Materials/454DL/Nomad2018 Predicting Transparent Conductors/test.csv")


# Add blank formation energy and bandgap energy values to test dataset
test$formation_energy_ev_natom <- NA
test$bandgap_energy_ev <- NA

# Create 70/30 training/validation split
set.seed(25678)
trainIndex = sample(1:nrow(train), size = round(0.7*nrow(train)), replace=FALSE)

# Add train, valid, test labels to each record
train$set <- "train"
train[-trainIndex,]$set <- "valid"
test$set <- "test"

#Combine test & train into single DF for EDA
full <- rbind(train, test)

# Drop ID
full$id <- NULL

# Check Data
summary(full[,-15])
str(full)
colSums(is.na(full)) # The only missing data are the values we need to predict
table(full$set)


# EDA Plots
p1 <- ggplot(data=full, aes(full$spacegroup)) + 
      geom_bar() +
      xlab("Spacegroup") +
      ylab("Count")
plot(p1)

p2 <- ggplot(data=full, aes(full$number_of_total_atoms)) + 
  geom_histogram(bins=20) +
  xlab("Number of Atoms") +
  ylab("Count")
plot(p2)

p3 <- ggplot(data=full, aes(full$percent_atom_al)) + 
  geom_histogram(bins=50) +
  xlab("Aluminum Composition") +
  ylab("Count")
plot(p3)

p4 <- ggplot(data=full, aes(full$percent_atom_ga)) + 
  geom_histogram(bins=50) +
  xlab("Gallium Composition") +
  ylab("Count")
plot(p4)

p5 <- ggplot(data=full, aes(full$percent_atom_in)) + 
  geom_histogram(bins=50) +
  xlab("Indium Composition") +
  ylab("Count")
plot(p5)

p6 <- ggplot(data=full, aes(full$lattice_vector_1_ang)) + 
  geom_histogram(bins=50) +
  xlab("Lattice Vector 1") +
  ylab("Count")
plot(p6)

p7 <- ggplot(data=full, aes(full$lattice_vector_2_ang)) + 
  geom_histogram(bins=50) +
  xlab("Lattice Vector 2") +
  ylab("Count")
plot(p7)

p8 <- ggplot(data=full, aes(full$lattice_vector_3_ang)) + 
  geom_histogram(bins=50) +
  xlab("Lattice Vector 3") +
  ylab("Count")
plot(p8)

p9 <- ggplot(data=full, aes(full$lattice_angle_alpha_degree)) + 
  geom_histogram(bins=50) +
  xlab("Alpha Angle") +
  ylab("Count")
plot(p9)

p10 <- ggplot(data=full, aes(full$lattice_angle_beta_degree)) + 
  geom_histogram(bins=50) +
  xlab("Beta Angle") +
  ylab("Count")
plot(p10)

p11 <- ggplot(data=full, aes(full$lattice_angle_gamma_degree)) + 
  geom_histogram(bins=50) +
  xlab("Gamma Angle") +
  ylab("Count")
plot(p11)

p12 <- ggplot(data=full, aes(full$formation_energy_ev_natom)) + 
  geom_histogram(bins=50) +
  xlab("Formation Energy") +
  ylab("Count")
plot(p12)

p13 <- ggplot(data=full, aes(full$bandgap_energy_ev)) + 
  geom_histogram(bins=50) +
  xlab("Bandgap Energy") +
  ylab("Count")
plot(p13)

# Borrowed from https://www.kaggle.com/headsortails/resistance-is-futile-transparent-conductors-eda
ggcorr(full,method = c("pairwise","spearman"), label = FALSE, angle = -0, hjust = 0.2) +
  coord_flip()


p14 <- ggplot(data=full,aes(x=full$percent_atom_al, y=full$bandgap_energy_ev)) + 
  geom_point(color="dodgerblue4") +
  xlab("AL") +
  ylab("Bandgap Energy")
plot(p14)

p15 <- ggplot(data=full,aes(x=full$percent_atom_al, y=full$formation_energy_ev_natom)) + 
  geom_point(color="darkred")
plot(p15)

p16 <- ggplot(data=full,aes(x=full$percent_atom_ga, y=full$bandgap_energy_ev)) + 
  geom_point(color="darkred") +
  xlab("Ga") +
  ylab("")
plot(p16)

p17 <- ggplot(data=full,aes(x=full$percent_atom_ga, y=full$formation_energy_ev_natom)) + 
  geom_point(color="red")
plot(p17)

p18 <- ggplot(data=full,aes(x=full$percent_atom_in, y=full$bandgap_energy_ev)) + 
  geom_point(color="olivedrab4") +
  xlab("In") +
  ylab("")
plot(p18)

p19 <- ggplot(data=full,aes(x=full$percent_atom_in, y=full$formation_energy_ev_natom)) + 
  geom_point(color="darkgreen")
plot(p19)

p20 <- ggplot(data=full,aes(x=full$number_of_total_atoms, y=full$bandgap_energy_ev)) + 
  geom_point()
plot(p20)

p21 <- ggplot(data=full,aes(x=full$number_of_total_atoms, y=full$formation_energy_ev_natom)) + 
  geom_point()
plot(p21)

p22 <- ggplot(data=full,aes(x=full$lattice_vector_1_ang, y=full$bandgap_energy_ev)) + 
  geom_point(color="blue") +
  xlab("") +
  ylab("Bandgap Energy")
plot(p22)

p23 <- ggplot(data=full,aes(x=full$lattice_vector_1_ang, y=full$formation_energy_ev_natom)) + 
  geom_point(color="blue") +
  xlab("LV1") +
  ylab("Formation Energy")
plot(p23)

p24 <- ggplot(data=full,aes(x=full$lattice_vector_2_ang, y=full$bandgap_energy_ev)) + 
  geom_point(color="red") +
  xlab("") +
  ylab("")
plot(p24)

p25 <- ggplot(data=full,aes(x=full$lattice_vector_2_ang, y=full$formation_energy_ev_natom)) + 
  geom_point(color="red") +
  xlab("LV2") +
  ylab("")
plot(p25)

p26 <- ggplot(data=full,aes(x=full$lattice_vector_3_ang, y=full$bandgap_energy_ev)) + 
  geom_point(color="darkgreen") +
  xlab("") +
  ylab("")
plot(p26)

p27 <- ggplot(data=full,aes(x=full$lattice_vector_3_ang, y=full$formation_energy_ev_natom)) + 
  geom_point(color="darkgreen") +
  xlab("LV2") +
  ylab("")
plot(p27)

p28 <- ggplot(data=full,aes(x=full$lattice_angle_alpha_degree, y=full$bandgap_energy_ev)) + 
  geom_point(color="blue")
plot(p28)

p29 <- ggplot(data=full,aes(x=full$lattice_angle_alpha_degree, y=full$formation_energy_ev_natom)) + 
  geom_point(color="blue")
plot(p29)

p30 <- ggplot(data=full,aes(x=full$lattice_angle_beta_degree, y=full$bandgap_energy_ev)) + 
  geom_point(color="red")
plot(p30)

p31 <- ggplot(data=full,aes(x=full$lattice_angle_beta_degree, y=full$formation_energy_ev_natom)) + 
  geom_point(color="red")
plot(p31)

p32 <- ggplot(data=full,aes(x=full$lattice_angle_gamma_degree, y=full$bandgap_energy_ev)) + 
  geom_point(color="darkgreen")
plot(p32)

p33 <- ggplot(data=full,aes(x=full$lattice_angle_gamma_degree, y=full$formation_energy_ev_natom)) + 
  geom_point(color="darkgreen")
plot(p33)

p34 <- ggplot(data=full,aes(x=full$bandgap_energy_ev, y=full$formation_energy_ev_natom)) + 
  geom_point(color="darkorange2") +
  xlab("Bandgap Energy") +
  ylab("Formation Energy")
plot(p34)

p35 <- ggplot(data=full,aes(x=full$spacegroup, y=full$number_of_total_atoms)) + 
  geom_point()
plot(p35)



# Feature Creation
full$spacegroup_noAtoms <- NA
full$spacegroup_noAtoms <- ifelse(full$spacegroup == 12 & full$number_of_total_atoms == 20,"12_20",
                           ifelse(full$spacegroup == 12 & full$number_of_total_atoms == 80,"12_80",
                           ifelse(full$spacegroup == 33 & full$number_of_total_atoms == 40,"33_40",
                           ifelse(full$spacegroup == 33 & full$number_of_total_atoms == 80,"33_80",
                           ifelse(full$spacegroup == 167 & full$number_of_total_atoms == 30,"167_30",
                           ifelse(full$spacegroup == 167 & full$number_of_total_atoms == 60,"167_60",
                           ifelse(full$spacegroup == 194 & full$number_of_total_atoms == 10,"194_10",
                           ifelse(full$spacegroup == 194 & full$number_of_total_atoms == 80,"194_80",
                           ifelse(full$spacegroup == 206 & full$number_of_total_atoms == 80,"206_80",
                           ifelse(full$spacegroup == 227 & full$number_of_total_atoms == 40,"227_40",NA))))))))))

full$mw <- 26.98*full$percent_atom_al + 69.72*full$percent_atom_ga + 114.82*full$percent_atom_in
full$Cp <- 24.2*full$percent_atom_al + 25.86*full$percent_atom_ga + 26.74*full$percent_atom_in


#More EDA Plots
p36 <- ggplot(data=full,aes(x=full$lattice_vector_1_ang, y=full$bandgap_energy_ev, colour=full$spacegroup_noAtoms)) + 
  geom_point() +
  xlab("LV1") +
  ylab("Bandgap Energy") + 
  theme(legend.position="bottom",legend.direction="horizontal")
plot(p36)

p37 <- ggplot(data=full,aes(x=full$lattice_vector_2_ang, y=full$bandgap_energy_ev, colour=full$spacegroup_noAtoms)) + 
  geom_point()+
  xlab("LV2") +
  ylab("Bandgap Energy") + 
  theme(legend.position="bottom",legend.direction="horizontal")
plot(p37)

p38 <- ggplot(data=full,aes(x=full$lattice_vector_3_ang, y=full$bandgap_energy_ev, colour=full$spacegroup_noAtoms)) + 
  geom_point() +
  xlab("LV3") +
  ylab("Bandgap Energy") + 
  theme(legend.position="bottom",legend.direction="horizontal")
plot(p38)

p39 <- ggplot(data=full,aes(x=full$bandgap_energy_ev, y=full$formation_energy_ev_natom, colour=full$spacegroup_noAtoms)) + 
  geom_point()
plot(p39)

p40 <- ggplot(data=full, aes(full$spacegroup_noAtoms)) + 
  geom_bar() +
  xlab("SG_N") +
  ylab("Count")
plot(p40)

p41 <- ggplot(data=full, aes(full$mw)) + 
  geom_histogram(bins=50) +
  xlab("Molar Weight") +
  ylab("Count")
plot(p41)

p42 <- ggplot(data=full, aes(full$Cp)) + 
  geom_histogram(bins=50) +
  xlab("Heat Capacity") +
  ylab("Count")
plot(p42)


# Stepwise linear models for each spacegroup_noAtoms - Used to check for best subsets

lm01 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='12_20'),], nvmax = 6, method="forward")
coef(lm01, 2)

lm02 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='12_80'),], nvmax = 6, method="forward")
coef(lm02, 2)

lm03 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='33_40'),], nvmax = 6, method="forward")
coef(lm03, 4)

lm04 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='33_80'),], nvmax = 6, method="forward")
coef(lm04, 4)


lm05 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='167_30'),], nvmax = 6, method="forward")
coef(lm05, 4)

lm06 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='167_60'),], nvmax = 6, method="forward")
coef(lm06, 4)

lm07 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in +
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='194_10'),], nvmax = 6, method="forward")
coef(lm07, 3)

lm08 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='194_80'),], nvmax = 6, method="forward")
coef(lm08, 4)

lm09 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='206_80'),], nvmax = 6, method="forward")
coef(lm09, 2)

lm10 <- regsubsets(bandgap_energy_ev ~ percent_atom_ga + percent_atom_in + 
                     poly(lattice_vector_1_ang,2, raw=TRUE) + poly(lattice_vector_2_ang,2,raw=TRUE) + poly(lattice_vector_3_ang,2,raw=TRUE) + 
                     lattice_angle_alpha_degree + lattice_angle_beta_degree + lattice_angle_gamma_degree,
                   data = full[which(full$spacegroup_noAtoms =='227_40'),], nvmax = 6, method="forward")
coef(lm10, 2)



# set up data for analysis
data.train <- full[full$set=="train",]
x1.train1 <- data.train[,c(1:11,16:17)] # Numeric Predictor Variables
x2.train1 <- data.train[,15] # Spacegroup_noAtom
y1.train <- data.train[,13] # bandgap_energy_ev
n.train <- nrow(y1.train) # Number of training observations


data.valid <- full[full$set=="valid",]
x1.valid1 <- data.valid[,c(1:11,16:17)] # Numeric Predictor Variables
x2.valid1 <- data.valid[,15] # Spacegroup_noAtom
y1.valid <- data.valid[,13] # bandgap_energy_ev
n.valid <- nrow(y1.valid) # Number of training observations


#data.test <- full[full$set=="test",]
#x1.test <- data.test[,c(1:11,16:17)] # Numeric Predictor Variables
#x2.test <- data.test[,15] # Spacegroup_noAtom
#n.test <- nrow(x1.test) # Number of training observations


x1.train.mean1 <- apply(x1.train1, 2, mean)
x1.train.sd1 <- apply(x1.train1, 2, sd)
x1.train.std1 <- t((t(x1.train1)-x1.train.mean1)/x1.train.sd1) # standardize to have zero mean and unit sd
apply(x1.train.std1, 2, mean) # check zero mean
apply(x1.train.std1, 2, sd) # check unit sd
data.train.std.y1 <- data.frame(x1.train.std1, x2.train1, bandgap_energy_ev=y1.train) # to predict bandgap_energy_ev


x1.valid.std1 <- t((t(x1.valid1)-x1.train.mean1)/x1.train.sd1) # standardize using training mean and sd
data.valid.std.y1 <- data.frame(x1.valid.std1, x2.valid1, bandgap_energy_ev=y1.valid) # to predict bandgap_energy_ev

#x1.test.std <- t((t(x1.test)-x1.train.mean)/x1.train.sd) # standardize using training mean and sd
#data.test.std <- data.frame(x1.test.std, x2.test)

# Add spacegroup_noAtoms factors to training data & validation data
data.train.std.y1.factors <- data.train.std.y1
train.factors <- factor(data.train.std.y1.factors$spacegroup_noAtoms)
data.train.std.y1.factors$spacegroup_noAtoms_factor <- as.numeric(train.factors)

data.valid.std.y1.factors <- data.valid.std.y1
valid.factors <- factor(data.valid.std.y1.factors$spacegroup_noAtoms)
data.valid.std.y1.factors$spacegroup_noAtoms_factor <- as.numeric(valid.factors)



##### PREDICTION MODELING ######

# Least squares regression
# Use all variables
model.ls1 <- lm(bandgap_energy_ev ~ spacegroup + number_of_total_atoms + percent_atom_al + percent_atom_ga + percent_atom_in +
                poly(lattice_vector_1_ang, 2, raw=TRUE) + poly(lattice_vector_2_ang, 2, raw=TRUE) + poly(lattice_vector_3_ang, 2, raw=TRUE) + lattice_angle_alpha_degree +
                lattice_angle_beta_degree + lattice_angle_gamma_degree + mw + Cp + spacegroup_noAtoms, 
                data.train.std.y1)
summary(model.ls1)
pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y1) # validation predictions
mean((y1.valid - pred.valid.ls1)^2) # mean prediction error
# 0.1072934
sd((y1.valid - pred.valid.ls1)^2)/sqrt(n.valid) # std error
# 0.007485222


# Use all variables with insignificant variables removed
model.ls2 <- lm(bandgap_energy_ev ~ spacegroup + number_of_total_atoms +
                  poly(lattice_vector_1_ang, 2, raw=TRUE) + poly(lattice_vector_2_ang, 2, raw=TRUE) + 
                  poly(lattice_vector_3_ang, 2, raw=TRUE) + spacegroup_noAtoms, 
                data.train.std.y1)
summary(model.ls2)
pred.valid.ls2 <- predict(model.ls2, newdata = data.valid.std.y1) # validation predictions
mean((y1.valid - pred.valid.ls2)^2) # mean prediction error
# 0.1231993
sd((y1.valid - pred.valid.ls2)^2)/sqrt(n.valid) # std error
# 0.00856243


# Simple linear models for Each Spacegroup_Factor
#12_20
model.ls3_sg1 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 1,])

summary(model.ls3_sg1)
pred.valid.ls3_sg1 <- predict(model.ls3_sg1, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 1,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 1,] - pred.valid.ls3_sg1)^2) # mean prediction error
# 0.01286753
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 1,] - pred.valid.ls3_sg1)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 1,])) # std error
# 0.003533726


#12_80
model.ls3_sg2 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 2,])

summary(model.ls3_sg2)
pred.valid.ls3_sg2 <- predict(model.ls3_sg2, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 2,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 2,] - pred.valid.ls3_sg2)^2) # mean prediction error
# 0.01995613
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 2,] - pred.valid.ls3_sg2)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 2,])) # std error
# 0.003423154

#167_30
model.ls3_sg3 <- lm(bandgap_energy_ev ~ poly(lattice_vector_1_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 3,])

summary(model.ls3_sg3)
pred.valid.ls3_sg3 <- predict(model.ls3_sg3, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 3,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 3,] - pred.valid.ls3_sg3)^2) # mean prediction error
# 0.009396359
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 3,] - pred.valid.ls3_sg3)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 3,])) # std error
# 0.001695


#167_60
model.ls3_sg4 <- lm(bandgap_energy_ev ~ poly(lattice_vector_1_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 4,])

summary(model.ls3_sg4)
pred.valid.ls3_sg4 <- predict(model.ls3_sg4, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 4,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 4,] - pred.valid.ls3_sg4)^2) # mean prediction error
# 0.01509232
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 4,] - pred.valid.ls3_sg4)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 4,])) # std error
# 0.01031428


#194_10
model.ls3_sg5 <- lm(bandgap_energy_ev ~ poly(lattice_vector_3_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 5,])

summary(model.ls3_sg5)
pred.valid.ls3_sg5 <- predict(model.ls3_sg5, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 5,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 5,] - pred.valid.ls3_sg5)^2) # mean prediction error
# 0.1413466
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 5,] - pred.valid.ls3_sg5)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 5,])) # std error
# 0.05549814


#194_80
model.ls3_sg6 <- lm(bandgap_energy_ev ~ poly(lattice_vector_1_ang, 2, raw=TRUE) + Cp,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 6,])

summary(model.ls3_sg6)
pred.valid.ls3_sg6 <- predict(model.ls3_sg6, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 6,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 6,] - pred.valid.ls3_sg6)^2) # mean prediction error
# 0.09943957
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 6,] - pred.valid.ls3_sg6)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 6,])) # std error
# 0.01738364


#206_80
model.ls3_sg7 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 7,])

summary(model.ls3_sg7)
pred.valid.ls3_sg7 <- predict(model.ls3_sg7, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 7,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 7,] - pred.valid.ls3_sg7)^2) # mean prediction error
# 0.009168514
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 7,] - pred.valid.ls3_sg7)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 7,])) # std error
# 0.002048923


#227_40
model.ls3_sg8 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + mw,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 8,])

summary(model.ls3_sg8)
pred.valid.ls3_sg8 <- predict(model.ls3_sg8, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 8,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 8,] - pred.valid.ls3_sg8)^2) # mean prediction error
# 0.2003594
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 8,] - pred.valid.ls3_sg8)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 8,])) # std error
# 0.02246048


#33_40
model.ls3_sg9 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + Cp,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 9,])

summary(model.ls3_sg9)
pred.valid.ls3_sg9 <- predict(model.ls3_sg9, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 9,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 9,] - pred.valid.ls3_sg9)^2) # mean prediction error
# 0.01806411
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 9,] - pred.valid.ls3_sg9)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 9,])) # std error
# 0.004726476


#33_80
model.ls3_sg10 <- lm(bandgap_energy_ev ~ poly(lattice_vector_2_ang, 2, raw=TRUE) + Cp,
                    data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 10,])

summary(model.ls3_sg10)
pred.valid.ls3_sg10 <- predict(model.ls3_sg10, newdata=data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 10,])
mean((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 10,] - pred.valid.ls3_sg10)^2) # mean prediction error
# 0.01047214
sd((y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 10,] - pred.valid.ls3_sg10)^2)/sqrt(nrow(y1.valid[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 10,])) # std error
# 0.001925567

pred.valid.ls3 <- c(pred.valid.ls3_sg1, pred.valid.ls3_sg2, pred.valid.ls3_sg9, pred.valid.ls3_sg10, pred.valid.ls3_sg3,
                    pred.valid.ls3_sg4, pred.valid.ls3_sg5, pred.valid.ls3_sg6, pred.valid.ls3_sg7, pred.valid.ls3_sg8)

mean((y1.valid - pred.valid.ls3)^2) # mean prediction error
# 0.05774371
sd((y1.valid - pred.valid.ls3)^2)/sqrt(n.valid) # std error
# 0.005342127


# Lasso
grid <- 10^seq(10,-2, length=100)


#12_20
data.train.std.y1_sg1 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 1,]
data.train.std.y1_sg1[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg1$lattice_vector_1_ang_sq <- data.train.std.y1_sg1$lattice_vector_1_ang^2
data.train.std.y1_sg1$lattice_vector_2_ang_sq <- data.train.std.y1_sg1$lattice_vector_2_ang^2
data.train.std.y1_sg1$lattice_vector_3_ang_sq <- data.train.std.y1_sg1$lattice_vector_3_ang^2
data.train.std.y1_sg1 <- data.train.std.y1_sg1[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg1 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 1,]
data.valid.std.y1_sg1[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg1$lattice_vector_1_ang_sq <- data.valid.std.y1_sg1$lattice_vector_1_ang^2
data.valid.std.y1_sg1$lattice_vector_2_ang_sq <- data.valid.std.y1_sg1$lattice_vector_2_ang^2
data.valid.std.y1_sg1$lattice_vector_3_ang_sq <- data.valid.std.y1_sg1$lattice_vector_3_ang^2
data.valid.std.y1_sg1 <- data.valid.std.y1_sg1[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg1 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg1)[,-1]
y_train_sg1 <- data.train.std.y1_sg1$bandgap_energy_ev
x_valid_sg1 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg1)[,-1]
y_valid_sg1 <- data.valid.std.y1_sg1$bandgap_energy_ev
model.lasso_sg1 <- glmnet(x_train_sg1, y_train_sg1, alpha=1, lambda=grid)
plot(model.lasso_sg1)
set.seed(12345)
cv.out_sg1 <- cv.glmnet(x_train_sg1, y_train_sg1, alpha=1)
plot(cv.out_sg1)
bestlam_sg1 <- cv.out_sg1$lambda.min
bestlam_sg1 # 0.005523021
lasso.pred_sg1 <- predict(model.lasso_sg1, s=bestlam_sg1, newx = x_valid_sg1) # validation predictions
mean((y_valid_sg1 - lasso.pred_sg1)^2) # mean prediction error
# 0.03739813
sd((y_valid_sg1 - lasso.pred_sg1)^2)/sqrt(length(y_valid_sg1)) # std error
# 0.01163044


#12_80
data.train.std.y1_sg2 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 2,]
data.train.std.y1_sg2[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg2$lattice_vector_1_ang_sq <- data.train.std.y1_sg2$lattice_vector_1_ang^2
data.train.std.y1_sg2$lattice_vector_2_ang_sq <- data.train.std.y1_sg2$lattice_vector_2_ang^2
data.train.std.y1_sg2$lattice_vector_3_ang_sq <- data.train.std.y1_sg2$lattice_vector_3_ang^2
data.train.std.y1_sg2 <- data.train.std.y1_sg2[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg2 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 2,]
data.valid.std.y1_sg2[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg2$lattice_vector_1_ang_sq <- data.valid.std.y1_sg2$lattice_vector_1_ang^2
data.valid.std.y1_sg2$lattice_vector_2_ang_sq <- data.valid.std.y1_sg2$lattice_vector_2_ang^2
data.valid.std.y1_sg2$lattice_vector_3_ang_sq <- data.valid.std.y1_sg2$lattice_vector_3_ang^2
data.valid.std.y1_sg2 <- data.valid.std.y1_sg2[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg2 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg2)[,-1]
y_train_sg2 <- data.train.std.y1_sg2$bandgap_energy_ev
x_valid_sg2 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg2)[,-1]
y_valid_sg2 <- data.valid.std.y1_sg2$bandgap_energy_ev
model.lasso_sg2 <- glmnet(x_train_sg2, y_train_sg2, alpha=1, lambda=grid)
plot(model.lasso_sg2)
set.seed(12345)
cv.out_sg2 <- cv.glmnet(x_train_sg2, y_train_sg2, alpha=1)
plot(cv.out_sg2)
bestlam_sg2 <- cv.out_sg2$lambda.min
bestlam_sg2 # 0.0003639331
lasso.pred_sg2 <- predict(model.lasso_sg2, s=bestlam_sg2, newx = x_valid_sg2) # validation predictions
mean((y_valid_sg2 - lasso.pred_sg2)^2) # mean prediction error
# 0.0225596
sd((y_valid_sg2 - lasso.pred_sg2)^2)/sqrt(length(y_valid_sg2)) # std error
# 0.004619657



#167_30
data.train.std.y1_sg3 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 3,]
data.train.std.y1_sg3[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg3$lattice_vector_1_ang_sq <- data.train.std.y1_sg3$lattice_vector_1_ang^2
data.train.std.y1_sg3$lattice_vector_2_ang_sq <- data.train.std.y1_sg3$lattice_vector_2_ang^2
data.train.std.y1_sg3$lattice_vector_3_ang_sq <- data.train.std.y1_sg3$lattice_vector_3_ang^2
data.train.std.y1_sg3 <- data.train.std.y1_sg3[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg3 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 3,]
data.valid.std.y1_sg3[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg3$lattice_vector_1_ang_sq <- data.valid.std.y1_sg3$lattice_vector_1_ang^2
data.valid.std.y1_sg3$lattice_vector_2_ang_sq <- data.valid.std.y1_sg3$lattice_vector_2_ang^2
data.valid.std.y1_sg3$lattice_vector_3_ang_sq <- data.valid.std.y1_sg3$lattice_vector_3_ang^2
data.valid.std.y1_sg3 <- data.valid.std.y1_sg3[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg3 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg3)[,-1]
y_train_sg3 <- data.train.std.y1_sg3$bandgap_energy_ev
x_valid_sg3 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg3)[,-1]
y_valid_sg3 <- data.valid.std.y1_sg3$bandgap_energy_ev
model.lasso_sg3 <- glmnet(x_train_sg3, y_train_sg3, alpha=1, lambda=grid)
plot(model.lasso_sg3)
set.seed(12345)
cv.out_sg3 <- cv.glmnet(x_train_sg3, y_train_sg3, alpha=1)
plot(cv.out_sg3)
bestlam_sg3 <- cv.out_sg3$lambda.min
bestlam_sg3 # 0.0005969395
lasso.pred_sg3 <- predict(model.lasso_sg3, s=bestlam_sg3, newx = x_valid_sg3) # validation predictions
mean((y_valid_sg3 - lasso.pred_sg3)^2) # mean prediction error
# 0.01956351
sd((y_valid_sg3 - lasso.pred_sg3)^2)/sqrt(length(y_valid_sg3)) # std error
# 0.003674636


#167_60
data.train.std.y1_sg4 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 4,]
data.train.std.y1_sg4[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg4$lattice_vector_1_ang_sq <- data.train.std.y1_sg4$lattice_vector_1_ang^2
data.train.std.y1_sg4$lattice_vector_2_ang_sq <- data.train.std.y1_sg4$lattice_vector_2_ang^2
data.train.std.y1_sg4$lattice_vector_3_ang_sq <- data.train.std.y1_sg4$lattice_vector_3_ang^2
data.train.std.y1_sg4 <- data.train.std.y1_sg4[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg4 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 4,]
data.valid.std.y1_sg4[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg4$lattice_vector_1_ang_sq <- data.valid.std.y1_sg4$lattice_vector_1_ang^2
data.valid.std.y1_sg4$lattice_vector_2_ang_sq <- data.valid.std.y1_sg4$lattice_vector_2_ang^2
data.valid.std.y1_sg4$lattice_vector_3_ang_sq <- data.valid.std.y1_sg4$lattice_vector_3_ang^2
data.valid.std.y1_sg4 <- data.valid.std.y1_sg4[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg4 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg4)[,-1]
y_train_sg4 <- data.train.std.y1_sg4$bandgap_energy_ev
x_valid_sg4 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg4)[,-1]
y_valid_sg4 <- data.valid.std.y1_sg4$bandgap_energy_ev
model.lasso_sg4 <- glmnet(x_train_sg4, y_train_sg4, alpha=1, lambda=grid)
plot(model.lasso_sg4)
set.seed(12345)
cv.out_sg4 <- cv.glmnet(x_train_sg4, y_train_sg4, alpha=1)
plot(cv.out_sg4)
bestlam_sg4 <- cv.out_sg4$lambda.min
bestlam_sg4 # 0.002208795
lasso.pred_sg4 <- predict(model.lasso_sg4, s=bestlam_sg4, newx = x_valid_sg4) # validation predictions
mean((y_valid_sg4 - lasso.pred_sg4)^2) # mean prediction error
# 0.01529438
sd((y_valid_sg4 - lasso.pred_sg4)^2)/sqrt(length(y_valid_sg4)) # std error
# 0.0006865251


#194_10
data.train.std.y1_sg5 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 5,]
data.train.std.y1_sg5[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg5$lattice_vector_1_ang_sq <- data.train.std.y1_sg5$lattice_vector_1_ang^2
data.train.std.y1_sg5$lattice_vector_2_ang_sq <- data.train.std.y1_sg5$lattice_vector_2_ang^2
data.train.std.y1_sg5$lattice_vector_3_ang_sq <- data.train.std.y1_sg5$lattice_vector_3_ang^2
data.train.std.y1_sg5 <- data.train.std.y1_sg5[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg5 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 5,]
data.valid.std.y1_sg5[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg5$lattice_vector_1_ang_sq <- data.valid.std.y1_sg5$lattice_vector_1_ang^2
data.valid.std.y1_sg5$lattice_vector_2_ang_sq <- data.valid.std.y1_sg5$lattice_vector_2_ang^2
data.valid.std.y1_sg5$lattice_vector_3_ang_sq <- data.valid.std.y1_sg5$lattice_vector_3_ang^2
data.valid.std.y1_sg5 <- data.valid.std.y1_sg5[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg5 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg5)[,-1]
y_train_sg5 <- data.train.std.y1_sg5$bandgap_energy_ev
x_valid_sg5 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg5)[,-1]
y_valid_sg5 <- data.valid.std.y1_sg5$bandgap_energy_ev
model.lasso_sg5 <- glmnet(x_train_sg5, y_train_sg5, alpha=1, lambda=grid)
plot(model.lasso_sg5)
set.seed(12345)
cv.out_sg5 <- cv.glmnet(x_train_sg5, y_train_sg5, alpha=1)
plot(cv.out_sg5)
bestlam_sg5 <- cv.out_sg5$lambda.min
bestlam_sg5 # 0.3622858
lasso.pred_sg5 <- predict(model.lasso_sg5, s=bestlam_sg5, newx = x_valid_sg5) # validation predictions
mean((y_valid_sg5 - lasso.pred_sg5)^2) # mean prediction error
# 0.04253483
sd((y_valid_sg5 - lasso.pred_sg5)^2)/sqrt(length(y_valid_sg5)) # std error
# 0.00174227


#194_80
data.train.std.y1_sg6 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 6,]
data.train.std.y1_sg6[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg6$lattice_vector_1_ang_sq <- data.train.std.y1_sg6$lattice_vector_1_ang^2
data.train.std.y1_sg6$lattice_vector_2_ang_sq <- data.train.std.y1_sg6$lattice_vector_2_ang^2
data.train.std.y1_sg6$lattice_vector_3_ang_sq <- data.train.std.y1_sg6$lattice_vector_3_ang^2
data.train.std.y1_sg6 <- data.train.std.y1_sg6[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg6 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 6,]
data.valid.std.y1_sg6[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg6$lattice_vector_1_ang_sq <- data.valid.std.y1_sg6$lattice_vector_1_ang^2
data.valid.std.y1_sg6$lattice_vector_2_ang_sq <- data.valid.std.y1_sg6$lattice_vector_2_ang^2
data.valid.std.y1_sg6$lattice_vector_3_ang_sq <- data.valid.std.y1_sg6$lattice_vector_3_ang^2
data.valid.std.y1_sg6 <- data.valid.std.y1_sg6[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg6 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg6)[,-1]
y_train_sg6 <- data.train.std.y1_sg6$bandgap_energy_ev
x_valid_sg6 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg6)[,-1]
y_valid_sg6 <- data.valid.std.y1_sg6$bandgap_energy_ev
model.lasso_sg6 <- glmnet(x_train_sg6, y_train_sg6, alpha=1, lambda=grid)
plot(model.lasso_sg6)
set.seed(12345)
cv.out_sg6 <- cv.glmnet(x_train_sg6, y_train_sg6, alpha=1)
plot(cv.out_sg6)
bestlam_sg6 <- cv.out_sg6$lambda.min
bestlam_sg6 # 0.0002734046
lasso.pred_sg6 <- predict(model.lasso_sg6, s=bestlam_sg6, newx = x_valid_sg6) # validation predictions
mean((y_valid_sg6 - lasso.pred_sg6)^2) # mean prediction error
# 0.09466328
sd((y_valid_sg6 - lasso.pred_sg6)^2)/sqrt(length(y_valid_sg6)) # std error
# 0.001705204


#206_80
data.train.std.y1_sg7 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 7,]
data.train.std.y1_sg7[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg7$lattice_vector_1_ang_sq <- data.train.std.y1_sg7$lattice_vector_1_ang^2
data.train.std.y1_sg7$lattice_vector_2_ang_sq <- data.train.std.y1_sg7$lattice_vector_2_ang^2
data.train.std.y1_sg7$lattice_vector_3_ang_sq <- data.train.std.y1_sg7$lattice_vector_3_ang^2
data.train.std.y1_sg7 <- data.train.std.y1_sg7[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg7 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 7,]
data.valid.std.y1_sg7[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg7$lattice_vector_1_ang_sq <- data.valid.std.y1_sg7$lattice_vector_1_ang^2
data.valid.std.y1_sg7$lattice_vector_2_ang_sq <- data.valid.std.y1_sg7$lattice_vector_2_ang^2
data.valid.std.y1_sg7$lattice_vector_3_ang_sq <- data.valid.std.y1_sg7$lattice_vector_3_ang^2
data.valid.std.y1_sg7 <- data.valid.std.y1_sg7[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg7 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg7)[,-1]
y_train_sg7 <- data.train.std.y1_sg7$bandgap_energy_ev
x_valid_sg7 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg7)[,-1]
y_valid_sg7 <- data.valid.std.y1_sg7$bandgap_energy_ev
model.lasso_sg7 <- glmnet(x_train_sg7, y_train_sg7, alpha=1, lambda=grid)
plot(model.lasso_sg7)
set.seed(12345)
cv.out_sg7 <- cv.glmnet(x_train_sg7, y_train_sg7, alpha=1)
plot(cv.out_sg7)
bestlam_sg7 <- cv.out_sg7$lambda.min
bestlam_sg7 # 0.006793453
lasso.pred_sg7 <- predict(model.lasso_sg7, s=bestlam_sg7, newx = x_valid_sg7) # validation predictions
mean((y_valid_sg7 - lasso.pred_sg7)^2) # mean prediction error
# 0.01001791
sd((y_valid_sg7 - lasso.pred_sg7)^2)/sqrt(length(y_valid_sg7)) # std error
# 0.00234324


#227_40
data.train.std.y1_sg8 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 8,]
data.train.std.y1_sg8[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg8$lattice_vector_1_ang_sq <- data.train.std.y1_sg8$lattice_vector_1_ang^2
data.train.std.y1_sg8$lattice_vector_2_ang_sq <- data.train.std.y1_sg8$lattice_vector_2_ang^2
data.train.std.y1_sg8$lattice_vector_3_ang_sq <- data.train.std.y1_sg8$lattice_vector_3_ang^2
data.train.std.y1_sg8 <- data.train.std.y1_sg8[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg8 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 8,]
data.valid.std.y1_sg8[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg8$lattice_vector_1_ang_sq <- data.valid.std.y1_sg8$lattice_vector_1_ang^2
data.valid.std.y1_sg8$lattice_vector_2_ang_sq <- data.valid.std.y1_sg8$lattice_vector_2_ang^2
data.valid.std.y1_sg8$lattice_vector_3_ang_sq <- data.valid.std.y1_sg8$lattice_vector_3_ang^2
data.valid.std.y1_sg8 <- data.valid.std.y1_sg8[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg8 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg8)[,-1]
y_train_sg8 <- data.train.std.y1_sg8$bandgap_energy_ev
x_valid_sg8 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg8)[,-1]
y_valid_sg8 <- data.valid.std.y1_sg8$bandgap_energy_ev
model.lasso_sg8 <- glmnet(x_train_sg8, y_train_sg8, alpha=1, lambda=grid)
plot(model.lasso_sg8)
set.seed(12345)
cv.out_sg8 <- cv.glmnet(x_train_sg8, y_train_sg8, alpha=1)
plot(cv.out_sg8)
bestlam_sg8 <- cv.out_sg8$lambda.min
bestlam_sg8 # 0.0004807494
lasso.pred_sg8 <- predict(model.lasso_sg8, s=bestlam_sg8, newx = x_valid_sg8) # validation predictions
mean((y_valid_sg8 - lasso.pred_sg8)^2) # mean prediction error
# 0.2068055
sd((y_valid_sg8 - lasso.pred_sg8)^2)/sqrt(length(y_valid_sg8)) # std error
# 0.02437353


#33_40
data.train.std.y1_sg9 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 9,]
data.train.std.y1_sg9[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg9$lattice_vector_1_ang_sq <- data.train.std.y1_sg9$lattice_vector_1_ang^2
data.train.std.y1_sg9$lattice_vector_2_ang_sq <- data.train.std.y1_sg9$lattice_vector_2_ang^2
data.train.std.y1_sg9$lattice_vector_3_ang_sq <- data.train.std.y1_sg9$lattice_vector_3_ang^2
data.train.std.y1_sg9 <- data.train.std.y1_sg9[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg9 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 9,]
data.valid.std.y1_sg9[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg9$lattice_vector_1_ang_sq <- data.valid.std.y1_sg9$lattice_vector_1_ang^2
data.valid.std.y1_sg9$lattice_vector_2_ang_sq <- data.valid.std.y1_sg9$lattice_vector_2_ang^2
data.valid.std.y1_sg9$lattice_vector_3_ang_sq <- data.valid.std.y1_sg9$lattice_vector_3_ang^2
data.valid.std.y1_sg9 <- data.valid.std.y1_sg9[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg9 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg9)[,-1]
y_train_sg9 <- data.train.std.y1_sg9$bandgap_energy_ev
x_valid_sg9 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg9)[,-1]
y_valid_sg9 <- data.valid.std.y1_sg9$bandgap_energy_ev
model.lasso_sg9 <- glmnet(x_train_sg9, y_train_sg9, alpha=1, lambda=grid)
plot(model.lasso_sg9)
set.seed(12345)
cv.out_sg9 <- cv.glmnet(x_train_sg9, y_train_sg9, alpha=1)
plot(cv.out_sg9)
bestlam_sg9 <- cv.out_sg9$lambda.min
bestlam_sg9 # 0.0006494422
lasso.pred_sg9 <- predict(model.lasso_sg9, s=bestlam_sg9, newx = x_valid_sg9) # validation predictions
mean((y_valid_sg9 - lasso.pred_sg9)^2) # mean prediction error
# 0.03870474
sd((y_valid_sg9 - lasso.pred_sg9)^2)/sqrt(length(y_valid_sg9)) # std error
# 0.01013506


#33_80
data.train.std.y1_sg10 <- data.train.std.y1.factors[data.train.std.y1.factors$spacegroup_noAtoms_factor == 10,]
data.train.std.y1_sg10[,c(1:5,14,16)] <- NULL
data.train.std.y1_sg10$lattice_vector_1_ang_sq <- data.train.std.y1_sg10$lattice_vector_1_ang^2
data.train.std.y1_sg10$lattice_vector_2_ang_sq <- data.train.std.y1_sg10$lattice_vector_2_ang^2
data.train.std.y1_sg10$lattice_vector_3_ang_sq <- data.train.std.y1_sg10$lattice_vector_3_ang^2
data.train.std.y1_sg10 <- data.train.std.y1_sg10[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
data.valid.std.y1_sg10 <- data.valid.std.y1.factors[data.valid.std.y1.factors$spacegroup_noAtoms_factor == 10,]
data.valid.std.y1_sg10[,c(1:5,14,16)] <- NULL
data.valid.std.y1_sg10$lattice_vector_1_ang_sq <- data.valid.std.y1_sg10$lattice_vector_1_ang^2
data.valid.std.y1_sg10$lattice_vector_2_ang_sq <- data.valid.std.y1_sg10$lattice_vector_2_ang^2
data.valid.std.y1_sg10$lattice_vector_3_ang_sq <- data.valid.std.y1_sg10$lattice_vector_3_ang^2
data.valid.std.y1_sg10 <- data.valid.std.y1_sg10[,c(1,10,2,11,3,12,4,5,6,7,8,9)]
x_train_sg10 <- model.matrix(bandgap_energy_ev ~ ., data.train.std.y1_sg10)[,-1]
y_train_sg10 <- data.train.std.y1_sg10$bandgap_energy_ev
x_valid_sg10 <- model.matrix(bandgap_energy_ev ~ ., data.valid.std.y1_sg10)[,-1]
y_valid_sg10 <- data.valid.std.y1_sg10$bandgap_energy_ev
model.lasso_sg10 <- glmnet(x_train_sg10, y_train_sg10, alpha=1, lambda=grid)
plot(model.lasso_sg10)
set.seed(12345)
cv.out_sg10 <- cv.glmnet(x_train_sg10, y_train_sg10, alpha=1)
plot(cv.out_sg10)
bestlam_sg10 <- cv.out_sg10$lambda.min
bestlam_sg10 # 0.001070925
lasso.pred_sg10 <- predict(model.lasso_sg10, s=bestlam_sg10, newx = x_valid_sg10) # validation predictions
mean((y_valid_sg10 - lasso.pred_sg10)^2) # mean prediction error
# 0.01437861
sd((y_valid_sg10 - lasso.pred_sg10)^2)/sqrt(length(y_valid_sg10)) # std error
# 0.003292753

pred.valid.lasso <- c(lasso.pred_sg1, lasso.pred_sg2, lasso.pred_sg9, lasso.pred_sg10, lasso.pred_sg3,
                      lasso.pred_sg4, lasso.pred_sg5, lasso.pred_sg6, lasso.pred_sg7, lasso.pred_sg8)

mean((y1.valid - pred.valid.lasso)^2) # mean prediction error
# 0.06198482
sd((y1.valid - pred.valid.lasso)^2)/sqrt(n.valid) # std error
# 0.005613193


# Hybrid
pred.valid.hybrid <- c(pred.valid.ls3_sg1, pred.valid.ls3_sg2, pred.valid.ls3_sg9, pred.valid.ls3_sg10, pred.valid.ls3_sg3,
                       pred.valid.ls3_sg4, lasso.pred_sg5, lasso.pred_sg6, pred.valid.ls3_sg7, pred.valid.ls3_sg8)
mean((y1.valid - pred.valid.hybrid)^2) # mean prediction error
# 0.05649821
sd((y1.valid - pred.valid.hybrid)^2)/sqrt(n.valid) # std error
# 0.005301184


# XGBoost Set Up
xgboost_train <- data.train.std.y1.factors
xgboost_train[,c(1:5,14)] <- NULL
xgboost_train$lattice_vector_1_ang_sq <- xgboost_train$lattice_vector_1_ang^2
xgboost_train$lattice_vector_2_ang_sq <- xgboost_train$lattice_vector_2_ang^2
xgboost_train$lattice_vector_3_ang_sq <- xgboost_train$lattice_vector_3_ang^2
xgboost_train <- xgboost_train[,c(10,1,11,2,12,3,13,4:9)]

xgboost_valid <- data.valid.std.y1.factors
xgboost_valid[,c(1:5,14)] <- NULL
xgboost_valid$lattice_vector_1_ang_sq <- xgboost_valid$lattice_vector_1_ang^2
xgboost_valid$lattice_vector_2_ang_sq <- xgboost_valid$lattice_vector_2_ang^2
xgboost_valid$lattice_vector_3_ang_sq <- xgboost_valid$lattice_vector_3_ang^2
xgboost_valid <- xgboost_valid[,c(10,1,11,2,12,3,13,4:9)]

# Basic XGBoost Model
model_xgb1 <- xgboost(data.matrix(xgboost_train[,-13]),
                      label=data.matrix(xgboost_train[,13]),
                      objective="reg:linear",
                      nrounds=1000,
                      max.depth=4,
                      eta=0.01,
                      colsample_bytree=0.8,
                      seed=235,
                      metric="rmse",
                      alpha=0.1)

xgb_imp_freq1 <- xgb.importance(feature_names = colnames(xgboost_train), model = model_xgb1)
xgb.plot.importance(xgb_imp_freq1)
print(xgb_imp_freq1)

valid_xgb1 <- predict(model_xgb1,data.matrix(xgboost_valid[,-13]))
mean((y1.valid - valid_xgb1)^2) # mean prediction error
# 0.05075837
sd((y1.valid - valid_xgb1)^2)/sqrt(n.valid) # std error
# 0.004653997

plot.df <- data.frame(y1.valid, predictions = valid_xgb1)
p41 <- ggplot(data=plot.df,aes(x=plot.df$bandgap_energy_ev, y=plot.df$predictions)) + 
  geom_point()
plot(p41)


#### From Joe G Week 3 Discussion
start.time = Sys.time()
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0

for (iter in 1:100) {
  
  param <- list(objective = "reg:linear",
                eval_metric = "rmse",
                max_depth = sample(5:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:10, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 150 
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data.matrix(xgboost_train[,-13]), label=data.matrix(xgboost_train[,13]), params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stop_round=8, maximize=FALSE)
  
  min_rmse = min(mdcv$evaluation_log[, test_rmse_mean])
  min_rmse_index = which.min(mdcv$evaluation_log[, test_rmse_mean])
  
  
  if (min_rmse < best_rmse) {
    best_rmse = min_rmse
    best_rmse_index = min_rmse_index
    best_seednumber = seed.number
    best_param = param
  }
}
total.time = Sys.time() - start.time
total.time 

model_xgb2 <- xgboost(data.matrix(xgboost_train[,-13]),
                      label=data.matrix(xgboost_train[,13]),
                      nrounds=1000,
                      params = best_param)

xgb_imp_freq2 <- xgb.importance(feature_names = colnames(xgboost_train), model = model_xgb2)
xgb.plot.importance(xgb_imp_freq2)
print(xgb_imp_freq1)

valid_xgb2 <- predict(model_xgb2,data.matrix(xgboost_valid[,-13]))
mean((y1.valid - valid_xgb2)^2) # mean prediction error
# 0.05611129
sd((y1.valid - valid_xgb2)^2)/sqrt(n.valid) # std error
# 0.005883836


# Results for bandgap_energy_ev predictions

# MPE  Model
# 0.1072934 LS1
# 0.1231993 LS2
# 0.05774371 LS3
# 0.06198482 Lasso
# 0.05649821 Hybrid
# 0.05075837 XGB1
# 0.05611129 XGB2

# select XGB1 since it has minimum mean prediction error


# set up data for analysis
data.train <- full[full$set=="train",]
x1.train2 <- data.frame(data.train[,c(1:11,16:17)], bandgap_energy_ev_predicted = predict(model_xgb1,data.matrix(xgboost_train[,-13]))) # Numeric Predictor Variables
x2.train2 <- data.train[,15] # Spacegroup_noAtom
y2.train <- data.train[,12] # formation_energy_ev_natom
n.train <- nrow(y2.train) # Number of training observations


data.valid <- full[full$set=="valid",]
x1.valid2 <- data.frame(data.valid[,c(1:11,16:17)], bandgap_energy_ev_predicted = predict(model_xgb1,data.matrix(xgboost_valid[,-13]))) # Numeric Predictor Variables
x2.valid2 <- data.valid[,15] # Spacegroup_noAtom
y2.valid <- data.valid[,12] # formation_energy_ev_natom
n.valid <- nrow(y2.valid) # Number of training observations

x1.train.mean2 <- apply(x1.train2, 2, mean)
x1.train.sd2 <- apply(x1.train2, 2, sd)
x1.train.std2 <- t((t(x1.train2)-x1.train.mean2)/x1.train.sd2) # standardize to have zero mean and unit sd
apply(x1.train.std2, 2, mean) # check zero mean
apply(x1.train.std2, 2, sd) # check unit sd
data.train.std.y2 <- data.frame(x1.train.std2, x2.train2, formation_energy_ev_natom=y2.train) # to predict bandgap_energy_ev


x1.valid.std2 <- t((t(x1.valid2)-x1.train.mean2)/x1.train.sd2) # standardize using training mean and sd
data.valid.std.y2 <- data.frame(x1.valid.std2, x2.valid2, formation_energy_ev_natom=y2.valid) # to predict bandgap_energy_ev


# XGBoost Set Up
xgboost_train.y2 <- data.train.std.y2
xgboost_train.y2[,c(15)] <- NULL

xgboost_valid.y2 <- data.valid.std.y2
xgboost_valid.y2[,c(15)] <- NULL

# Basic XGBoost Model
model_xgb3 <- xgboost(data.matrix(xgboost_train.y2[,-15]),
                      label=data.matrix(xgboost_train.y2[,15]),
                      objective="reg:linear",
                      nrounds=1000,
                      max.depth=4,
                      eta=0.01,
                      colsample_bytree=0.8,
                      seed=235,
                      metric="rmse",
                      alpha=0.1)

xgb_imp_freq3 <- xgb.importance(feature_names = colnames(xgboost_train.y2), model = model_xgb3)
xgb.plot.importance(xgb_imp_freq3)
print(xgb_imp_freq3)

valid_xgb3 <- predict(model_xgb3,data.matrix(xgboost_valid.y2[,-15]))
mean((y2.valid - valid_xgb3)^2) # mean prediction error
# 0.001834693
sd((y2.valid - valid_xgb3)^2)/sqrt(n.valid) # std error
# 0.0001892432

# Try Lasso for Model 2
x_train <- model.matrix(formation_energy_ev_natom ~ .-spacegroup_noAtoms, data.train.std.y2)[,-1]
y_train <- data.train.std.y2$formation_energy_ev_natom
x_valid <- model.matrix(formation_energy_ev_natom ~ .-spacegroup_noAtoms, data.valid.std.y2)[,-1]
y_valid <- data.valid.std.y2$formation_energy_ev_natom
model.lasso_y2 <- glmnet(x_train, y_train, alpha=1, lambda=grid)
plot(model.lasso_y2)
set.seed(12345)
cv.out_y2 <- cv.glmnet(x_train, y_train, alpha=1)
plot(cv.out_y2)
bestlam_y2 <- cv.out_y2$lambda.min
bestlam_y2 # 4.24416e-05
lasso.pred_y2 <- predict(model.lasso_y2, s=bestlam_y2, newx = x_valid) # validation predictions
mean((y_valid - lasso.pred_y2)^2) # mean prediction error
# 0.004619098
sd((y_valid - lasso.pred_y2)^2)/sqrt(length(y_valid)) # std error
# 0.0003269765
out <- glmnet(x_train, y_train, alpha=1, lambda = grid)
lasso.coef <- predict(out, type="coefficients", s=bestlam_y2)
lasso.coef

# Results for formation_energy_ev predictions

# MPE  Model
# 0.001834693 XGB3
# 0.004619098 Lasso

# Use XGB3 because of lower MPE


# Set up Test Data
data.test <- full[full$set=="test",]
x1.test1 <- data.test[,c(1:11,16:17)] # Numeric Predictor Variables
x2.test1 <- data.test[,15] # Spacegroup_noAtom
n.test <- nrow(x1.test1) # Number of training observations


x1.test.std1 <- t((t(x1.test1)-x1.train.mean1)/x1.train.sd1) # standardize using training mean and sd
data.test.std.y1 <- data.frame(x1.test.std1, x2.test1)


# Add spacegroup_noAtoms factors to training data & validation data
data.test.std.y1.factors <- data.test.std.y1
test.factors <- factor(data.test.std.y1.factors$spacegroup_noAtoms)
data.test.std.y1.factors$spacegroup_noAtoms_factor <- as.numeric(test.factors)

# XGB1 Set UP
xgboost_test <- data.test.std.y1.factors
xgboost_test[,c(1:5,14)] <- NULL
xgboost_test$lattice_vector_1_ang_sq <- xgboost_test$lattice_vector_1_ang^2
xgboost_test$lattice_vector_2_ang_sq <- xgboost_test$lattice_vector_2_ang^2
xgboost_test$lattice_vector_3_ang_sq <- xgboost_test$lattice_vector_3_ang^2
xgboost_test <- xgboost_test[,c(9,1,10,2,11,3,12,4:8)]

# Set up data for model 2
data.test <- full[full$set=="test",]
x1.test2 <- data.frame(data.test[,c(1:11,16:17)], bandgap_energy_ev = predict(model_xgb1,data.matrix(xgboost_test))) # Numeric Predictor Variables
x2.test2 <- data.test[,15] # Spacegroup_noAtom
n.test<- nrow(x2.test2) # Number of test observations

# Normalize data
x1.test.std2 <- t((t(x1.test2)-x1.train.mean2)/x1.train.sd2) # standardize using training mean and sd
data.test.std.y2 <- data.frame(x1.test.std2, x2.test2)

# XGBoost set up
xgboost_test.y2 <- data.test.std.y2
xgboost_test.y2[,c(15)] <- NULL

## Final Results
predictions <- test[,1]
predictions <- data.frame(predictions, formation_energy_ev_natom = predict(model_xgb3,data.matrix(xgboost_test.y2)), bandgap_energy_ev = predict(model_xgb1,data.matrix(xgboost_test)))

#Check nrow and ncol
ncol(predictions)
nrow(predictions)

#Export
write.csv(predictions, file="predictions_20180210_01.csv", row.names = FALSE)




