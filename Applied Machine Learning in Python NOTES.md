# Model Evaluation & Selection
## Evaluation metrics :
### theorical meaning :
We select the model with *best* value of evaluation.

- Accuracy with Imbalanced Classes
Balanced VS Imbalanced
![[Pasted image 20230202151838.png]]

**Dummy classifier uses to make prediction based on the stratigy we give to him without even looking at the data that we wanna predict**
![[Pasted image 20230202152103.png]]
![[Pasted image 20230202152307.png]]
It is bad when our classifier archive a level of accuracy near to the dummy classifier (Imbalanced data) 
it means : ![[Pasted image 20230202152740.png]]

Dummy Regressors or Classifier are like sanity check for our classifier.
![[Pasted image 20230202152915.png]]
Called also Confusion Matrix for Binary Prediction Task
![[Pasted image 20230202153205.png]]

![[Pasted image 20230202154726.png]]
FN and FP are the errors made (FN means it suppose to be POSITIVE but its classofied as NEGATIVE , FP means it suppose to be NEGATIVE but its classofied as POSITIVE).
We can judge just how Classifier act/do based on 4 categories of results [Type of errors made by our classifier]

![[Pasted image 20230202153717.png]]
*y_majority_predicted* is the predicted result of our X_test.

**Recall** also known as the true positive rate, sensitivity, or probability of detection, is such an evaluation metric, and it's obtained by dividing the number of true positives by the sum of true positives and false negatives.
Recall is a evaluation metric that would give higher scores to classifiers that not only achieved a high number of true positives, but also avoided false negatives.
![[Pasted image 20230202181351.png]]
Recall is used when the objective is to minimize false negatives.

**Precision** is an evaluation metric that reflects this situation. It's obtained by dividing the number of true positives by the sum of true positives and false positives.
![[Pasted image 20230202183750.png]] ![[Pasted image 20230202184249.png]]
(False Positive Rate)
**F1-score : combining precision & recall into a single number**
![[Pasted image 20230202184352.png]]
![[Pasted image 20230202184507.png]]
**AUC**
AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.
### Python code :

![[Pasted image 20230202184631.png]]
![[Pasted image 20230202184702.png]]
*Support :* show number of instances in the test set that have that true label.


## Classifier Decision Functions :

main function are **decision_function()** & **predict_proba()**.
We need to select a Decision Threshold to predict based on how conservative we want our model to be.


## Model Selection using evaluation metrics :
### Cross-validation example : 
![[Pasted image 20230210205806.png]]
we r here just evaluating our model a cross different folds
### GridSearch :
![[Pasted image 20230210205940.png]]
we try to find the optimal gamma value that gives the best score in accuracy in 1st case and for AUC in the second case.

![[Pasted image 20230210210350.png]]
How decision boundries change based on each Evaluation Metrics, thats why it should be taken into consideration
(Means like a focusing on minimising false negative or false positive or recall or ......)

**THE POINT OF EVALUATION IS TO SEE HOW THE MODEL LL PERFORM ON  NEW/UNSEEN DATA**

![[Pasted image 20230210211450.png]]
the test set is not seen till the very end, its very important thing.

***Conclusion :***
![[Pasted image 20230210211650.png]]
Dont forget the runtime when making deploying decision for your model.

## Random Forest :
![[Pasted image 20230214231520.png]]


## Grandient-boosted decision trees :
![[Pasted image 20230214233212.png]]
![[Pasted image 20230214233649.png]]

## Neural Network :
![[Pasted image 20230215002126.png]]
![[Pasted image 20230215002302.png]]
Add regularization to the model
![[Pasted image 20230215002624.png]]
Regression with NT
![[Pasted image 20230215003015.png]]
![[Pasted image 20230215003037.png]]

# Deep Learning
![[Pasted image 20230215013246.png]]

# Data Leakage
https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931
![[Pasted image 20230216224843.png]]
https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/discussion/4865#25839#post25839
![[Pasted image 20230216225218.png]]
# BEST PRACTICES :
http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf
