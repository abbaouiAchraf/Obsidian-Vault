# Neural networks and Deep learning
## Neural networks
![[Pasted image 20230422153000.png]]
![[Pasted image 20230422153309.png]]
*Input* -> *nodes+* -> *output*
*Nodes* r called hidden units.
*Number of layers = hidden layers + output layer*
## Supervised Learning
![[Pasted image 20230422154245.png]]
![[Pasted image 20230422154543.png]]
**Structured Data** refers to tables in databases that follows a tabular structure, mean while **Unstructed Data** refers to data in form of audio, image or text ...

## Basics of Neural Network programming
### Binary Classification
*m :* size of training set.
*Nx :* dimention of input feature vector.
![[Pasted image 20230422215430.png]]
**Notations :**
![[Pasted image 20230422215805.png]]
![[Pasted image 20230422215836.png]]

### Logistic Regression (binary classification)
![[Pasted image 20230422220717.png]]
![[Pasted image 20230422221130.png]]
![[Pasted image 20230422220637.png]]
 ![[Pasted image 20230422220833.png]]
When we program neural networks we try to keep *w* and *b* separated
**NOT RECOMMENDED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**
![[Pasted image 20230422221013.png]]
in this exemple *w* and *b* are linked and related
**Loss (error) function :**
![[Pasted image 20230507143121.png]]
### Gradient Descent
![[Pasted image 20230509191002.png]]
Later we ll set conditions about how to choose *Learning rate : Alpha*
![[Pasted image 20230509191130.png]]
![[Pasted image 20230509191505.png]]

### Logistic Regression & Gradient Descent
#### Exemple : 
![[Pasted image 20230509201051.png]]
**Let try the computation graph :**
![[Pasted image 20230509201157.png]]
Modify *w1*, *w2* and *b* to reduce the lost function
![[Pasted image 20230509205232.png]]
variation of LOST FUNCTION Depending on every variable
**Vectorization ?** : its the art of getting rid of explicite for loop in your code


### Vectorization
![[Pasted image 20230510205239.png]]
![[Pasted image 20230510223139.png]]
**IMPORTANT**
When ever you try to write a for-loop search if their any numpy function that will compute the operation (Use  vectorization)
#### Vectorizing Logistic regressing
![[Pasted image 20230510225541.png]]
![[Pasted image 20230511130315.png]]
### Broad Casting
![[Pasted image 20230511133659.png]]
*Read numpy documentation about broadcasting*
### Exam : 
![[Pasted image 20230511184424.png]]






## Shallow Neural Networks
![[Pasted image 20230531222005.png]]
### Computing neural network's output
![[Pasted image 20230529175813.png]]
**We will vectorize the layer Neurons :**
![[Pasted image 20230529175952.png]]
= equal to vector z^[1]
![[Pasted image 20230529180102.png]]
![[Pasted image 20230529181215.png]]
### Vectorizing Across Multiple Examples
![[Pasted image 20230529182226.png]]
![[Pasted image 20230529182235.png]]
if you have an unvectorized implementation and want to compute the predictions of all your training examples, 
you need to do : 
**for i = 1 to m :**
![[Pasted image 20230529182351.png]]
#### Justification doing this ?
![[Pasted image 20230529185505.png]]
## Activation functions :
- Best usage of *sigmoid* is in **binary classification**, else we may consider using *tanh()* function in the output layer
-> Example : **tanh()** for hidden layer and **sigmoid()** for output layer.
-> negative effect of those two functions is when, the *z* value is either High or Low, it can slow down gradient descent, (the slope became very low)
-  the *ReLU* **= max(0,z)** function
- if your output is zero one value, if you're using binary classification, then the **sigmoid activation function is very natural choice for the output layer**. And then **for all other units  ReLU or the rectified linear unit is increasingly the default choice of activation function**. So if you're **not sure what to use for your hidden layer**, I would just use the **ReLU activation function**, is what you see most people using these days.
- *Leaky ReLU :* Lil bit better thab Relu. **max(0.01z,z)**
  ![[Pasted image 20230529193913.png]]
#### Why do you need Non-Linear Activation Functions?
![[Pasted image 20230530151544.png]]
The Neural Network will just output a linear function no matter how big and complex is the Neural Network



## Derivatives of Activation Functions
It just simple calculus
 ![[Pasted image 20230530153034.png]]
## Gradient Descent for Neural Networks (Propagation)
![[Pasted image 20230530171138.png]]
![[Pasted image 20230530231920.png]]
## Random Initialization of Weights in NN

*Remarque :* You should not initialize the weights to zero, because the neuron in one layer will be identical and in back propagation their will remain so, so the dW in every iteration will have identical values in each row. You will want to have different computations in different unites in your hidden layer. So the solution is to initialize your parameter's randomly.
```
>> W^[1] = np.random.randn((2,2)) * 0.01 
# the 0.01 is for not making the weights too large
# Large values of W means large values of Z means you ll end up in the slope of the activation function which means the gradient descent will learn slowly (in sgmoid function, tanh)
>> b^[1] = np.zeros((2,1)) 
# its fine to initialize b with zeros
```
**Large values of W means large values of Z means you ll end up in the slope of the activation function which means the gradient descent will learn slowly (in sigmoid function, tanh)**

## Exam
![[Pasted image 20230530223800.png]]
![[Pasted image 20230530223810.png]]

## Deep L-layer Neural Network
**L = Number of layers**
**n<sup>[i]</sup> = number of nodes in i-layer**
**n<sup>0</sup> = number of input layer**

|      Notation       |              Dimension               |
|:-------------------:|:------------------------------------:|
|   W<sup>[l]</sup>   | (n<sup>[l]</sup>, n<sup>[l-1]</sup>) |
|   z<sup>[l]</sup>   |         (n<sup>[l]</sup>, 1)         |
|   a<sup>[l]</sup>   |         (n<sup>[l]</sup>, 1)         |
|   Z<sup>[l]</sup>   |         (n<sup>[l]</sup>, m)         |
|   A<sup>[l]</sup>   |         (n<sup>[l]</sup>, m)         |
| A<sup>[0]</sup> = X |         (n<sup>[0]</sup>, m)         |

![[Pasted image 20230601133234.png]]

**Hyperparameters :** are the parameters that controls the final values of *w* and *b*.
-> Example : Learning rate $\alpha$, Number of iterations, Number of hidden layers, Number of hidden units, Choice of activation functions …
**denotes element-wise multiplication)**
![[Pasted image 20230601140105.png]]

## NOTES PDF:
### Week 1:
![[C1_W1.pdf]]

### Week 2:
![[C1_W2.pdf]]
![[1. Standard notations for Deep Learning 1.pdf]]
![[2. Binary_Classification 1.pdf]]
![[3. Logistic_Regression 1.pdf]]
![[4. Logistic_Regression_Cost_Function 1.pdf]]
### Week 3:
![[C1_W3.pdf]]
### Week 4:
*Useful links :*
- https://jonaslalin.com/2021/12/10/feedforward-neural-networks-part-1/
- https://jonaslalin.com/2021/12/21/feedforward-neural-networks-part-2/
- https://jonaslalin.com/2021/12/22/feedforward-neural-networks-part-3/
![[C1_W4.pdf]]


# Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
## Train/ Dev/ Test sets
Data                                                                           [100%]
| Training set                                                                [60%]
| Hold-out Cross Validation - Development set          [20%]
| Test                                                                            [20%]

- We train the *algorithm* with the *Training set* and use *CV set* to see which of the *models* perform the *best* then we take this model and test it with the *test set.*
- In **big data** case (1,000,000) we can use those ratios : 98% / 1% / 1%.
- Make sure that the *dev set* and *test set* comes from the same distribution
[Training set : High resolution images from webpages]
[Dev/test sets : images from users using an app]
- Some times it is fine to not have a *test set* because the role of it, is to *have an estimation of the performance of the model* ; means if you don't need that estimation means you can skip the *test set* and use the full data in the *training set*.
## Bias / Variance
![[Pasted image 20230603004813.png]]
**Examples of how to detect those scenarios :**
Those results are base on Human error is : ***0%***

|                   | High variance | High bias | Both high variance and bias | Low bias and variance |
|:----------------- |:-------------:|:---------:|:---------------------------:|:---------------------:|
| Train set error : |      1%       |    15%    |             15%             |         0.5%          |
| Dev set error :   |      11%      |    16%    |             30%             |          1%           |

## Basic Recipe for Machine Learning
*align \`\`\` with mermaid to visualize the diagram
```
mermaid
graph TD 
A[Check for High Bias] --> |No|C[Check for High Variance] 
A --> |Yes| B[Implement Strategies for Bias Reduction] 
C --> |No| E[Algorithm is Good: Low Bias, Low Variance] 
C --> |Yes| D[Implement Strategies for Variance Reduction] 
B --> F[Increase Network Complexity] 
B --> G[Train Longer] 
B --> H[Use Advanced Optimization Algorithms] 
B --> I[Explore Different Network Architectures] 
D --> J[Obtain More Data] 
D --> K[Apply Regularization Techniques] 
D --> L[Explore Different Network Architectures] 
```
![[Pasted image 20230603012804.png]]
