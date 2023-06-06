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
**Large values of W means large values of Z means you will end up in the slope of the activation function which means the gradient descent will learn slowly (in sigmoid function, tanh)**

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
*align \`\`\` with mermaid to visualize the diagram* 
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
## Regularization
### Logistic regression
![[Pasted image 20230603145644.png]]
We don't include regularization of *b* because it has a small impact in the function so we don't include it
![[Pasted image 20230603145744.png]]
**$\lambda$ : Regularization parameter.**
### Neural network
*Error in the formula of norm(W<sup>[l]</sup>)*
![[Pasted image 20230603150642.png]]

## Regularization's impact on overfitting
- if $\lambda$ is big and *w<sup>[l]</sup>* is small => *z<sup>[l]</sup>* will be relatively small the NN will compute something not too far from a linear function, which means it will not fit to other values.
![[Pasted image 20230603152234.png]]

## Dropout Regularization
### General :
- In each layer we add a probability or removing or keeping a node in that layer, after doing that we remove the nodes and the connections with it. 
   we will end up training much smaller networks in every iteration.
   ![[Pasted image 20230603171251.png]]
- Its better to choose *keep-prob* value by layer.
### Implementing Inverted dropout :
![[Pasted image 20230603172427.png]]
### Making predictions at test time :
![[Pasted image 20230603172759.png]]
In test time we are not using dropout, because using it will just add noise the prediction
-> The effect of */= Keep-prob* is to ensure that without implementing dropout in test time, the expected calculations wont change.
*Humm something in here I did not get right*

## Data augmentation
### General :
If you do not have a way to access more data : 
*$Example:$*
You have flip and image or take a random proportion from the original image.
![[Pasted image 20230603180111.png]]
### Early stopping
*I did not get it :( its so sad*
## Normalizing Inputs
### Normalizing training sets
#### General :
*Why doing this ?*
![[Pasted image 20230603213819.png]]
Its easy to optimize the cost function $J(w,b)$
$$ J(w,b)= \frac{1}{m} \sum_{i=1}^{m} L(\hat y^{(i)},{y}^{(i)})$$
![[Pasted image 20230603213933.png]]
*Normalizing :*
![[Pasted image 20230603213646.png]]
#### Sub. mean:
![[Pasted image 20230603213429.png]]
![[Pasted image 20230607003625.png]]
![[Pasted image 20230603213510.png]]
#### Normalize variance :
![[Pasted image 20230603213522.png]]
![[Pasted image 20230603213528.png]]


## Vanishing / Exploding Gradients
### General :
Its about how bigger or too small weights *W* can make $\hat y$ exponentially big or small.
- if $W^{[l]}$ > $I$ : the activation function will get bigger exponentially 
- if $W^{[l]}$ < $I$ : the activation function will get smaller exponentially ![[Pasted image 20230603231857.png]]
->$I:$ is the *Identity* matrix
-> Those cases will make training difficult and make gradients decent slow.


## Weight Initialization for Deep Networks
![[Pasted image 20230603233635.png]]
- When using other functions :
![[Pasted image 20230603233818.png]]

## Numerical Approximation of Gradients
![[Pasted image 20230603235139.png]]
![[Pasted image 20230603235159.png]]
![[Pasted image 20230603235322.png]]
## Gradient Checking
![[Pasted image 20230603235817.png]]
also : $J(\theta) = J(\theta_{1},\theta_{2},\theta_{3},...,\theta_{L})$
![[Pasted image 20230604023922.png]]
**Gradient checking**
![[Pasted image 20230604000250.png]]
*We will need to let check* : $$\frac{||d\theta_{approx} - d\theta||_{2}}{||d\theta_{approx}||_{2}+||d\theta||_{2}} \approx \epsilon = 10^{-7}$$
![[Pasted image 20230604001041.png]]
*It will show you if exist a bug in forward or backward propagation*

## Tips :
![[Pasted image 20230604001639.png]]
## Exam + Quiz
- Data normalization doesn't affect the variance of the model.
- When increasing the keep_prob value the probability that a node gets discarded during training is less thus reducing the regularization effect.
- The dropout is a regularization technique and thus helps to reduce the overfit.
	**What you should remember**:
- The weights $W^{[l]}$ should be initialized randomly to break symmetry. 
- However, it's okay to initialize the biases $b^{[l]}$ to zeros. Symmetry is still broken so long as $W^{[l]}$ is initialized randomly. 
**Observations**:
- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when log($a^{[3]}$)=log(0)log⁡($a^{[3]}$)=log⁡(0), the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.
**In summary**:
- Initializing weights to very large random values doesn't work well.
- Initializing with small random values should do better. The important question is, how small should be these random values be? Let's find out up next!

**Optional Read:**
The main difference between Gaussian variable (`numpy.random.randn()`) and uniform random variable is the distribution of the generated random numbers:

- numpy.random.rand() produces numbers in a [uniform distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/rand.jpg).
- and numpy.random.randn() produces numbers in a [normal distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/randn.jpg).

When used for weight initialization, randn() helps most the weights to Avoid being close to the extremes, allocating most of them in the center of the range.

An intuitive way to see it is, for example, if you take the [sigmoid() activation function](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/sigmoid.jpg).

You’ll remember that the slope near 0 or near 1 is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the center will speed the convergence.
![[Pasted image 20230604011256.png]]
Here's a quick recap of the main takeaways:

- Different initializations lead to very different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Resist initializing to values that are too large!
- He initialization works well for networks with ReLU activations
![[Pasted image 20230604015454.png]]
![[Pasted image 20230604022415.png]]
![[Pasted image 20230604022426.png]]
![[Pasted image 20230604024450.png]]
## Optimizations' Algorithms 
### Mini-batch Gradient Descent
Vectorization allows you to efficiently compute on $m$ examples : 
$X_{(n_x ,m)} = [x^{(1)},x^{(2)},x^{(3)},..,x^{(m)}]$ 
$Y_{(1 ,m)} = [y^{(1)},y^{(2)},y^{(3)},..,y^{(m)}]$ 
*What if m = 5,000,000 ?*
- The batch gradient decent will take a long time in every iteration
=> You can have a faster algorithm, if you let the gradient decent have some progress before finish processing the entire 5,000,000 example.
- We split the training set on smaller sets *mini-batches.*
![[Pasted image 20230605003031.png]]
each *mini-batch t* is symbolized with $X^{\{t\}}$,$Y^{\{t\}}$. 
**How to run it:**
![[Pasted image 20230605004211.png]]
![[Pasted image 20230605004326.png]]
this called : *1 epoch*
![[Pasted image 20230605004445.png]]
![[Pasted image 20230605005237.png]]
![[Pasted image 20230605005540.png]]

![[Pasted image 20230605005604.png]]
**How to choose mini-batch size**
![[Pasted image 20230605010709.png]]
### Exponentially Weighted moving Averages
![[Pasted image 20230605175234.png]]
We want to draw the moving *local-average*, we will use this approach to calculate.
![[Pasted image 20230605175640.png]]
![[Pasted image 20230605181153.png]]
![[Pasted image 20230605175649.png]]
![[Pasted image 20230605181138.png]]
over all : 
*Exponentially weighted averages :* $$v_t = \beta v_{t-1} + (1 - \beta)\theta_t$$
![[Pasted image 20230605182936.png]]
**Advantage is it takes low memory but you lose some of the accuracy.**
### Bias Correction in Exponentially Weighted Averages
- the moving averages gets initialized with $v$ = 0 and the next term will be based on $(1-\beta)\theta$, which will give a low start to the *curve* and *the estimation*.
$$v_0 = 0 \ |\  v_t = \frac{\beta v_{t-1} + (1-\beta) v_t}{1-\beta^t}$$
### Gradient Descent with Momentum
![[Pasted image 20230605191100.png]]
- It allows the gradient descent to take more straight forward path and dumb out the isolations
![[Pasted image 20230605191334.png]]
-> $\beta = 0.9 \approx$ averaging to 10 days
-> $v_{dw} = 0,\ b_{db} = 0$ equals to matrix of zeros and an array of zeros.
-> it preferable to use $v_{dw} = \beta v_{dw} +dw$

### RMSprop
![[Pasted image 20230605203449.png]]
![[Pasted image 20230605202602.png]]
*Avoids diverging the function by a huge amount*
**RMS : Root Mean Squares**
### ! ! Recommended to try ! ! Adam Optimization Algorithm
Combines all what we saw in those algorithms
![[Pasted image 20230605204630.png]]
- $\alpha : needs\ to\ be\ tune$
- $\beta_1 : 0.9\ (dw)$
- $\beta_2 : 0.999\ (dw^{2})$
- $\epsilon : 10^{-8}$
**Adam :** *Adaptive Moment Estimation*
## Learning Rate Decay
- Slowly reduce $\alpha$ :
1 epoch : 1 pass though data;
-> *Equation :*$$\alpha = \frac{1}{1 + decay\_rate * epoch\_num} \alpha_0 $$
-  $\alpha_0=0.2,\ decay\_rate=1$
-> *other formulas :*
![[Pasted image 20230605212328.png]]
## The Problem of Local Optima
![[Pasted image 20230605214720.png]]
## Exam
![[Pasted image 20230606205918.png]]
![[Pasted image 20230606211120.png]]
![[Pasted image 20230606213028.png]]
- With batch size 1, you lose all the benefits of vectorization across examples in the mini-batch.
## Tuning Process
## Appropriate Scale to pick Hyperparameters

- Find a good scale to choose randomly an hyperparameter at random
-> Sampling :
![[Pasted image 20230606231536.png]]
![[Pasted image 20230607002346.png]]

## Hyperparameters Tuning in Practice: Pandas vs. Caviar
![[Pasted image 20230607003348.png]]
- Working with one model or too many at once
What way to choose is depending on the computational power available.












## Notes PDF:
### Week 1:
![[C2_W1.pdf]]
### Week 2:
![[C2_W2.pdf]]
### Week 3:
