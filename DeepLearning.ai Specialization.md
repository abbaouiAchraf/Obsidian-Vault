# 1. Neural networks and Deep learning
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


# 2. Improving Deep Neural Networks
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

## Normalizing Activations in a Network
We normalize value of $a^{[l]}$ so $w^{[l+1]}, b^{[l+1]}$ can train faster
*normalizing means :* $X = \frac{X-\mu}{\sigma}$  
![[Pasted image 20230607215018.png]]
![[Pasted image 20230607220801.png]]

**Applying it to Network**
![[Pasted image 20230607230043.png]]
->*tf.nn-batch-normalization*
![[Pasted image 20230607230932.png]]
![[Pasted image 20230607231244.png]]
-> **Why does it work ?**
Batch normalization (batch norm) works by normalizing the input features (X) and the values in hidden units to improve learning speed. By normalizing the features, all inputs take on a similar range of values, which accelerates learning. Batch norm extends this normalization concept to the hidden units in the network.

One reason why batch norm is effective is that it makes the weights in deeper layers of the network more robust to changes in weights in earlier layers. This is important because changes in the input distribution, known as covariate shift, can hinder the performance of a network. Even if the underlying function remains the same, if the distribution of inputs changes, the network might need to be retrained. Batch norm reduces the impact of covariate shift by stabilizing the distribution of hidden unit values. It ensures that the mean and variance of these values remain constant, even as the parameters of earlier layers change. This allows each layer to learn more independently, speeding up learning throughout the network.

Batch norm also has a slight regularization effect. Each mini-batch used in training introduces noise due to the estimation of mean and variance with a small sample of data. Scaling the hidden units by these noisy estimates adds some noise to the activations, similar to the dropout regularization technique. This regularization effect prevents the network from relying too heavily on any single hidden unit.

It's worth noting that batch norm should primarily be used for normalizing hidden unit activations and accelerating learning, rather than relying on it solely as a regularization technique. However, it does have a small unintended regularization effect. Additionally, using larger mini-batch sizes reduces the regularization effect, as the noise introduced by the estimation of mean and variance decreases.

During testing or making predictions, batch norm needs to be handled differently since only single examples are processed rather than mini-batches. The specific details of how to apply batch normalization at test time are typically discussed separately.

Overall, batch norm is a powerful technique that helps stabilize the distribution of hidden unit values, making learning more efficient and improving the robustness of deep neural networks.

## Batch Norm at Test Time
 ![[Pasted image 20230607234332.png]]
 ![[Pasted image 20230607235118.png]]
## Softmax (Multiclass classification)
![[Pasted image 20230608001319.png]]
![[Pasted image 20230608001502.png]]
$\hat y : (4,1)$
**How ?**
$Z^{[l]} = W^{[l]}a^{[l-1]}+b^{[l]}$
*activation function*
$$a^{[l]} = \frac{e^{Z^{[l]}}}{\sum^{4}_{i=1} t_i}\ | \ t = e^{Z^{[l]}}\ | \ a^{[l]}:(4,1)$$
**It s like a generalization of logistic regression**
-> all decision boundaries will be linear with a NN with no hidden layers.
![[Pasted image 20230608005647.png]]
## Training a Softmax Classifier
### General :
-> Hard max is when one output get value of *1* and the else get value of *0*.
![[Pasted image 20230608011323.png]]
### Loss function :
![[Pasted image 20230608012308.png]]
## Deep Learning Frameworks : TENSORFLOW
- In TensorFlow you need only to implement the forward_prob and the back_prob will be taken care by the framework.
- ```with tf.GradientTape() as tape :``` this will record the computational steps of the operation under it.
-> training a model to find the minimum of the cost function $w^2 -10*w +25$
![[Pasted image 20230608013800.png]]
- *0.1* in ```optimizer = tf.keras.optimizers.Adam(0.1) ``` represent the learning rate $\alpha$.
**Simple Alternative**
![[Pasted image 20230608014448.png]]
![[Pasted image 20230608014436.png]]
## Exam
- the main object to get used and transformed is the `tf.Tensor`. These tensors are the TensorFlow equivalent of Numpy arrays, i.e. multidimensional arrays of a given data type that also contain information about the computational graph.
- TensorFlow will compute the derivatives for you, by moving backwards through the graph recorded with `GradientTape`. All that's left for you to do then is specify the cost function and optimizer you want to use!
- use `tf.Variable` to store the state of your variables. Variables can only be created once as its initial value defines the variable shape and type. Additionally, the `dtype` arg in `tf.Variable` can be set to allow data to be converted to that type. But if none is specified, either the datatype will be kept if the initial value is a Tensor, or `convert_to_tensor` will decide. It's generally best for you to specify directly, so nothing breaks!
- Since TensorFlow Datasets are generators, you can't access directly the contents unless you iterate over them in a for loop, or by explicitly creating a Python iterator using `iter` and consuming its elements using `next`. Also, you can inspect the `shape` and `dtype` of each element using the `element_spec` attribute.
- There's one more additional difference between TensorFlow datasets and Numpy arrays: If you need to transform one, you would invoke the `map` method to apply the function passed as an argument to each of the elements.![[Pasted image 20230608235504.png]]
- ![[Pasted image 20230609000558.png]]
- ![[Pasted image 20230609001118.png]]
  ```one_hot = tf.reshape(tf.one_hot(label,depth,axis = 0),shape=[-1,])```
- 

## Notes PDF:
### Week 1:
![[C2_W1.pdf]]
### Week 2:
![[C2_W2.pdf]]
### Week 3:
![[C2_W3.pdf]]

# 3. Structuring Machine Learning Projects
## ML strategy
- To improve a ML model we have too many ideas
![[Pasted image 20230610140547.png]]
	a poor choice of ideas may lead to waste of time and resources into a direction with no good result
***Orthogonalization :*** What to tune to achieve that effect.
![[Pasted image 20230610150800.png]]

## Setting Up your Goal
### Single Evaluation Metric 
![[Pasted image 20230610152314.png]]
- It speeds the the iteration cycle of 'Idea, code and Experiment'
### Satisficing and Optimizing Metric
![[Pasted image 20230610152726.png]]
- trading of measures.
- If we have N metrics that we care about, its better to choose *1 optimizing* (best value possible) and *N-1 satisficing* (means it should not surpass a threshold)
### Distribution Train/dev/test sets to speed up
- Find a way to make your Dev and Test sets come from the same distribution (All your data mixed together).
- Choose a dev set and test set to reflect data you expect to get in the future and consider important todo well on.
-  Rule of thumb [70%, 30%] - [60%, 20%, 20%], that was reasonable in early stages of deep learning era. In *modern deep learning era* (1,000,000 sample) [98%, 1%, 1%]
- Set your test set to be big enough to give high confidence in the overall performance of your system.
- In some usages, we don't need a test set, so people attend to use Train-Dev set split to iterate and improve the model **Not recommended !**
### When to Change Dev/Test Sets and Metrics?
![[Pasted image 20230610160911.png]]
- This is a sign that you should change the eval. metric.
- Define new evaluation/Error metric.
![[Pasted image 20230610162621.png]]
## Human-level performance
### General
* *Bayes optimal error :* is the best possible error, where no function cannot surpass it at all. Best theoretical function that cannot be surpassed.
![[Pasted image 20230610175426.png]]
- some times performances slows down when it surpasses human performance, because mostly it's not far from *Bayes optimal error*.
- As long as ML model is worse than humans, you can:
	-  Get labeled data from humans.
	-  Gain insight from manual error analysis: Why did a person get this right?
	-   Better analysis of bias/variance.

### Avoidable Bias 
![[Pasted image 20230610182959.png]]
-  Study case
![[Pasted image 20230610211520.png]]
-> What is *human-level* error?
We can conclude that *Bayes error* $\lt 0.5\%$ 

### Surpassing Human-level Performance
*I don't know what to write in here.*
### Improving your Model Performance
![[Pasted image 20230610231641.png]]
![[Pasted image 20230610232312.png]]
## Error Analysis
### Case study:
- We have a cat image classifier who achieves 90% accuracy, but it miss labeled dog images with cats.
![[Pasted image 20230611211213.png]]
*Evaluate if a single idea is worth working on.*
![[Pasted image 20230611211301.png]]
![[Pasted image 20230611211503.png]]
### Cleaning Up Incorrectly Labeled Data
- **DL algorithms are quite robust to random errors in the training set**
- ![[Pasted image 20230611220938.png]]
9.4% of error worth working on
- ![[Pasted image 20230611221926.png]]
### Build your First System Quickly, then Iterate
![[Pasted image 20230611222709.png]]
- Get Data + Label it
- Split into Train/dev/test
- Choose a metric (Target)
- Build you 1st DL model
- Use bias, variance and error analysis to know your next step 
## Mismatched Training and Dev/Test Set
### Training and Testing on Different Distributions
![[Pasted image 20230611230641.png]]
- *Option 01:* Merge the both datasets and shuffle them.
	- $Advantage :$ The dev and test set will come from the same distribution 
	- $desadvantage :$ The dev and test set will have right percentage or webpages images rather what u actually care about.
- *Option 02:* Have all the webpages images + a little of mobile app, and dev/test set will be purely from mobile app.
	- $Advantage :$ You are aiming your target properly, so you have a model that will focus on the mobile  app images. 
	- $desadvantage :$ Train set will be from different distribution than dev and test set
**Conclusion:** the *Option 02* is recommended because it will give the best result on the long term.
![[Pasted image 20230612000517.png]]
![[Pasted image 20230612000952.png]]
![[Pasted image 20230612001048.png]]

### Bias and Variance with Mismatched Data Distributions
![[Pasted image 20230612002701.png]]
![[Pasted image 20230612002803.png]]
![[Pasted image 20230612003144.png]]
![[Pasted image 20230612003317.png]]
![[Pasted image 20230612003532.png]]
![[Pasted image 20230612004040.png]]
### Addressing Data Mismatch
![[Pasted image 20230612012258.png]]
**Creating more data**
![[Pasted image 20230612012643.png]]
- Let say we have 10,000 Hours of recorded audio and only 1 Hour of car noise.
	- *Approach 01:* we can repeat the 1 hour * 10,000 time but their is a possibility for the model to overfit for the car noise.
 It is possible to use 10,000 hour of unique car noise avoid the issue of overfitting
![[Pasted image 20230612013414.png]]
*Try to avoid getting data of images from a video game (Because they are limited) or AI generated images (Because they follow a pattern and the model may overfit)*
## Learning from Multiple Tasks
### Transfer Learning
- Using a model ability to recognize (Knowledge) cats as example and transfer this ability or knowledge to help recognize X-RAY images
- ![[Pasted image 20230612014300.png]]
- ![[Pasted image 20230612015421.png]]
Remove  the last output layer and the connection leading to it and replace it with a new output layer or multiple layers with randomly initialized weights and bias $w^{[l]}, b^{[l]}$ and re-train the model the new data $(x,y)$ .
-> if small data : We can fix all weights and biases of old model and only train the added weight and bias.
-> if enough data : we can re-train the full model.
**$(x_{\_},y_{\_}):$ called pre-trained model.**
**$(x,y):$ called fine-tuned model.**
![[Pasted image 20230612015148.png]]
**WHY?**
because a lot of low level tasks like detecting edges, lines , shapes and … . That are learned from a huge dataset by a pre-trained model may do better and help the fine-tuned model to learn faster.
**WHEN?**
Transfer learning makes sense when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to.
![[Pasted image 20230612020639.png]]
### Multi-task Learning
#### Study case : *Simplified autonomous driving example*
The car will have to detect : Pedestrians, cars, stop signs, traffic lights...
![[Pasted image 20230612211025.png]]
$X^{(i)}:$ Multi-labeled image.
![[Pasted image 20230612211454.png]]
>To train it we need to define :
>$Loss :\hat y_{(4,1)}^{(i)}$ 
>![[Pasted image 20230612211650.png]]
>Unlike Softmax regression, One image can have multiple labels
	 
**Multi-task learning:** is when developing a model to identify multiple classes (Objects). Instead of developing 4 NN that will have similar earlier features you can develop one model.
-> You can use multi-task leaning when having a data set with some features not really labeled "?" in all images.
![[Pasted image 20230612214527.png]]
**When does it make sense**
- If you don't have a big NN, then Multi-task learning will hurt performance.
![[Pasted image 20230612220149.png]]
## What is End-to-end Deep Learning?
### General:
there have been some data processing systems, or learning systems that require multiple stages of processing. And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network.
![[Pasted image 20230612223204.png]]
- One of the challenges is u may need a lot of data.
![[Pasted image 20230612232105.png]]
Instead of trying to learn all at once, it's better to learn things step by step when not having a really a lot of data.
### Pros and cons:
- **Pros**
	- Let your data speak: you don't force a learning path to it
	- Less hand-designing of components needed
- **Cons**
	- May need large amount of data
	- Excludes potentially useful hand-designed components
### Applying end-to-end deep learning
![[Pasted image 20230613001317.png]]
![[Pasted image 20230613001725.png]]
![[Pasted image 20230613001824.png]]
## EXAM
- Adding this data to the training set will change the training set distribution. However, it is not a problem to have different training and dev distributions. In contrast, it would be very problematic to have different dev and test set distributions.




## NOTES PDF
### Week 1:
![[C3_W1.pdf]]
![[1 - Orthogonalization.pdf]]
![[2 - Single_number_evaluation_metric-2.pdf]]
![[3 - Satisficing_and_optimizing_metric.pdf]]
![[4 - Training_development_and_test_distributions.pdf]]
![[5 - Size_of_the_development_and_test_sets.pdf]]
![[6 - When_to_change_develpment_test_sets_and_metrics 1.pdf]]
![[7 - Why_human_level_performance.pdf]]
![[8 - Avoidable_bias.pdf]]
![[9 - Understanding_human_level_performance.pdf]]
![[10 - Surpassing_human_level_performance.pdf]]
![[11 - Improving_your_model_performance.pdf]]

### Week 2:
![[C3_W2.pdf]]
![[1 - Build_System_Quickly 1.pdf]]
![[2 - Training_and_testing_on_different_distributions 1.pdf]]

![[3 - Bias_and_variance_with_mismatched_data_distributions 1.pdf]]

![[4 - Adressing_data_mismatch 1.pdf]]

![[5 - Transfer_Learning 1.pdf]]

![[6 - Multi_Task_Learning 1.pdf]]

![[7 - What_is_end_to_end_deep_learning 1.pdf]]

![[8 - Whether_to_use_end_to_end_deep_learning 1.pdf]]

# 4. Convolutional Neural Networks
## CNN
### Computer vision
![[Pasted image 20230613160838.png]]
### Edge Detection Example
#### Vertical edge detection
![[Pasted image 20230613162840.png]]
- To detect edges we filters to extract them (KERNEL) basing on the convolutional operation
-  **Calculation :** vertical edge detection
>![[Pasted image 20230613163315.png]]
>![[Pasted image 20230613163335.png]]
>![[Pasted image 20230613163519.png]]
>![[Pasted image 20230613163637.png]]

- Prove for it:
![[Pasted image 20230613164131.png]]
->
![[Pasted image 20230613164501.png]]
#### Horizontal edge detection
![[Pasted image 20230613164553.png]]
 >![[Pasted image 20230613164824.png]]

**Sobel filter/Schorr filter** are kind of more robust
![[Pasted image 20230613164956.png]]
![[Pasted image 20230613165019.png]]
**We can make back-prob learn the appropriate filter for extracting edges (maybe even on 45°, 73°, 70°...) in a image by treating weights as filters/kernels.**

### Padding
$(n,n) * (f,f) = (n-f+1,n-f+1)$
- *Problems:* 
	- Shrinking output
	- Losing information's of the edge of the image
- *Solution:*
	- Add padding to the image
	- ![[Pasted image 20230613202820.png]]
	- Padding amount = P = 0
	- $(n+2p,n+2p) * (f,f) = (n+2p-f+1,n+2p-f+1)$
	![[Pasted image 20230613203223.png]]
**U can use 1x1, 3x3, 5x5, 7x7 filters**
### Strided convolution
![[Pasted image 20230614005259.png]]
![[Pasted image 20230614011030.png]]
**Cross-correlation (Convolution in math textbooks)**
![[Pasted image 20230614011331.png]]
### Convolutions Over Volume
![[Pasted image 20230614140648.png]]
>We add up the result from each RGB channel 

![[Pasted image 20230614141301.png]]
![[Pasted image 20230614142106.png]]
### One Layer of a Convolutional Network
![[Pasted image 20230614142541.png]]
![[Pasted image 20230614142644.png]]
**Summary of notation**
![[Pasted image 20230614145045.png]]
### Simple CNN
**TYPE OF LAYERS IN A CNN:**
![[Pasted image 20230614155900.png]]
### Pooling layers
![[Pasted image 20230614160916.png]]
>Taking the max from each area.
>It has no parameter to learn from Gradian descent
>![[Pasted image 20230614161301.png]]

![[Pasted image 20230614161526.png]]
>to collapse the image representation

![[Pasted image 20230614162240.png]]
### CNN Example
#### LeNet-5
![[Pasted image 20230614163747.png]]
**Very common patterns: **![[Pasted image 20230614163933.png]]

![[Pasted image 20230614162747.png]]
### Why convolutions
![[Pasted image 20230614175431.png]]
![[Pasted image 20230614175754.png]]
>![[Pasted image 20230614175808.png]]
>One output is only related to a few inputs (9 pixels)

![[Pasted image 20230614180927.png]]

## EXAM (KERAS & TF)
![[Pasted image 20230614203547.png]]



## NOTES PDF
### Week 1:
![[C4_W1.pdf]]