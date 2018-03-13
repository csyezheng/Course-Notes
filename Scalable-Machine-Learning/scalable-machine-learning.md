

## Course Software Setup

### Installing the Required Software Packages

- Oracle's [Virtual Box](https://www.virtualbox.org/)
- [Vagrant](https://www.vagrantup.com/) automatic VM configuration

### Downloading and Installing the VM Image

1. Create a custom directory under your user profile
   \- Windows: c:\users\<your_user_name>\sparkvagrant
   \- Mac: /Users/<your_user_name>/sparkvagrant
   \- Linux: /home/<yousr_user_name>/sparkvagrant
2. Please ensure you have enough space the disk where your user profile is.
3. Download this [file](https://github.com/spark-mooc/mooc-setup/archive/master.zip) to the custom directory and unzip it.
4. From the unzipped file, copy Vagrantfile to the custom directory you created in step #1 (*NOTE: The file must be named exactly "*Vagrantfile*" with no extension*)
5. Open a command prompt (Windows) or Terminal (Mac/Linux), change to the custom directory, and issue the command "vagrant up --provider=virtualbox"

### Basic Instructions for Using the Virtual Machine

1. To start the VM, from a DOS prompt (Windows) or Terminal (Mac/Linux), issue the command "vagrant up".
2. To stop the VM, from a DOS prompt (Windows) or Terminal (Mac/Linux), issue the command "vagrant halt". *Note: You should always stop the VM before you log off, turn off, or reboot your computer.*
3. At the end of the course, to erase or delete the VM, from a DOS prompt (Windows) or terminal (Mac/Linux), issue the command "vagrant destroy". **Warning: If you erase or delete the VM, you will lose any work you have done and data you have saved, and you will have download it again when you use the "vagrant up" command.**
4. Once the VM is running, to access the notebook, open a web browser to "<http://localhost:8001/>" (on Windows and Mac) or "<http://127.0.0.1:8001/>" (on Linux).

NOTE: In step #4 above, try "<http://localhost:8001/>" first, if that doesn't work, try "<http://127.0.0.1:8001/>", the proper choice depends on the configuration of your computer.  We pre-configuted the virtual machine to by default start the iPython notebook on port 8001.  If you are having problems connecting to the notebook (neither of the two links works), you should check the output of the "vagrant up" command. It may be the case that there was a conflict on your computer with a program already using port 8001 and a higher port number was automatically used. If this occurs, you should use that port number instead of port 8001, and all the rest of the instructions will be the same.

## Course Overview and Machine Learning Basics

### Course Overview and Introduction to Machine Learning

#### Distributed Computing and Apache Spark

The motivation for why scalable machine:

* Scaling out is another option, and the idea here is to work with many small machines, and to connect them together **over a network in a distributed**. 
* And so we can easily add additional nodes to our network.  And as a result, each node is cheap, and both readily accessible.
* So the first is that each of these machines is connected via a network.
* And network communication can be a **real bottleneck** in the distributed setting.
* Second, there's added software complexity when programming in a distributed setting.
* Spark, as a general open source cluster computing engine, is designed to simplify this process, programming in the distributed setting.

This course is motivated by three related trends.

1. The first trend is the rapid growth of **massive data sets**
2. The second trend is the **pervasiveness of distributed and cloud-based clusters**, which crucially provide both the storage and computational resources for processing these massive data sets.
3. The third trend is the **maturation of statistical methods** for a variety of common learning problems, including classification, regression, collaborative filtering, and clustering.

How we can use raw data to train statistical models, and we'll study **common machine learning pipelines**, include pipelines for classification, regression, and exploratory analysis.

* This broad area involves tools and ideas from various domains, including computer science, probability and statistics, optimization, and linear algebra.

#### What is Machine Learning?

**supervised setting** (learning from labeled observations):

* **classification**: the categories of the labels that we're trying to predict are discreet. And there's generally no notion of closeness in the multi-class classification setting.
* **regression**: aim to predict a real value for each item. our labels are continuous, and we can, in fact, define closeness when comparing predictions with labels. 

**unsupervised learning** (learning from unlabeled observations.):

* **clustering**: partition observations into homogeneous regions. For instance, to try to identify communities within large groups in a social network.
*  **dimensionality reduction**: transform an initial feature representation into a more concise one. For instance, representation of high dimensional digital images represented initially as pixels.

And there's two main reasons why we might want to perform unsupervised learning.

* One is that it can be a goal in and of itself to better understand our data, to discover hidden patterns, or to perform exploratory data analysis. 
* Alternatively, it can be a means to an end. It can be in some sense a **preprocessing step** before we perform a downstream supervised learning task.

A variety of **distributed machine learning algorithms** and **data-processing** techniques that are well-suited for large data sets.

#### Typical Supervised Learning Pipeline

1. **obtaining raw data**: web information, emails, genomic data, or other scientific information, images,
2. **feature extraction**: 
   * represent each of our observations via numeric features. 
   * And it's important to note here that the success of a supervised learning pipeline crucially depends on the choice of features.
   * Additionally, there's a connection between feature extraction and unsupervised learning. Unsupervised learning can be used as a preprocessing step for a downstream supervised task. And this is typically involved in the process of feature extraction.
3. **perform supervised learning**: this typically involves training a classification or a regression model on a set of label training data.
4. **evaluation**: And so we can simulate this process by evaluating it on tests or hold out data. And this is label data that wasn't used for training. Once we perform this evaluation, we can decide whether we're happy with the current model we have. And if we're not, we can **iterate**. And this typically involves **extracting new features and/or trying different supervised learning approaches**.
5. **prediction**: prediction on future observations, or in other words, observations that really don't have labels.

#### Sample Classification Pipeline

Examples of classification:

* **fraud detection**: where we have a bunch of user activity and want to determine which, if any, of this activity is fraudulent. 
* **Face detection**: where we have a bunch of images of people and we might want to determine which images correspond to which people's faces. And in this setting it could be a multi-class classification problem if there are many people in the images. 
* **Link prediction**: for any pair of users we want to decide whether or not we should suggest a link to connect these users either as friends or potential business colleagues or so on. 
* **click through rate prediction**: the idea here is to the user is presented with an ad, for instance on a website, and we want to predict whether the user will click on the ad or the probability with which the user will click on it.

Spam classification Pipeline:

1. **obtaining raw data**:  observations consist of emails.
2. **feature extraction**: transforming each of our observations into a vector of real numbers.it would be to use a **bag of words** representation. So remember that our observations here are documents. And to create a bag of words representation we need to perform two steps. In the first step, we **create a vocabulary**. And this vocabulary consists of words that we've seen in our training data or our training observations, and not necessarily all words. We might **remove some words** that we don't think a very predictive or very common words such as "the" or "a" or "by." But basically we want a vocabulary of words that we think might be predictive. So once we create this vocabulary, we can then **use it to derive feature vectors** for each of our observations. So for instance on the top right what we're doing here is we take this vocabulary and we **associate a feature with each word in this vocabulary**. And for a given email we want to check whether or not this word appears, each word in the vocabulary, whether or not it appears in the email. *And if it does, we set that feature equal to one. If it doesn't we set it equal to zero.*
3. **perform supervised learning**: common classifiers include logistic regression, support vector machines, decision trees, random forests, and so on. And in many cases, especially when training at large scale, algorithms that train these different models involve iterative computations with **gradient descent**. Logistic regression has a nice probabilistic interpretation, aim to find a linear decision boundary. 
4. **evaluation**: Well, at a high level, what we really care about is generalization ability. And if we're not, we can **iterate**. And this typically involves **extracting new features and/or trying different supervised learning approaches**. Additionally, there's a chance that our classifiers overfit to the training data.  it will be quite complex and it seems to be overfitting to the particularities. OK, so with these ideas of **generalization** and **overfitting** in mind, how can we go about evaluating the quality of our classifier? Well, the simple idea is that we want to create a **test set** that simulates unobserved data and use this unobserved data to actually perform the evaluation. 
   1. Split the data into training and testing data sets. 
   2. training our classifier on training set. (don't expose test set to classifier) 
   3. Make predictions on test set (ignoring test labels) 
   4. Compare test predictions with underlying test labels.
   5. various metrics that we can use to compare the predictions and the true labels 
   6. Accuracy is common for classification.
5. **prediction**: prediction on future observations, predicting whether new emails that we've seen are or are not spam.

#### Math and Python Review

* Transpose
* Addition and Subtraction:  element-wise operations
* Matrix Scalar Multiplication
* Scalar Products ( a dot product or an inner product):  $$x^Tw=y$$
* Matrix-Vector Multiplication
* Matrix-Matrix Multiplication: it's not commutative.
* Outer Product: $$xw^T=C$$
* Identity Matrix
* Inverse Matrix: $$AA^{-1}=I_n$$   $$A^{-1}A=I_n$$ 
* Euclidean norms for vectors: 
  * the magnitude or length of a number.
  * vector norms generalize this idea for vectors.
  *  the Euclidean norm for $$x \in R^m​$$ is denoted as $$\Vert x \Vert_2​$$ 
    *  $$\Vert x \Vert_2 = \sqrt{x_1^2 + x_2^2 + ... + x_m^2}$$
    * Equals absolute value when $m=1$
    * Related to scalar product: $$\Vert x \Vert_2 = x^Tx$$ 

#### Big O Notation for Time and Space Complexity

Big O notation measure the time and space complexity for the various algorithms. Big O notation describes how algorithms respond to changes in input size, both in terms of processing time and space requirements.

In terms of storage, the space complexity of an algorithm is proportional to the space required to store a floating point number, which is typically 8 bites. In the case of time complexity, the basic unit of time is the time to perform an elementary operation, such as an arithmetic operation or a logical test.

Vector-Vector adding:

​	Goal: adding two n-dimensional vectors 

​	Computing result it takes is O of n time. 

​	Storing result vector takes O of n space.

Vector-Vector outer product:

​	Goal: outer product two n-dimensional vectors 

​	Computing result it takes is quadratic time. 

​	Storing result vector takes is quadratic space.

Vector-Vector inner product:

​	Goal: outer product two n-dimensional vectors 

​	Computing result it takes is O of n time. 

​	The space complexity is O of 1, or constant.

Matrix inversion:

​	Goal: outer product two n-dimensional vectors 

​	Computing result it takes cubic time complexity 

​	Storing an n by n matrix requires n squared space.

Matrix-Vector Multiply:

​	Goal: multiply an n by n matrix with an m by 1 vector. 

​	Computing result it takes is O of nm time. 

​	Storing result vector takes O of n space.

Matrix-Matrix Multiply:

​	Goal: multiply an n by m matrix with an m by p matrix. 

​	Computing the result takes O of npm time

​	Storing result takes O of np space.

# Introduction to Apache Spark

### Big Data, Hardware Trends, and Apache Spark

### Spark Essentials

### Learning Apache Spark

## Linear Regression and Distributed Machine Learning Principles

### Linear Regression and Distributed ML Principles

### Millionsong Regression Pipeline

## Logistic Regression and Click-through Rate Prediction

### Logistic Regression and Click-through Rate Prediction

### Click-through Rate Prediction Pipeline 

## Principal Component Analysis and Neuroimaging

### Principal Component Analysis and Neuroimaging

### Neuroimaging Analysis via PCA