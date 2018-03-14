

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

* Vector-Vector adding:
  * Goal: adding two n-dimensional vectors 
  * Computing result it takes is O of n time. 
  * Storing result vector takes O of n space.
* Vector-Vector outer product:
  * Goal: outer product two n-dimensional vectors 
  * Computing result it takes is quadratic time. 
  * Storing result vector takes is quadratic space.
* Vector-Vector inner product:
  * Goal: outer product two n-dimensional vectors 
  * Computing result it takes is O of n time. 
  * The space complexity is O of 1, or constant.
* Matrix inversion:
  * Goal: outer product two n-dimensional vectors 
  * Computing result it takes cubic time complexity 
  * Storing an n by n matrix requires n squared space.
* Matrix-Vector Multiply:
  * Goal: multiply an n by n matrix with an m by 1 vector. 
  * Computing result it takes is O of nm time. 
  * Storing result vector takes O of n space.
* Matrix-Matrix Multiply:
  * Goal: multiply an n by m matrix with an m by p matrix. 
  * Computing the result takes O of npm time
  * Storing result takes O of np space.

# Introduction to Apache Spark

### Big Data, Hardware Trends, and Apache Spark

#### Useful Links

- <http://www-formal.stanford.edu/jmc/history/lisp/lisp.html>
- [Google: MapReduce: Simplified Data Processing on Large Clusters](http://research.google.com/archive/mapreduce.html)
- [Apache Hadoop](http://research.yahoo.com/files/cutting.pdf)
- [Yahoo! web scale search indexing](http://developer.yahoo.com/hadoop/)
- [Amazon Web Services Elastic MapReduce](http://aws.amazon.com/elasticmapreduce/) 
- [Spark: Cluster Computing with Working Sets](http://people.csail.mit.edu/matei/papers/2010/hotcloud_spark.pdf)
- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](http://usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) 

#### The Big Data Problem

Some of the traditional analysis tools that we use include **Unix shell commands, Pandas, and R**. One of the things that all of these tools have in common is they all **run on a single machine**.

Storage is getting cheaper. But CPUs are not increasing in speed. The big data problem means that a single machine can no longer process or even hold all of the data that we want to analyze. The only solution we have is to **distribute the data over large clusters**.

#### Hardware for Big Data

Modern Big Data Hardware for big data uses consumer grade hardware.

Network speeds are much slower than the shared memory speeds that we found in those big box 1990s solutions. It's much slower to read something over the network, and the network can be much slower than reading something from a storage drive. Also, these machines have very uneven performance. Some may be very fast, others may be very slow, sometimes because they're failing.

#### Distributing Work

##### Counting Words Locally

One challenge is how do we split work across machines? Let's look at an example. How do we count the number of occurrences of each word in a document? So one approach we could do is to use a hash table.

##### Counting Words in Really Big Documents

We could use **Divide and Conquer**. So we're going to take our document, **partition it, and process it individually on machines one through four**. And then, all of the machines will **send their value for their counts for** I **to one machine, and their counts for **do **to one machine, and so on**. And we can do this for all of the words. Partition everything across the machines. And in fact, these could be the same machines one through four that we used already. Now, **our result is also partitioned across multiple machines**. This first step of counting all of the words is a **Map**. Combining all of those results is a **Reduce**. And this is what Google wrote about in 2004 in a research paper.

##### Using Map Reduce for Sorting

What makes cloud computing difficult? Well, there are two challenges we have to deal with. 

* The first is how do we divide work across machines? 
  * We have to **consider our network, how it's organized, how fast it is, or how slow it is**. And we have to consider where data is located, data locality. 
  * Because moving data may be very expensive, especially if we have a lot of data that has to be moved. 
* The second thing that makes cluster computing difficult is dealing with failures. 
  * If a server fails on average every three years, with 10,000 nodes in our cluster we'll see 10 faults per day. 
  * But an even more difficult problem to deal with is **stragglers**. Nodes that have not failed, but are just running very slowly. Maybe they're about to fail or they have some other problem. And that's causing them to perform very slowly.

#### Failures and Slow Tasks

* How do we deal with failures? 
  * The simplest solution is to just launch another task, either on that machine if it's recovered, or on another machine. 
* How do we deal with slow tasks? 
  * All we do is just simply launch another task. And then kill that original task. And we can launch that task on a different machine because maybe that first machine was about to fail and so it's running very slowly.

#### Map Reduce and Disk I/O

With distributed execution in Map Reduce, each stage that we perform passes through the hard drives. So, the initial step of the **map** reads data from the hard drive, processes it, and then writes it out to disk before we perform the shuffle operation to send data to the reducers. At the **reducers**, they read the data in from disk, process it, and write the results out to disk. **The problem here's that disk I/O is very slow**. So if we're running iterative jobs, they're primarily going to run at the speed of the disks instead of the speed of our CPUs. **This is a motivation for Apache Spark**. Because it's not just iterative jobs that we want to perform when we're doing data science, but also complex jobs like interactive mining or stream processing or interactive queries.

#### The Spark Computing Framework

##### Technology Trends and an Opportunity

Cheaper memory means we can put a lot more memory in each server. We can **keep more data in memory instead of writing it out to slow disks**, and then having to read it right back in from those slow disks. So this opportunity led to the creation of a new distributed execution engine, Apache Spark. Remember that what **happens when we have an iterative task** is we read in data from disk, process it, write it out to disk, read it in for the next iteration, process it, write it back to disk, and so on. The abstraction that Apache Spark provides is that of the Resilient Distributed Data sets or RDDs. We write our programs in terms of operations on distributed data sets, and these are partition collections of objects that are spread across a cluster stored either in memory or on disk. We can manipulate and build RDDs using a diverse set of **parallel transformations, including map, filter, and join, and actions, including count, collect, and save**. And Spark automatically keeps track of how we created the RDDs and will automatically rebuild them if a machine fails or a job is running slowly.

##### The Spark Computing Framework

The Spark Computing Framework provides programming abstractions and a parallel runtime that hides the complexities of fault-tolerance and slow machines. Spark will automatically run it multiple times and guarantee the correct result. 

The Spark Framework actually consists of four components. There's the core Apache Spark component along with Spark SQL, Spark Streaming, the ML machine learning library, and the GraphX graphical computing library.

#### Spark and Map Reduce Differences

Taken together, the faster access times and avoidance of serialization/deserialization overhead make Spark much faster than Map Reduce - up to 100 times faster!

|                          | Hadoop Map Reduce | Spark                             |
| ------------------------ | ----------------- | --------------------------------- |
| Storage                  | Disk Only         | In-memory or on disk              |
| Operations               | Map and Reduce    | Map, Reduce, Join, Sample, etc... |
| Execution model          | Batch             | Batch, interactive, streaming     |
| Programming environments | Java              | Scala, Java, R, and Python        |

Other differences between Spark and Map Reduce:

* include Spark having support for generalized patterns. It has a unified engine that supports many different kinds of programming use cases. 
* Spark also supports lazy evaluation of the lineage graph, which leads to reduced wait states and better pipelining. 
* Spark also has a lower overhead for starting jobs and less expensive shuffles.

Pulling all of this together, **in-memory operation makes a huge difference in terms of overall performance.**

#### Historical References

##### Selected Research Papers

- [Spark: Cluster Computing with Working Sets](http://people.csail.mit.edu/matei/papers/2010/hotcloud_spark.pdf), Matei Zaharia, Mosharaf Chowdhury, Michael J. Franklin, Scott Shenker, Ion Stoica. USENIX HotCloud (2010).
- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](http://usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf), Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Murphy McCauley, Michael J. Franklin, Scott Shenker, Ion Stoica. NSDI (2012) 
- [MLlib: Machine Learning in Apache Spark](http://arxiv.org/pdf/1505.06807.pdf), X. Meng, J. Bradley, B. Yuvaz, E. Sparks, S. Venkataraman, D. Liu, J. Freeman, D. Tsai, M. Amde, S. Owen, D. Xin, R. Xin, M. Franklin, R. Zadeh, M. Zaharia, A. Talwalkar. Preprint (2015).

### Spark Essentials

#### Useful Links

The [Spark Documentation](https://spark.apache.org/documentation.html) includes screencasts, training materials, and hands-on exercises. The [Spark Programming Guide](https://spark.apache.org/docs/latest/programming-guide.html) is anothe good starting point, and the [pySpark API documentation](https://spark.apache.org/docs/latest/api/python/index.html) is a great reference to use when writing Spark programs.

#### Python Spark (pySpark)

A Spark program consists of two programs, a **driver** program and a **workers** program. The drivers program runs on the **driver machine**. And the worker programs run on **cluster nodes** or in local threads. And then RDDs are distributed across the workers.

The first thing a program does is to create a **SparkContext** object. This tells Spark how and where to access a cluster. pySpark shell and Databricks Cloud automatically create the **sc** variable for you. In iPython and other programs, you have to use a constructor to create a new Spark context. You then use the **SparkContext** to create our RDDs.

The **master** parameter for a **SparkContext** determines which type and size of cluster to use. And there are two choices, **local clusters** and **remote clusters**. 

| Master Parameter  | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| local             | run Spark with just one worker (no parallelism)              |
| local[k]          | run it with K worker threads (ideally set to number of cores) |
| spark://HOST:PORT | connect to a Spark standalone cluster; PORT depends on config (7077 by default) |
| mesos://HOST:PORT | connect to an Apache Mesos cluster; PORT depends on config (5050 by default) |

#### Resilient Distributed Datasets (RDDs)

* Resilient distributed datasets are the primary abstraction in Spark.

  * Resilient distributed datasets are immutable once constructed. You cannot change IT. You can transform it. You can perform actions on it. But you cannot change an RDD once you construct it. 
  * Spark tracks lineage information to enable the efficient recomputation of any lost data if a machine should fail or crash. 
  * It also enables operations on collections of elements in parallel. 

* Construct RDDs 

  * by paralyzing existent Python collections such as lists
  * by transforming an existing RDD, 
  * from files in HDFS, or any other storage system.

* A programmer specifies the number of partitions for an RDD. The more partitions that you split an RDD into, the more opportunities you have for parallelism.

  Here we have an example of an RDD that's split into five partitions. Since there's only three worker machines, some of the worker machines have two partitions, and one has one partition.

* Two types of operations you can perform on an RDD: **transformations** and **actions**. 

  * Transformations are **lazily evaluated**. They're not computed immediately. 
  * A transformed RDD is executed only when an action runs on it. 
  * You can also **persist**, or cache, RDDs in memory or on disk.

1. **create** an RDD from a data source, a file, or a list. 
2. apply **transformations** to that RDD. So examples are map and filter. 
3. apply **actions** to that RDD. Examples are collect and count.

![](http://ww1.sinaimg.cn/large/006kHYIcgy1fpbhh5fghkj316q0dg771.jpg)

```
>>> data = [1, 2, 3, 4, 5]
# No computation occurs with sc.parallelize()
# just record how to create the RDD with four partitions
>>> rDD = sc.parallelize(data, 4) 
```

we can use a file that comes from **HDFS**, from a text file, from Hypertable, Amazon S3 Apache Hbase, SequenceFiles, or any other Hadoop input format or even a directory or wildcard (/data/201803*).

```
>>> distFile = sc.textFile("README.md", 4)
```

#### Spark Transformations

Spark transformations enable us to create new data sets from an existing one. Again, we're using **lazy evaluations**. It can then optimize the required calculations and automatically recover from failures and slow workers.

| Transformation                   | Meaning                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| **map**(*func*)                  | Return a new distributed dataset formed by passing each element of the source through a function *func*. |
| **filter**(*func*)               | Return a new dataset formed by selecting those elements of the source on which *func* returns true. |
| **flatMap**(*func*)              | Similar to map, but each input item can be mapped to 0 or more output items (so *func* should return a Seq rather than a single item) |
| **distinct**([*numPartitions*])) | Return a new dataset that contains the distinct elements of the source dataset. |

#### Python lambda Functions

You can use lambda functions wherever function objects are required, but they're restricted to a single expression.

#### Transformations

Now, spark turns this function literal (lambda) into a closure and passes it automatically to the worker. So the code in black (eg. map, filter) runs at the driver. The code in green (lambda function) runs at individual workers.

```
>>> rdd = sc.parallelize([1,2,3,4])
>>> rdd.map(lambda x: x * 2)
>>> rdd.filter(lambda x: x % 2 == 0)
>>> rdd2 = sc.parallelize([1,4,2,2,3])
>>> rdd2.distinct()
```

```
>>> rdd = sc.parallelize([1,2,3])
>>> rdd.map(lambda x: [x, x + 5])  # [[1,6],[2,7],[3,8]]
>>> rdd.flatMap(lambda x: [x, x + 5])  # [1,6,2,7,3,8]
```

```
>>> lines = sc.textFile("...", 4)
>>> comments = lines.filter(isComment)
```



#### Spark Actions

Spark actions cause Spark to **execute the recipe** to transform the source through the mechanism for getting results out of Spark.

| Action                             | Meaning                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| **reduce**(*func*)                 | Aggregate the elements of the dataset using a function *func* (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel. |
| **collect**()                      | Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data. |
| **count**()                        | Return the number of elements in the dataset.                |
| **take**(*n*)                      | Return an array with the first *n* elements of the dataset.  |
| **takeOrdered**(*n*, *[ordering]*) | Return the first *n* elements of the RDD using either their natural order or a custom comparator. |

Always be careful when using collect to **make sure that the results will actually fit in the driver's memory**.

```
>>> rdd = sc.parallelize([5,3,1,2])
>>> rdd.collect()  # [5,3,1,2]
>>> rdd.takeOrdered(3, lambda s: -1 * s)  # [5,3,2]
```

#### Spark Programming Model

```
>>> lines = sc.textFile("...", 4)
>>> comments = lines.filter(isComment)
# Nothing happens when we actually do that until we execute an action.
>>> lines.count() # read data; sum within partitions; combine sums in driver
>>> comments.count() # read data again; sum within partitions; combine sums in driver
```

#### Caching RDDs

To avoid having to reload the data, we can add the cache directive to the lines RDD. 

```
>>> lines = sc.textFile("...", 4)
>>> lines.cache()  # save, don't recompute
>>> comments = lines.filter(isComment)
>>> lines.count() # read data; sum within partitions; combine sums in driver
>>> comments.count() #  instead of that having to read from disk, it will read from memory
```

#### Spark Program Lifecycle

Spark Program Lifecycle:

1. create RDDs from some external data source or parallelize a collection in your driver program. 
2. lazily transform these RDDs into new RDDs. 
3. cache some of those RDDs for future reuse.
4. perform actions to execute parallel computation and to produce results.

#### Spark Key-Value RDDs

Spark supports Key-Value pairs for RDDs. Each element of a pair RDD is a pair tuple. Apply SC parallelize operation to a list consisting of two pairs.

Some of the Key-Value transformations

Reduce by key, which takes an RDD and returns a new distributed RDD of key-value pairs where the values for each key are aggregated using the given reduced function.

| Transformation                                               | Meaning                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **groupByKey**([*numPartitions*])                            | When called on a dataset of (K, V) pairs, returns a dataset of (K, Iterable<V>) pairs. **Note:** If you are grouping in order to perform an aggregation (such as a sum or average) over each key, using `reduceByKey` or `aggregateByKey` will yield much better performance. **Note:** By default, the level of parallelism in the output depends on the number of partitions of the parent RDD. You can pass an optional `numPartitions` argument to set a different number of tasks. |
| **reduceByKey**(*func*, [*numPartitions*])                   | When called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function *func*, which must be of type (V,V) => V. Like in `groupByKey`, the number of reduce tasks is configurable through an optional second argument. |
| **sortByKey**([*ascending*], [*numPartitions*])              | When called on a dataset of (K, V) pairs where K implements Ordered, returns a dataset of (K, V) pairs sorted by keys in ascending or descending order, as specified in the boolean `ascending` argument. |
| **aggregateByKey**(*zeroValue*)(*seqOp*, *combOp*, [*numPartitions*]) | When called on a dataset of (K, V) pairs, returns a dataset of (K, U) pairs where the values for each key are aggregated using the given combine functions and a neutral "zero" value. Allows an aggregated value type that is different than the input value type, while avoiding unnecessary allocations. Like in `groupByKey`, the number of reduce tasks is configurable through an optional second argument. |

```
>>> rdd = sc.parallelize([(1,2),(3,4),(3,6)])
>>> rdd.reduceByKey(lambda a, b: a + b)   # [(1,2),(3,10)]
>>> rdd2 = sc.parallelize([(1,'a'),(2,'c'),(1,'b')])
>>> rdd2.sortByKey() # [(1,'a'),(1,'b')],(2,'c')]
```

**Be very careful when using group by key** as it can cause a large amount of data movement across the network. 

```
>>> rdd2 = sc.parallelize([(1,'a'),(2,'c'),(1,'b')])
>>> rdd2.groupByKey() # [(1,['a','b']),(2,'c')]
```

It also can create very large iterables at a worker. Imagine you have an RDD where you have 1 million pairs that have the key 1. All of the values will have to fit in a single worker if you use group by key. **So instead of a group by key, consider using reduced by key or a different key value transformation**.

#### pySpark Closures

* Spark automatically creates closures for:
  * Functions that run on RDDs at workers
  * Any global variables that are used by those workers. 
* One closure per worker 
  * sent for every task, 
  * no communication between workers. 
  * **Any changes to global variables that are at the workers are not sent to the driver or to other workers.** 
* What if we have an iterative or a single job as a large global variable? 
  * So this would result in sending a large read-only look-up table to the workers or sending a large feature vector in a machine learning algorithms to the workers. 
* What if we want to count events that occur during job execution? 
  * How many input lines were blank 
  * How many input records were corrupt? 
* The problem we have is that:
  * **these closures are automatically created are sent or re-sent with every job**. So for an interview job with a large global variable we'll be sending that large global variable with every single job to every single worker. It's also **very inefficient** to continually send large amounts of data to each worker, and **closures are one way from the driver to the worker**. 
* **So how can we count events that occur at a worker and communicate that back to the driver?** Well, to do this, pySpark provides **shared variables** in two different types. 
  * The first is **broadcast variables**. 
    * These enable us to efficiently send large **read-only** values **to all of the workers**. 
    * They're saved at the workers for use in on or more Spark operations. 
    * It's like sending a large read-only look-up table to all of the nodes. 
  * The second shared variable type is **accumulators**. 
    * And they allow us to aggregate values from workers **back to the driver**. 
    * Now **only the driver can access** the value of the accumulator 
    * **For the tasks** the accumulators are basically **write-only**. 
    * And we can use this as an example to count errors that are seen in an RDD across workers.

#### Spark Broadcast Variables

In iterative or repeated computations, broadcast variables avoid the problem of repeatedly sending the same data to workers:

* Keep a **read-only** variable cached at a worker. 
  * Ship to the worker only once instead of with each task 
* Example: efficiently give every worker a large data set or table. 
* Usually distributed using very efficient broadcast algorithms. 

```
# At the driver, specify the broadcast variable.
>>> broadcastVar = sc.broadcast([1, 2, 3])
# At a worker (in code that's automatically passed via a closure)
>>> broadcastVar.value   # [1,2,3]
```

```python
# signPrefixes = loadCallSignTable()            # Expensive if this table is very large
signPrefixes = sc.broadcast(loadCallSignTable())  # Efficiently
def processSignCount(sign_count, signPrefixes):
    # country = lookupCountry(sign_count[0], signPrefixes)
    country = lookupCountry(sign_count[0], signPrefixes.value)
    count = sign_count[1]
    return (country, count)
# RDD contactCounts contains a list of pairs about call sign and account
countryContactCounts = (contactCounts
                        .map(processSignCount)
                        .reduceByKey((lambda x, y: x + y)))
```

#### Spark Accumulators

* Variables that could only be added to by an associative operation.
* Used to very efficiently implement parallel counters and sums
* Only the driver can read an accumulator's values, not the tasks. 

```
>>> accum = sc.accumulator(0)
>>> rdd = sc.parallelize([1,2,3,4])
>>> def f(x):
>>>     global accum
>>>     accum += x
>>> rdd.foreach(f)  # at each worker, always add the value of the element.
>>> accum.value     # 10; 
```

```python
# Counting empty lines

file = sc.textFile(inputFile)
# Create Accumulator[Int] initialized to 0
blankLines = sc.accumulator(0)

def extractCallSigns(lines):
    global blankLines # Make the global variable accessible
    if (line == ""):
        blankLines += 1
    return line.split(" ")

callSigns = file.flatMap(extractCallSigns)
# print out the accumulator at the driver
print("Blank lines: {0}".format(blankLines.value))
```

* Tasks at workers cannot access an accumulator's value. 
* Task see accumulators as write only variables.  
* Accumulators can be used in actions or in transformations. 
  * In an action, each tasks update to the accumulator is **guaranteed** by spark to **only be applied once**. 
  * Perform transformations, there's **no guarantee** because a transformation might have to be run multiple times if there are slow nodes or a node fails. So you should **only use accumulators for debugging** purposes when you have transformations. 
* Accumulators support the types integers, double, long, and float, and you can also create accumulators with custom types. We use one of those in one of the labs.

#### Summary

So in summary, when you write a Spark program, use the **master parameter** to specify the number of **workers**. When you create an **RDD**, you can specify the number of **partitions** for that RDD, and Spark will automatically create that RDD spread across the workers. When you perform ***transformations** and **actions** that use functions, Spark will automatically push a **closure** containing that function to the workers so that it can run at the workers.

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