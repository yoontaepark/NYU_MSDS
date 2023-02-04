# Homework 1 Instructions

NLP with Representation Learning, Fall 2022

Name: Yoon Tae Park

Net ID: yp2201

## Part 1: N-Gram Model

### Part 1a

```
**TODO**: Paste command output
Vocab size: 33175
Train Perplexity: 8.107
Valid Perplexity: 10121.316
```

## Part 2: N-Gram Model with Additive Smoothing

### Part 2a

**TODO**: Short response.
The purpose of smoothing in an n-gram model is to make a language model robust to unseen data. 
If we don't use smoothing methods, we might assign zero probability to unseen data.
However, if we use smoothing to give some probability to the unseen data, we can have some probability for unseen data.
This may have more robust performance for test sets which would contain unseen data

### Part 2b

```
**TODO**: Paste command output.
Vocab size: 33175
Train Perplexity: 116.398
Valid Perplexity: 2844.227
```

### PART 2c

**TODO**: Report validation perplexity for different values of delta and select best delta.
No, delta 0.005 can be improved by using a more smaller value. 
Best-performing pair of hyperparameters (n, delta) is (2, 0.0005). (Valid Perplexity: 440.781)

- n=2, delta=0.0005 
Vocab size: 33175
Train Perplexity: 90.228
Valid Perplexity: 440.781

- n=3, delta=0.0005 
Vocab size: 33175
Train Perplexity: 26.768
Valid Perplexity: 2421.402

- n=2, delta=0.005 
Vocab size: 33175
Train Perplexity: 138.787
Valid Perplexity: 447.895

- n=3, delta=0.005 
Vocab size: 33175
Train Perplexity: 116.398
Valid Perplexity: 2844.227

- n=2, delta=0.05 
Vocab size: 33175
Train Perplexity: 331.400
Valid Perplexity: 663.045

- n=3, delta=0.05 
Vocab size: 33175
Train Perplexity: 733.721
Valid Perplexity: 4648.274


## Part 3: N-Gram Model with Interpolation Smoothing

### Part 3a

```
**TODO**: Paste command output.
Vocab size: 33175
Train Perplexity: 17.596
Valid Perplexity: 293.566
```

### Part 3b

**TODO**: Report validation perplexity for different lambda values and select best lambdas.
--lambda1=0.3 --lambda2=0.6 --lambda3=0.1 works the best in terms of valid perplexity. 
(Train Perplexity: 31.230, Valid Perplexity: 278.929)

--lambda1=0.2 --lambda2=0.3 --lambda3=0.5
Vocab size: 33175
Train Perplexity: 13.169
Valid Perplexity: 319.886

--lambda1=0.5 --lambda2=0.2 --lambda3=0.3
Vocab size: 33175
Train Perplexity: 20.440
Valid Perplexity: 315.517

--lambda1=0.2 --lambda2=0.5 --lambda3=0.3
Vocab size: 33175
Train Perplexity: 17.529
Valid Perplexity: 288.764

--lambda1=0.3 --lambda2=0.5 --lambda3=0.2
Vocab size: 33175
Train Perplexity: 22.677
Valid Perplexity: 279.375

--lambda1=0.3 --lambda2=0.6 --lambda3=0.1
Vocab size: 33175
Train Perplexity: 31.230
Valid Perplexity: 278.929


## Part 4: Backoff

### Part 4a

```
**TODO**: Paste command output.
Vocab size: 33175
Train Perplexity: 8.107
Valid Perplexity: 142.995

```

## Part 5: Test Set Evaluation

### Part 5a

**TODO**: Test set perplexity for each model type. Indicate best model.
- Best backoff model performs the best overall (n=3, Test Perplexity: 136.922)

 Best vanila model: n=2, Test Perplexity: 513.943
--n=3, Test Perplexity: 9279.365
--n=2, Test Perplexity: 513.943
--n=1, Test Perplexity: 919.534

- Best addictive model: n=2, delta=0.0005, Test Perplexity: 412.149
--n=3 --delta=0.0005, Test Perplexity: 2274.174
--n=2 --delta=0.0005, Test Perplexity: 412.149
--n=1 --delta=0.0005, Test Perplexity: 919.535

--n=3 --delta=0.005, Test Perplexity: 2675.325 
--n=2 --delta=0.005, Test Perplexity: 421.384
--n=1 --delta=0.005, Test Perplexity: 919.544

--n=3 --delta=0.05, Test Perplexity: 4409.937
--n=2 --delta=0.05, Test Perplexity: 623.056
--n=1 --delta=0.05, Test Perplexity: 919.641

- Best interpolation model: lambda1=0.3 --lambda2=0.6 --lambda3=0.1, Test Perplexity: 264.141
--lambda1=1/3 --lambda2=1/3 --lambda3=1/3, Test Perplexity: 277.857
--lambda1=0.4 --lambda2=0.5 --lambda3=0.1, Test Perplexity: 268.228
--lambda1=0.3 --lambda2=0.6 --lambda3=0.1, Test Perplexity: 264.141

- Best backoff model: n=3, Test Perplexity: 136.922
--n=3, Test Perplexity: 136.922
--n=2, Test Perplexity: 189.822
--n=1, Test Perplexity: 919.534

### Part 6: A Taste of Neural Networks

**TODO**: Report train and development set accuracy

=====Train Accuracy=====
Accuracy: 5253 / 6920 = 0.759104;
Precision (fraction of predicted positives that are correct): 2691 / 3439 = 0.782495;
Recall (fraction of true positives predicted correctly): 2691 / 3610 = 0.745429;
F1 (harmonic mean of precision and recall): 0.763513;

=====Dev Accuracy=====
Accuracy: 648 / 872 = 0.743119;
Precision (fraction of predicted positives that are correct): 341 / 462 = 0.738095;
Recall (fraction of true positives predicted correctly): 341 / 444 = 0.768018;
F1 (harmonic mean of precision and recall): 0.752759;

Time for training and evaluation: 13.16 seconds

