# OMML-HW1

This repository contains the code for the first homework of the *Optimization Methods for Machine Learning* course (1041415), Sapienza University. The group was composed by Nina Kaploukhaya, Mehrdad Hassanzadeh and me.

**TL;DR**: The task involved building two shallow *FNN* architectures entirely from scratch, <ins>without</ins> relying on specific frameworks such as *TensorFlow* or *Pytorch*, and optimizing the loss function using a routine from *scipy.optimize*. 

## Scope 

The scope of this project is reconstructing the two-dimensional function $F:\mathbb{R}^2 \rightarrow \mathbb{R}$ plotted below, whose analytic expression is not given, within a specified region. 

![rip image pog](stuff/trueFunc.png)

## Data

The data set contains 250 points randomly sampled from the function.

## Models

The two shallow *FNNs* requested are an *MLP* and an *RBF*. 

For `Question 1` the optimization method chosen is gradient-based (to improve the performance), and the gradient is computed using the *backpropagation algorithm*. The parameters of the FFNs to optimize are: $v_j, w_{ji}, b_j$.

For `Question 2` an *Extreme Learning* procedure must be implemented. So the value of $w_{ji}, b_j$ are fixed randomly $\forall i,j$ and the *regularized quadratic convex training error E(v)* is minimized only wrt variables $v$.

For `Question 3` a *two block decomposition method* must be implemented, alternating the convex minimization wrt the output weights v and the non convex minimization with respect to the other parameters.

For the `Bonus` we are required to use our best model, with the best hyperparameters found, and submit it st. the teacher can try it on a new test set and evaluate its performance.

## Results 
Everything is explained in `Report.pdf`
