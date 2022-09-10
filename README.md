<!-- ## CITS4002 Research Project : Gradient Boosting Decision Tree Training with Swarm Intelligence -->

<!-- ### Project Goal
Developing a robust framework to integrate the PSO algorithm to train the GBDT model.

Conducting experiments to compare the predictive performance with the state-of-art GBDT implementations.

### Packages Required

<p>In order to run the main program, you need to install the following packages.</p>
<p>These packages are:</p>

<ul>
    <li>Numpy</li>
    <li>Pandas</li>
    <li>XGBoost</li>
    <li>Sklearn</li>
    <li>CatBoost</li>
    <li>pickle</li>
    <li>matplotlib</li>
</ul>

### Running Swarm Gradient Boosting Decision Tree training

Fork this repo and clone to your desktop by 

``` git clone  https://github.com/GREYXXX/PSO-GBDT.git```

Run SGBDT classification by

``` python test.py ```

Run SGBDT regression by

``` python test.reg.py ```

Run PSGBDT classification by

``` python test_pretrain.py ```

To run the PSGBDT for regression task, you would need to comment line 223 and uncomment line 225 and 226, since I do not have enough time to implement the python argparse. -->

## Gradient Boosting Decision Tree Training using PSO

### TODO

1. Modify data.py. Re-organise encoding and data preprocessing.
2. Make PSO input as (X : pd.DataFrame/np.array, y : pd.DataFrame/np.array)
3. Finish PSO runner, make training more scalable and reusable.
4. Add gbdt training with normal decision tree. (Attempt bitvector for decision tree base class ? To speed up DFS) 
5. Add pso training for single decision tree and random forest.
6. Experiment for GBDT (ODT, DT) , RF (ODT?, DT), DT. 



