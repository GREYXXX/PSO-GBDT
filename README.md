## CITS4002 Research Project : Gradient Boosting Decision Tree Training with Swarm Intelligence

### Project Goal
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

To run the PSGBDT for regression task, you would need to comment line 223 and uncomment line 225 and 226, since I do not have enough time to implement the python argparse.


