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

### Experiment

Use the following command to run experiment for both SGBDT and PSGBDT.

Run SGBDT by ``` python run.py ```

You can specify the dataset path by ``` --dataset_path $dataset name``` , currelty support 

<ul>
    <li>BankNote.csv (classification)</li>
    <li>wine.csv (classification)</li>
    <li>winequality-red.csv (regression)</li>
    <li>higgs_0.005.csv (classification)</li>
    <li>covat_0.3.csv (classification)</li>
    <li>insurance.csv (regression)</li>
    <li>kc_house_data.csv (regression)</li>
</ul>

and ``` --model_type ``` need to be specified by 

<ul>
    <li>binary_cf</li>
    <li>regressiom</li>
</ul>

with respect to the type of dataset

Run PSGBDT by specifying the ``` --pretrain_file ``` and ``` --pretrain_type ```, pretrain files are under the folder ``` pretrain_models ```. And ``` --pretrain_type ``` needs to set as either ```xgb``` or ```skr```

### Example

Run SGBDT:

```python run.py --dataset_path kc_house_data.csv --model_type regression```

Run PSGBDT:

```python run.py --dataset_path kc_house_data.csv --model_type regression --pretrain_file kc_house.pkl --pretrain_type xgb``





