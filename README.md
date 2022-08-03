# Should You Follow the Gradient Flow? Insights from Runge-Kutta Gradient Descent

Code for ICML 2022 workshop paper ["Should You Follow the Gradient Flow?
Insights from Runge-Kutta Gradient
Descent"](https://drive.google.com/file/d/1L5dJ3lPYYB-842CgHuxjN5MfoqDzkoIV/view?usp=sharing).
Xiang Li, Antonio Orvieto.

## Install dependencies

Install all packages in `requirements.txt` manually or
using 
````
pip install -r requirements.txt
````
There is an additional package, [`hessian-eigenthings`](https://github.com/noahgolmant/pytorch-hessian-eigenthings), that is not
available in PyPI. Please install it by
````
pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
````

## Run the code

To reproduce the results in the paper, simply run
````
bash run.sh
````

Note that the `lr_denom` argument of the main script `main.py`
means the denominator of the learning rate, and the learning rate
used in the training is `2 / lr_denom`.
This is more convenient than specifying a small floating number
in our case.
