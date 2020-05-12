## Implementation of Transfer Learning and Few Shot Learning Methods ##

### Data Preparation ###

All the data exploration and preparation can be found in ```data_preparation.ipynb```. There is a heavy class-imbalance and it needs to be considered during training.

### Model ###

We fintuned ResNet50 with different number of classes.


### Training ###

Training configurations can be changed in ```parameters.py```.

```
python train.py
```

### Evaluation ###

The model was trained for 4 epochs and metrics was calculated on test-set.


Class          | Top-1         | Top-5     
---------------|---------------|------------
Shirts         | 0.98          | 1.0       
Tshirts        | 0.91          | 0.99       
Casual Shoes   | 0.75          | 0.99       
Flip Flops     | 0.86          | 0.98       
Sandals        | 0.86          | 0.99       
Formal Shoes   | 0.93          | 0.98       
Handbags       | 0.98          | 0.99       
Kurtas         | 0.95          | 0.99       
Belts          | 0.99          | 0.99       
Sports Shoes   | 0.90          | 0.99       
Heels          | 0.63          | 0.99       
Wallets        | 0.97          | 1.00       
Tops           | 0.77          | 0.99       
Briefs         | 0.96          | 1.0
Dresses		   | 0.95          | 1.0
Socks          | 0.96          | 0.99
Watches        | 1.0           | 1.0
Sunglasses     | 1.0           | 1.0
**Average**    | 0.90          | 0.99


### Finetuning ###

