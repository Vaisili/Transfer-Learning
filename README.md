## Implementation of Transfer Learning and Few Shot Learning Methods ##

### Data Preparation ###

All the data exploration and preparation can be found in ```data_preparation.ipynb```. There is a heavy class-imbalance and it needs to be considered during training.

### Model ###

We finetuned ResNet50 with different number of classes.


### Training ###

Training configurations can be changed in ```parameters.py```.

```
python train.py
```

### Evaluation ###

Code for model evaluation can be found in ```model_evaluation.ipynb```

The model was trained for 4 epochs on images with top-20 classes and metrics were calculated on test-set.


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

Next, we finetuned for 10 epochs on above model on remaining classes and evaluated it on test set. Below table shows the top-1 and top-5 accuracy.

Class                | Top-1         | Top-5     
---------------------|---------------|------------
Track Pants          | 0.84          | 0.97
Sweatshirts  		 | 0.68          | 0.91
Shorts  			 | 0.86          | 0.98
Clutches             | 0.98          | 1.0
Backpacks            | 0.97          | 1.0
Caps                 | 0.92          | 0.98
Trousers  			 | 0.91          | 0.98
Jeans  				 | 0.86          | 0.99
Bra  				 | 0.99          | 1.0
Lounge Pants  		 | 0.21          | 0.78
Duffel Bag           | 0.78          | 0.97 
Night suits          | 0.5           | 0.88
Pendant              | 0.85          | 0.97
Tracksuits           | 0.95          | 1.0
Tunics               | 0.37          | 1.0
Nightdress           | 0.29          | 0.85
Ties                 | 0.98          | 0.98
Leggings             | 0.14          | 0.87
Jackets              | 0.80          | 0.95
Messenger Bag        | 0.58          | 1.0 
Mufflers             | 0.13          | 1.0 
Kurta Sets           | 1.0           | 1.0 
Accessory Gift Set   | 1.0           | 1.0 
Bracelet             | 0.80          | 1.0 
Kurtis               | 0.75          | 1.0 
Sweaters             | 0.50          | 0.94 
Ring                 | 0.87          | 1.0
Scarves              | 0.63          | 0.91 
Earrings             | 1.0           | 1.0 
Capris               | 0.73          | 0.98
Skirts               | 0.66          | 0.92 
Headband             | 1.0           | 1.0 
Sports Sandals       | 0.9           | 1.0 
Tights               | 0.0           | unk
Travel Accessory     | 0.2           | 0.9 
Gloves               | 1.0           | 1.0 
Mobile Pouch         | 0.33          | 1.0 
Wristbands           | 1.0           | 1.0 
Stockings            | 0.29          | 0.88 
Footballs            | 1.0           | 1.0
Jumpsuit             | 1.0           | 1.0 
Free Gifts           | 0.08          | 0.16 
Necklace and Chains  | 0.92          | 1.0 
Cufflinks            | 1.0           | 1.0 
Jewellery Set        | 0.69          | 1.0
Laptop Bag           | 0.62          | 0.97 
Camisoles            | 0.26          | 0.91 
Rucksacks            | 0.37          | 1.0 
Swimwear             | 0.12          | 0.62
Stoles               | 0.5           | 1.0 
Basketballs          | 0.63          | 1.0
Rain Jacket          | 1.0           | 1.0
Churidar             | 0.75          | 1.0
Umbrellas            | 1.0           | 1.0
Bangle               | 0.21          | 1.0 
Dupatta              | 0.5           | 1.0 
Water Bottle         | 1.0           | 1.0
Makeup Remover       | 0.66          | 0.66
Rain Trousers        | 1.0           | 1.0
Waist Pouch          | 0.0           | unk
**Average**          | 0.77          | 0.85

​
### Extension ###

It was observed that different model gave drastically different accuracy for different class. Above table is for a model which gave highest overall accuracy.

Moreover, here, ResNet50 was used with different fully connected layer. We can try with more fully connected layers or other type of layers to improve the accuracy.

But, the biggest factor here is class imbalance. Accuracy for top-20 classes on test set is really good. On the other side, accuracy for remaining classes on the test set is very low compare to the top-20 classes.  

There are two common ways to handle the class imbalance. One is to oversample instances with minority class. The second way is to pass weight to the loss function. The above models were trained with the second scheme as it performed better.

We can still notice the class-imbalance problem in the remaining classes. One possible approach to try is to use oversampling along with loss weights. Here, we will oversample classes with really low number of instances OR classes with really low accuracy.

Alternatively, we can also augment images for minority classes.

The productDisplayName also contains helpful information to classify the image. We can possibly train CNN and RNN jointly to classify the image. The CNN will generate encoding for the image and RNN will generate the encoding for the text followed by a fully connected layer to classify the instance.

Alternatively, we can have above model as it is and trained a seperate RNN model to classify class from productDisplayName. Whenever, our CNN model predicts classes with very low confidence, we can pass the productDisplayName to classify the same instance. The class which has highest confidence from both networks can be considered as the output.