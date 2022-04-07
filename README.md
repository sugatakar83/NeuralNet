# Neural Net

## Question No.1. Vision Dataset: Please find your dataset from the link -
https://github.com/zalandoresearch/fashion-mnist (Links to an external site.)

Plot at least two samples from each class of the dataset (use matplotlib/seaborn/any other library).
Apply rotation and height shift augmentation (rotation_range, height_shift_range) to the dataset separately. Print the augmented image and the original image for each class and each augmentation.

Sequential Model layers- Use AT LEAST 3 hidden layers with appropriate input for each. Choose the best number for hidden units and give reasons.
Add L2 regularization to all the layers.
Add one layer of dropout at the appropriate position and give reasons.
Choose the appropriate activation function for all the layers.
Print the model summary.

Compile the model with the appropriate loss function.
Use an appropriate optimizer. Give reasons for the choice of learning rate and its value.
Use accuracy as a metric.

Train the model for an appropriate number of epochs. Print the train and validation accuracy and loss for each epoch. Use the appropriate batch size.
Plot the loss and accuracy history graphs for both train and validation set. Print the total time taken for training.

Print the final train and validation loss and accuracy. Print confusion matrix and classification report for the validation dataset. Analyse and report the best and worst performing class.
Print the two most incorrectly classified images for each class in the test dataset.




## Question No.2. NLP Dataset: Please find your dataset from the link - https://ai.stanford.edu/~amaas/data/sentiment/ 

Load the texts and add labels as ‘pos’ and ‘neg’

Print at least two texts from each class of the dataset(pos and neg), for a sanity check that labels match the text.
Plot a bar graph of class distribution in a dataset. Each bar depicts the number of texts belonging to a particular sentiment. (recommended - matplotlib/seaborn libraries)
Any other visualizations that seem appropriate for this problem are encouraged but not necessary, for the points.

Need for this Step - Since the models we use cannot accept string inputs or cannot be of the string format. We have to come up with a way of handling this step. The discussion of different ways of handling this step is out of the scope of this assignment.
Please use this pre-trained embedding layer (Links to an external site.) from TensorFlow hub for this assignment. This link also has a code snippet on how to convert a sentence to a vector. Refer to that for further clarity on this subject.

Sequential Model layers- Use AT LEAST 3 hidden layers with appropriate input for each. Choose the best number for hidden units and give reasons.
Add L2 regularization to all the layers.
Add one layer of dropout at the appropriate position and give reasons.
Choose the appropriate activation function for all the layers.

Compile the model with the appropriate loss function.
Use an appropriate optimizer. Give reasons for the choice of learning rate and its value.
Use accuracy as a metric.

Train the model for an appropriate number of epochs. Print the train and validation accuracy and loss for each epoch. Use the appropriate batch size.
Plot the loss and accuracy history graphs for both train and validation set. Print the total time taken for training.

Print the final train and validation loss and accuracy. Print confusion matrix and classification report for the validation dataset. Analyze and report the best and worst performing class.
Print the two most incorrectly classified texts for each class in the test dataset.
Hyperparameter Tuning- Build two more models by changing the following hyperparameters one at a time. Write the code for Model Building, Model Compilation, Model Training and Model Evaluation as given in the instructions above for each additional model. 

Dropout: Change the position and value of dropout layer
Batch Size: Change the value of batch size in model training
