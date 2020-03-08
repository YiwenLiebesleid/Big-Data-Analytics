# Trajectory Data Classification
### Proposal
Using GPS data from 5 taxi drivers in 179 days to classify. The model should be able to predict which driver a trajectory belongs to given a whold day's GPS data.

### Data Description
The raw data has these features: [plate(the label to be predicted), longitude, latitude, time, status(1 as the taxi is occupied)].

### Preprocessing
I didn't upload the first preprocessing file here. That file will drop those days with less than 20,000 data point out, transform time into a calculatable form, and a new feature "velocity", and combine all 150 files into one csv file.

The formal preprocessing process is included in datagen.py. The time will be transformed into cosine and sine format, and all of these features will be normalized. Another thing to mention is that some global features are also extracted here.

### Training
I use LSTMs in this project, 3 LSTMs is the basic part of my model to deal with trajectory. To make use of global features, I add another Dense layer, and concatenate the trajectory part and features part.

Activation function is ReLU, I have proved in report that it works far more better than sigmoid.

For other hyperparameters learning rate and dropout, the selection of value is based on the experiment result I made.

For other training strategy, early stopping is used to prevent overfitting.

Cross validation is implemented in this process. An average of 0.81 is reached for validation accuracy. In some good cases, it can reach more than 0.85.

### Evaluation
This project is evaluated to have an accuracy of 0.84 by TA.

---

Of course, I'm still trying to make a progress.
