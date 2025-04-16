# Bird vs Drone Classification


This project performs classification of images of drones and birds using neural networks with fine-tuning of MobileNetV2. Built in Google Colab using TensorFlow and Keras, the pipeline processes a dataset of approximately 800 labeled images, automatically split into training (70%), validation (20%), and test (10%) sets using the splitfolders library. To enhance performance on the small dataset, image augmentation techniques such as rotation, zoom, flipping, and brightness adjustments are applied. MobileNetV2, pre-trained on ImageNet, is used as the base model with additional custom classification layers. Due to the limited size of the dataset, results can vary significantly depending on how the data is split. To address this, the training process is repeated 10 times with different random seeds, and the results are averaged to provide a more reliable and consistent evaluation. The model achieved an average test accuracy of 96%, reflecting strong and consistent performance across all runs. Throughout the project, I used various tools and libraries for data augmentation, statistical evaluation, and building efficient image classification models, including TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Scikit-learn.


![image](https://github.com/user-attachments/assets/24c403b2-c794-428e-91af-cf793ed3cf1f)

The graph shows the average training and validation accuracy over 30 epochs across 10 runs. Both accuracies increase rapidly during the initial epochs, reaching high performance. However, starting around epoch 15, training accuracy continues to improve while validation accuracy begins to plateau and fluctuate slightly, which may suggest that the model is starting to overfit slightly. Despite this, the gap remains small, suggesting that the model still generalizes relatively well.


![image](https://github.com/user-attachments/assets/6cbac797-4444-4e7f-a1ff-207f41d1bb14)

The graph shows the average training and validation loss over 30 epochs across 10 runs. Both losses decrease steadily, especially in the early epochs, indicating effective learning. Starting around epoch 15, the training loss continues to decline, while the validation loss begins to flatten, which aligns with the slight overfitting observed in the accuracy graph. However, the two curves remain close throughout the training, suggesting that overfitting is minimal and the model maintains good generalization.


![image](https://github.com/user-attachments/assets/c818fd2f-c2b4-440c-9fec-c616f26481e4)

The average confusion matrix shows strong classification performance. The model correctly identified birds and drones with high accuracy across the 10 runs, averaging approximately 35.8 correct bird predictions and 41.9 correct drone predictions. Misclassifications are minimal, with only about 1.9 birds misclassified as drones and 1.1 drones misclassified as birds. This indicates a well-balanced model with strong generalization and minimal bias toward either class.

