**Comparing MLP against RF and Logistic Regression models using F1-score**

F1 scores of :

a. MLP : 0.919

b. Random Forest : 0.97

c. Logistic Regression : 0.921

*Observations*

1. Random Forest outperforms both MLP and Logistic Regression in terms of F1 score, with the highest F1 score of 0.97.

2. The F1 scores of MLP and Logistic Regression are relatively close, suggesting that the complexity of the MLP architecture might not be necessary for achieving comparable performance to Logistic Regression on this dataset.

**Comparing MLP against RF and Logistic Regression models using Confusion matrix**

1. Random Forest have the fewest misclassifications overall, as indicated by smaller off-diagonal values in the confusion matrix.

2. MLP and Logistic Regression also perform well overall, but they seem to have slightly more misclassifications compared to Random Forest.

3. MLP tends to have a higher number of misclassifications across various classes. There are noticeable off-diagonal values in almost every row, indicating misclassifications across different digits. 
 
4. Random Forest misclassifications seem to be more sporadic, with some digits having very few misclassifications and others having slightly more.

5. Logistic Regression also shows misclassifications across various classes, although the distribution of errors might be slightly different from MLP.

6. In MLP, 8 is mostly misclassified as 5 and 3 as 5.

7. In Logistic Regression, 2 is mostly misclassified as 8, 5 as 8 and 3.

8. In RF, 7 is mostly misclassified as 2, 4 as 9.

**What all digits are commonly confused?**

1. The most common digit confused in all three models is 5.(next one is 8)

2. MLP - 5, 8, 6, RF - 5, 8,  Log Reg -  5, 8 

 **Contrast t-SNE for the second layer for trained and untrained model.**
 
The untrained model plot gives no information, it is random data continously distributed over a range of numbers with mean 0. In the trained model, the plot can represent the relationship between the examples using which we would be able to classify the hand written digits. Very closely related examples are classified together and similarly looking digits are located nearer in the plot than digits with less similarities.

For example 0 is more closer to 8 than 1 in the plot, which makes sense if we observe the hand written samples.

 **Observations on Fashion MNIST data classification**

The Fashion-MNIST data is very poorly classified as the model is trained with features learnt from digits. Although labels are same, they have different meanings. For example in MNIST, label 0 indicates digit 0 whereas in Fashion MNIST, it indicates T-shirt/top. MNIST has hand written digits and fashion MNIST has different fashion labels. By training on digits, it is able to learn features of various digits such as horizontal and vertical lines, intensity etc. By using these patterns to classify fashion items fails because it tries to classify them into digits which they are not.

**Observations on embeddings for MNIST and Fashion-MNIST**

The embeddings are very poor for fashion MNIST which are trained on MNIST dataset as expected. But it is very different from the plot observed for MNIST using untrained model because in the former case the model could recognise certain features say horizontal lines in a t-shirt or pant for example and classify them nearer. So even though the classification is not correct, it is not random as in the latter case.




 
