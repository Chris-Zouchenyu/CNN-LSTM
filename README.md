# CNN-LSTM
This is a model for stock prediction that uses the CNN-LSTM network!  
We have all known that fully connected neural network couldn't deal well with the data that had time seies.Because the data used for fully connected neural networks needs to maintain independence between each other.To address this issue, LSTM was proposed and has been well applied in semantic prediction, speech recognition, and other problems. The addition of a forget gate in the algorithm effectively solved the problem of long data being easily forgotten in its predecessor, RNN networks.You can find more details in:  
LSTM: https://arxiv.org/abs/1402.1128.  
In this project, I used a combination of Convolutional Neural Network (CNN) and LSTM algorithm to predict data for a portion of stocks, which came from Ping An's daily data. The data used included opening, closing, and trading volume, sourced from:   https://gushitong.baidu.com/stock/ab-601318.  
The core of the algorithm is to convert sequence data into image data. For sequence format data (sequence number, window size, feature number), it can be regarded as multiple images with window size * feature number, and thus CNN can be used for training.  
The module is here:
![image](https://github.com/Chris-Zouchenyu/CNN-LSTM/blob/main/CNN-LSTM.jpg)
The predicted results are shown in the figure：  
![image](https://github.com/Chris-Zouchenyu/CNN-LSTM/blob/main/output.png)  

