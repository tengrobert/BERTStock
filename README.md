# Stock Prediction using Numerical and Textual Information with Deep Learning
This project use BERT pre-trained language model with Donald Trump's tweets to predict the stock price.

### Project structure
- data.csv - Donald Trump's historic tweets
- SPY.csv - historic stock price
- pre.py - preprocessing script for creating training data
- plot.py - generate result figure
- test.py - evaulation script
- train.py - train the model script

### To run the project
1. Firstly, run the preprocessing script
```
python3 pre.py
```
2. Run the following script to train the model
```
python3 train.py
```
3. Run the following script to evaulate the result
```
python3 test.py
```

### Results
- Our results (red means time to buy, blue means time to sell)   
![image](https://lh5.googleusercontent.com/v47PP4uQy-09cvRdhcGsOAQM65bhKP2Lu_TTW-djvn8vhrZkyrzQHe2Rsr7Qpt5IiK_jJgvz6B9AN-D6Vsdv2yHcE_u2gc0ePWFDKDJLhNyNJSsgpPE0kyn134cvt4UuQE3z7xTLVngXXrHU0jPsvQ)

- Traditional RNN results
![image](https://lh4.googleusercontent.com/_aZaAqNmWGrObDSUEKpI71YN1YE0CQ2jCei7XJNFOZasr8yIdEFf_QnxWxAvcnkhwzG03CshExwkmcKtIUXZ4fzN5EyDpFEiunTjojbjsTHBwWuEU6wkm3MK-8tJ3IgZdGQ3UzAQ0M0YOjODs9qCUw)

- Loss over steps
![image](https://drive.google.com/uc?export=view&id=1E72yxDdHq1OO5CfrOtSjI0JHrQfeauGC)

- demo video
[demo video](https://drive.google.com/file/d/1Po6Rqb9omaI7aOn0DYZws-O0Uxh58OmB/view?usp=sharing)
