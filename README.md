### EC500 K1 Final Project

Components:
 * Obtain the data for SEC Portion: collect10k.py and collect10q.py scripts
 * Parse the data and train/test the mlp model with model_sec.py
 * To perform RNN over twitter data sentiments, use sentiment_rnn.py 
 * Make the final decision with ./combine.sh outrnn outsec where the two files
   are the two outputs of the SEC and Twitter models.

Sources:
 * Twitter_sentiment_DJIA30: https://figshare.com/articles/The_effects_of_Twitter_sentiment_on_stock_price_returns/1533283
 * To produce additional sentiments for tweets we used this GLoVE example: https://github.com/laviavigdor/twitter-sentiment-analysis
 * For sentiment_rnn, we referred to https://github.com/aymericdamien/TensorFlow-Examples for portions of the implementation


