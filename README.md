# ECE_657
ECA Chatbot project myself and Ye Fan developed for ECE 657.

## Main Description
Abstract—There has been a large push in recent years to
develop improved emotional conversational agents (chatbots)
from both an industry and academia standpoint. An important
application that has been rarely explored to this date, is the
usage of an empathetic, counselling conversational agent. To this
end, in this paper we propose the Empathetic Conversational
Agent (ECA) which is designed to generate emotional and
empathetic responses to the user. ECA is built upon a sequence to
sequence (Seq2Seq) model utilizing LSTM layers, and is trained
on specifically designed empathetic dialogue (ED) corpus. Most
importantly, ECA utilizes emotion enriched word embeddings
that are built on the global vectors for word representation
(GloVe) embeddings. The combination of these factors allows
ECA to adjust its tone and responses in an empathetic manner.
We compared ECA to three other chatbot models including a
model trained on the Cornell movie dialogue corpus (Baseline
Chatbot), a model trained on the ED corpus without pre-trained
embedding (baseline ECA) and a model trained on the ED corpus
using GloVe embedding (GloVe ECA). We found that all 3 models
trained on the ED corpus have superior performance based on
human evaluation in terms of relevance, understandability and
emotional awareness. Overall, participants reported that ECA
and GloVe ECA have significantly (p<.05) higher relevance, and
understandability scores. Additionally, ECA was rated as having
the highest emotional awareness although it was not statistically
significantly higher than GloVe ECA. The perplexity and cosine
similarity were calculated as quantitative evaluation metric for
the chatbot models, though these metrics alone are not sufficient
to fairly represent the model performance. In conclusion, we
present a small, well tuned, generative conversational agent
that has the capability of producing empathetic responses in
emotionally grounded conversations.

## Important Links 
1. Chatbot Tutorial: https://medium.com/swlh/how-to-design-seq2seq-chatbot-using-keras-framework-ae86d950e91d
2. Tutorial repo: https://github.com/dredwardhyde/Seq2Seq-Chatbot-English Tutorial repo
3. Kaggle Data: https://www.kaggle.com/hassanamin/chatbot-nlp
4. Sentiment Analysis: https://github.com/cjhutto/vaderSentiment

## Datasets Used
1. Empathetic Dialogues: https://arxiv.org/abs/1811.00207
2. Cornell Movie Dialogue Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
3. Emotion Word Embeddings (EWE): https://www.aclweb.org/anthology/C18-1081.pdf
