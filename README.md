# BERT State-Of-The-Union Modeling
 Re-writing State of the Union addresses with Harry Potter

# Project by : Alexander Nichols and Jessica Warren

# Credits & Foundational Code Sources :

[Base Model](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)
[File Reading](https://github.com/rtealwitter/dl-demos/blob/b537a5dd94953ea656a2140227af78f67c042540/demo08-word2vec.ipynb)
[Linear/Sequential Dimensions](https://theaisummer.com/transformer/)
[Activation Layers](https://pytorch.org/docs/master/special.html)
[Original Paper](https://arxiv.org/pdf/1810.04805.pdf)

This project takes the complete works of Harry Potter by J.K. Rowling, and trains a modified BERT model 
(to fit parameters of our training) on the text. Then, taking a sample few sentences from State of the Union addresses,
we mask all nouns and use our Harry Potter-trained model to replace those nouns with what it thinks is best, to get 
some quite comical results.
