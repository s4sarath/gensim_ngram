gensim – Topic Modelling in Python
==================================


This is an extension of gensim model, which helps to create a **N-gram model**. Unlike using some phrases, this model is making use of **N grams as context and center words**. 

Have a look at **train_ngram.py** for a sample training scripts.

Written by modifying gensim source code, but not supporting GIL, as I am not familiar with Cython, but still faster 

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

Features
--------

-   This **not-memory-optimized** as gensim, but
    (can process input larger than RAM, streamed, out-of-core),
-   **Intuitive interfaces**
    -   Training n-grams ( but it is recommended to stop at trigrams, bigram + unigram with min_count = (10,5)
        respectively has around 17 million wordds
    -   Supports n gram training for word2vec and fasttetx
-   A previously trained model on WikiData + some review datasets will be available on the following link
    **Latent Semantic Analysis (LSA/LSI/SVD)**
    
Sample Results
--------------

model.wv.most_similar

a.) amazing product

[('amazing product', 1.0),
 ('awesome product', 0.9272927),
 ('amazing product,', 0.888031),
 ('incredible product', 0.8867724),
 ('amazing product!', 0.88521475),
 ('amazing product.', 0.8845437),
 ('awesome product!', 0.8644207),
 ('amazing product!!', 0.8612526),
 ('amazing product!!!', 0.85835207),
 ('awesome product.', 0.8530247),
 ('awesome product!!', 0.8516336),
 ('awesome product,', 0.8495761),
 ('awesome item', 0.8434567),
 ('product. amazing', 0.84247625),
 ('incredible product.', 0.8421844),
 ('awesome product!!!', 0.84074044),
 ('wonderful product', 0.8406575),
 ('awesome device', 0.836467),
 ('incredible product!', 0.8337494),
 ('fantastic product', 0.8330554)]


b.) brad pitt

[('brad pitt', 1.0),
 ('julia roberts', 0.84390914),
 ('angelina jolie', 0.84303164),
 ('ben affleck', 0.8231394),
 ('matt damon', 0.81166387),
 ('affleck', 0.8074477),
 ('george clooney', 0.80540144),
 ('costner', 0.80255926),
 ('tom hanks', 0.8017744),
 ('dustin hoffman', 0.79872185),
 ('natalie portman', 0.798303),
 ('ryan gosling', 0.79511935),
 ('dicaprio', 0.79246503),
 ('kevin spacey', 0.7921234),
 ('alec baldwin', 0.7907918),
 ('actor brad', 0.7901952),
 ('russell crowe', 0.78980654),
 ('kevin costner', 0.7894964),
 ('christopher walken', 0.7882538),
 ('jennifer aniston', 0.7878684)]


c.) mohanlal

[('mohanlal', 1.0),
 ('mammootty', 0.9794469),
 ('kamal haasan', 0.9596181),
 ('haasan', 0.9563364),
 ('rajkumar', 0.95312166),
 ('gopi', 0.9529321),
 ('sivaji', 0.95167804),
 ('madhavan', 0.9510826),
 ('dileep', 0.95085794),
 ('chiranjeevi', 0.95059955),
 ('jayaram', 0.9503455),
 ('nagesh', 0.9484335),
 ('sathyaraj', 0.9479996),
 ('rajinikanth', 0.94777143),
 ('suresh gopi', 0.9466225),
 ('sivaji ganesan', 0.94393903),
 ('prakash raj', 0.9437847),
 ('sathyan', 0.9431832),
 ('prabhu', 0.942392),
 ('bharath', 0.9391954)]


d.) machine learning

[('machine learning', 1.0000001),
 ('learning algorithms', 0.8841063),
 ('data mining', 0.8291545),
 ('machine translation', 0.814913),
 ('support vector', 0.80520463),
 ('algorithms', 0.8029659),
 ('learning theory', 0.8026564),
 ('algorithms and', 0.80255526),
 ('information retrieval', 0.7991563),
 ('neural networks', 0.7982512),
 ('vector machines', 0.79787594),
 ('machine intelligence', 0.79575825),
 ('learning algorithm', 0.7918976),
 ('reinforcement learning', 0.7897328),
 ('language processing', 0.78945714),
 ('and computational', 0.7862742),
 ('vector machine', 0.78508246),
 ('knowledge representation', 0.7850384),
 ('algorithmic', 0.7817018),
 ('distributed systems', 0.7809721)]


e.) mortal kombat

[('mortal kombat', 0.99999994),
 ('kombat', 0.92918265),
 ('tekken', 0.855644),
 ('kombat ii', 0.8423183),
 ('virtua fighter', 0.82694477),
 ('soulcalibur', 0.8240025),
 ('ninja gaiden', 0.8233547),
 ('darkstalkers', 0.8189633),
 ('kombat vs', 0.8051237),
 ('kombat armageddon', 0.80245066),
 ('kombat series', 0.80217266),
 ('samurai shodown', 0.8003039),
 ('resident evil', 0.8001634),
 ('game mortal', 0.7937777),
 ('in capcom', 0.7936872),
 ('kombat mortal', 0.7936853),
 ('mortal', 0.79330146),
 ('kombat deception', 0.7923815),
 ('onimusha', 0.7913557),
 ('virtua', 0.79038495)]


f.) nissan

[('nissan', 1.0000002),
 ('mazda', 0.9355751),
 ('toyota', 0.89277387),
 ('lexus', 0.89011514),
 ('subaru', 0.8749101),
 ('toyota corolla', 0.86015534),
 ('nissan skyline', 0.85717183),
 ('mazda rx', 0.8544719),
 ('volkswagen', 0.8482176),
 ('bmw', 0.84316957),
 ('mitsubishi', 0.8426397),
 ('honda', 0.8378298),
 ('infiniti', 0.8358605),
 ('celica', 0.83509576),
 ('chevrolet corvette', 0.8315984),
 ('isuzu', 0.8309591),
 ('nissan gt', 0.8307908),
 ('datsun', 0.8291819),
 ('chevrolet', 0.8271923),
 ('opel', 0.8265841)]

