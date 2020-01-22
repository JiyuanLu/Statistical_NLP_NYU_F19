import fasttext

train_path = "code-fall2019-a3/NLP_class/data5/training-data/training-data.1m"

'''
# default model
skipgram_model = fasttext.train_unsupervised(train_path, model='skipgram')
cbow_model = fasttext.train_unsupervised(train_path, model='cbow')
skipgram_model.save_model("models/skipgram.bin")
cbow_model.save_model("models/cbow.bin")
'''
# change dimensions
'''
# dim 3
model = fasttext.train_unsupervised(train_path, model='skipgram', dim=3)
model.save_model("models/dim3.bin")
'''

dims = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for dim in dims:
    model = fasttext.train_unsupervised(train_path, model='skipgram', dim=dim)
    model.save_model("models/dim" + str(dim) + ".bin")
'''
# change loss function
# hs
model = fasttext.train_unsupervised(train_path, model='skipgram', loss='hs')
model.save_model("models/hs.bin")
'''

'''
# softmax
model = fasttext.train_unsupervised(train_path, model='skipgram', loss='softmax')
model.save_model("models/softmax.bin")

# ova
model = fasttext.train_unsupervised(train_path, model='skipgram', loss='ova')
model.save_model("models/ova.bin")
'''

'''
# change amount of training data
scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for scale in scales:
    train_path = "code-fall2019-a3/NLP_class/data5/training-data/training-data." + str(scale) + "m"
    model = fasttext.train_unsupervised(train_path)
    model.save_model("models/" + str(scale) + ".bin")
'''