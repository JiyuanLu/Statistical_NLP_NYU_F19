import fasttext
import os

'''
# default model
model_path = "models/skipgram.bin"
embedding_path = "embeddings/skipgram.embedding"
model = fasttext.load_model(model_path)

with open(embedding_path, 'w') as f:
    dim = model.get_dimension()
    num_of_words = len(model.words)
    f.write(str(num_of_words) + " " + str(dim))
    for word in model.words:
        f.write('\n')
        f.write(word)
        for value in model[word]:
            f.write(" " + str(value))
'''
# change dimensions
'''
# dim 3
model_path = "models/dim3.bin"
embedding_path = "embeddings/dim3.embedding"
model = fasttext.load_model(model_path)

with open(embedding_path, 'w') as f:
    dim = model.get_dimension()
    num_of_words = len(model.words)
    f.write(str(num_of_words) + " " + str(dim))
    for word in model.words:
        f.write('\n')
        f.write(word)
        for value in model[word]:
            f.write(" " + str(value))
'''

# dims = [10, 20, 30, 40, 50, 60, 70, 80, 90]
dims = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for dim in dims:
    model_path = "models/dim" + str(dim) + ".bin"
    embedding_path = "embeddings/dim" + str(dim) + ".embedding"
    model = fasttext.load_model(model_path)

    with open(embedding_path, 'w') as f:
        dim = model.get_dimension()
        num_of_words = len(model.words)
        f.write(str(num_of_words) + " " + str(dim))
        for word in model.words:
            f.write('\n')
            f.write(word)
            for value in model[word]:
                f.write(" " + str(value))

'''
# change scales
scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for scale in scales:
    model_path = "models/" + str(scale) + ".bin"
    embedding_path = "embeddings/" + str(scale) + ".embedding"
    model = fasttext.load_model(model_path)
    with open(embedding_path, 'w') as f:
        dim = model.get_dimension()
        num_of_words = len(model.words)
        f.write(str(num_of_words) + " " + str(dim))
        for word in model.words:
            f.write('\n')
            f.write(word)
            for value in model[word]:
                f.write(" " + str(value))
'''