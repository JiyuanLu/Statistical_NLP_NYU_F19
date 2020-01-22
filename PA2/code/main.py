from Models import *
from matplotlib import pyplot as plt

# 22866 features
# 43031 features with first_word and last_word
customized_trigram = CustomizedTrigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
objectives, train_accs, val_accs = customized_trigram.train()
customized_trigram.produceTestOutput()

customized_train_acc = customized_trigram.evaluate("train")
customized_val_acc = customized_trigram.evaluate("val")
print(customized_train_acc)
print(customized_val_acc)

f1 = plt.figure(1)
plt.plot(objectives)
plt.title("Objective History Every 100 Iterations")
plt.xlabel("t")
plt.ylabel("objective")
plt.show()

f2 = plt.figure(2)
plt.title("Training Accuracy History Every 100 Iterations")
plt.xlabel("t")
plt.ylabel("training accuracy")
plt.plot(train_accs)
plt.show()

f3 = plt.figure(3)
plt.title("Validation Accuracy History Every 100 Iterations")
plt.xlabel("t")
plt.ylabel("validation accuracy")
plt.plot(val_accs)
plt.show()

'''
unigram = UnigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
objectives, train_accs, val_accs = unigram.train()
unigram.produceTestOutput()

f1 = plt.figure(1)
plt.plot(objectives)
plt.title("Objective History Every 10 Iterations")
plt.xlabel("t")
plt.ylabel("objective")
plt.show()

f2 = plt.figure(2)
plt.title("Training Accuracy History Every 10 Iterations")
plt.xlabel("t")
plt.ylabel("training accuracy")
plt.plot(train_accs)
plt.show()

f3 = plt.figure(3)
plt.title("Validation Accuracy History Every 10 Iterations")
plt.xlabel("t")
plt.ylabel("validation accuracy")
plt.plot(val_accs)
plt.show()
'''

'''
# 21629 features with first_word and last_word
customized_unigram = CustomizedUnigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
customized_unigram.train()
customized_train_acc = customized_unigram.evaluate("train")
customized_val_acc = customized_unigram.evaluate("val")
print(customized_train_acc)
print(customized_val_acc)

'''
'''
# 24503 features with first_word and last_word
customized_bigram = CustomizedBigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
customized_bigram.train()
customized_train_acc = customized_bigram.evaluate("train")
customized_val_acc = customized_bigram.evaluate("val")
print(customized_train_acc)
print(customized_val_acc)
customized_bigram.produceTestOutput()
'''

'''
customized_trigram = CustomizedTrigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
customized_trigram.train()
customized_train_acc = customized_trigram.evaluate("train")
customized_val_acc = customized_trigram.evaluate("val")
print(customized_train_acc)
print(customized_val_acc)
'''
'''
bigram = BigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
bigram.train()
train_acc = bigram.evaluate("train")
val_acc = bigram.evaluate("val")
print(train_acc)
print(val_acc)
'''

'''
trigram = TrigramModel("../data/pnp-train.txt", "../data/pnp-validate.txt", "../data/pnp-test.txt")
trigram.train()
train_acc = trigram.evaluate("train")
val_acc = trigram.evaluate("val")
print(train_acc)
print(val_acc)
'''   

