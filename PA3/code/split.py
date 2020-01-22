source = "code-fall2019-a3/NLP_class/data5/training-data/training-data.1m"
scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
size = 1000000
for scale in scales:
    destination = "code-fall2019-a3/NLP_class/data5/training-data/training-data." + str(scale) + "m"
    with open(source, 'r') as src, open(destination, 'w') as dst:
        for i, line in enumerate(src):
            dst.write(line)
            if i == scale * size:
                break