# -*- coding: utf-8 -*-  
# From https://www.cnblogs.com/zhangtianyuan/p/6922825.html
import gensim
import codecs

def main():
    path_to_model = 'wiki.zhs.model'
    output_file = 'wiki.zh.vec'
    bin2txt(path_to_model, output_file)

def bin2txt(path_to_model, output_file):
    output = codecs.open(output_file, 'w' , 'utf-8')
    model = gensim.models.Word2Vec.load(path_to_model)
    print('Done loading Word2Vec!')  
    vocab = model.wv.vocab  
    for item in vocab:
        vector = list()
        for dimension in model.wv[item]:
            vector.append(str(dimension))
        vector_str = ",".join(vector)
        line = item + "\t"  + vector_str
        output.writelines(line + "\n")
    output.close()

if __name__ == "__main__":
    main()
