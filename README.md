# NER
一个中文的实体命名识别系统

当前版本基于双向循环神经网络（BiRNN）+ 条件随机场（CRF）来完成实体的标注。 基本思路是利用深度神经网络提取特征，从而避免了手动提取特征的麻烦。
第二步和传统的方式一样，使用CRF在做最后的标注。

该程序使用Tensorflow完成，使用了当前较新的DataSet API，使数据预处理和feed更优雅。

如何使用？

    1. 建议安装tensorflow = 1.8.0

    2. 提供训练所需的数据，具体格式在resource文件夹里有展示。但是需要自行分词。只需提供3个文件：
        source.txt target.txt 和 预训练的词向量。

    3. 训练词向量，训练工具可以是gensim的word2vector或者glove等等，然后将词和对应的词向量以以下格式保存。
        具体格式是： 
        单词A 0.001 0.001 0.001 ....
        单词B 0.001 0.001 0.001 ....
        单词C 0.001 0.001 0.001 ....
        .
        .
        .
        有些训练工具得出的文件结果就是以上格式不需要修改，程序默认embedding size是400。
        使用gensim训练好并转换格式的词向量下载地址：https://pan.baidu.com/s/1u5my22zmHqDy8TUudx8x1A 密码：t9gz
        下载后解压到resource文件夹即可。
        gensim训练词向量的方法和数据链接：https://docs.qingcloud.com/product/ai/deeplearning/#nlp

    4. 修改config.py里的文件存路径，所有的配置都在这个文件里。

    5. 训练
        $ python ner.py --mode train

    6. 预测：修改args.py，设置要加载的checkpoint路径
        parser.add_argument('--demo_model', type=str, default='1537601731', help='model for test and demo')
        $ python ner.py --mode demo


注意：
    原本resource文件中只包含predict.txt, source.txt, target.txt, 如果更换自己的词向量文件记得删除其他自动生成的文件。
