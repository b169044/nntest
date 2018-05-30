#-*- coding: UTF-8 -*-
# https://blog.csdn.net/m0_37650263/article/details/77343220
# https://blog.csdn.net/leiting_imecas/article/details/71246541
import numpy as np
import tensorflow as tf
import random
from sklearn.feature_extraction.text import CountVectorizer
import os
import csv

real_dir_path = os.path.split(os.path.realpath(__file__))[0]
file = os.path.join(real_dir_path, 'data_train.csv')
file2 = os.path.join(real_dir_path, 'data_test.csv')
nn = 1

for b in ['食品餐饮', '旅游住宿', '金融服务', '医疗服务', '物流快递']:
    print(b)
    # str =''
    # list = []
    # with open(file, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         str = line.split('\t')[1]
    #         try:
    #             list.index(str)
    #         except:
    #             list.append(str)
    #             print(str)
    # print(list)
    #[ '食品餐饮','旅游住宿', '金融服务', '医疗服务', '物流快递']

    from pyltp import Segmentor, Postagger
    seg = Segmentor()
    seg.load('cws.model')
    poser = Postagger()
    poser.load('pos.model')
    real_dir_path = os.path.split(os.path.realpath(__file__))[0] #文件所在路径
    stop_words_file = os.path.join(real_dir_path, 'stopwords.txt')
    #定义允许的词性
    allow_pos_ltp = ('a', 'i', 'j', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v', 'ws')

    def cut_stopword_pos(s):
        words = seg.segment(''.join(s.split()))
        poses = poser.postag(words)
        stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words_file,'r', encoding='UTF-8')])
        sentence = []
        for i, pos in enumerate(poses):
            if (pos in allow_pos_ltp) and (words[i] not in stopwords):
                sentence.append(words[i])
        return sentence

    def create_vocab(file):
        def process_file(file_path):
            with open(file_path, 'r') as f:
                v = []
                lines = f.readlines()
                for line in lines:
                    if(line.split('\t')[1]==b):
                        sentence = cut_stopword_pos(line.split('\t')[2])
                        v.append(' '.join(sentence))
                print('finished')
                return v
        sent = process_file(file)
        tf_v = CountVectorizer(max_df=0.9, min_df=1)
        tf = tf_v.fit_transform(sent)
        #print tf_v.vocabulary_
        return list(tf_v.vocabulary_.keys())

    vocab = create_vocab(file)
    print('vocab')

    def normalize_dataset(vocab):
        dataset = []
        # vocab:词汇表; review:评论; clf:评论对应的分类, 0：负面、1：中性、2：正面
        def string_to_vector(vocab, review, clf):
            words = cut_stopword_pos(review) # list of str
            features = np.zeros(len(vocab))
            clfs = np.zeros(3)
            clfs[int(clf)] = 1
            for w in words:
                if w in vocab:
                    features[vocab.index(w)] = 1
            return [features, clfs]
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if (line.split('\t')[1] == b):
                    one_sample = string_to_vector(vocab, line.split('\t')[2], line.split('\t')[-1].replace('\n',''))
                    dataset.append(one_sample)
            print('finished2')
        return dataset

    dataset = normalize_dataset(vocab)
    print('dataset')
    #取样本的10%作为测试数据
    test_size = int(len(dataset) * 0.1)
    dataset = np.array(dataset)
    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]
    print ('test_size = {}'.format(test_size))

    def normalize_dataset2(vocab):
        dataset = []
        # vocab:词汇表; review:评论; id:序号
        def string_to_vector(vocab, review, id):
            words = cut_stopword_pos(review) # list of str
            features = np.zeros(len(vocab))
            for w in words:
                if w in vocab:
                    features[vocab.index(w)] = 1
            return [id,features]
        with open(file2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if (line.split('\t')[1] == b):
                    one_sample = string_to_vector(vocab, line.split('\t')[2], line.split('\t')[0].replace('\n',''))
                    dataset.append(one_sample)
            print('finished3')
        return dataset
    #需要预测的数据
    dataset2 = normalize_dataset2(vocab)

    #定义神经网络的输入输出结点，每个样本为1*315维，以及输出分类结果
    INPUT_NODE=len(vocab)
    OUTPUT_NODE=3

    #定义两层隐含层的神经网络，一层500个结点，一层300个结点
    LAYER1_NODE=500
    LAYER2_NODE=300

    #定义学习率，学习率衰减速度，正则系数，训练调整参数的次数以及平滑衰减率
    LEARNING_RATE_BASE=0.5
    LEARNING_RATE_DECAY=0.99
    REGULARIZATION_RATE=0.0001
    TRAINING_STEPS=20
    MOVING_AVERAGE_DECAY=0.99
    batch_size = 50


    #定义整个神经网络的结构，也就是向前传播的过程，avg_class为平滑可训练量的类，不传入则不使用平滑
    def inference(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
        if avg_class==None:
            #第一层隐含层，输入与权重矩阵乘后加上常数传入激活函数作为输出
            layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
            #第二层隐含层，前一层的输出与权重矩阵乘后加上常数作为输出
            layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2)
            #返回 第二层隐含层与权重矩阵乘加上常数作为输出
            return tf.matmul(layer2,w3)+b3
        else:
            #avg_class.average()平滑训练变量，也就是每一层与上一层的权重
            layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
            layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))
            return tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3)


    #定义输出数据的地方，None表示无规定一次输入多少训练样本,y_是样本标签存放的地方
    x = tf.placeholder('float', [None, INPUT_NODE])  # None表示样本数量任意; 每个样本纬度是term数量
    y_ = tf.placeholder('float')

    #依次定义每一层与上一层的权重，这里用随机数初始化，注意shape的对应关系
    w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
    b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
    b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))

    w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
    b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #输出向前传播的结果
    y =inference(x,None,w1,b1,w2,b2,w3,b3)

    # 每训练完一次就会增加的变量
    global_step = tf.Variable(0, trainable=False)

    # 定义平滑变量的类，输入为平滑衰减率和global_stop使得每训练完一次就会使用平滑过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 将平滑应用到所有可训练的变量，即trainable=True的变量
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 输出平滑后的预测值
    average_y = inference(x, variable_averages, w1, b1, w2, b2, w3, b3)

    # 定义交叉熵和损失函数，但为什么传入的是label的arg_max(),就是对应分类的下标呢，我们迟点再说
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_,1))
    # 计算交叉熵的平均值，也就是本轮训练对所有训练样本的平均值
    cross_entrip_mean = tf.reduce_mean(cross_entropy)

    # 定义正则化权重，并将其加上交叉熵作为损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w1) + regularizer(w2) + regularizer(w3)
    loss = cross_entrip_mean + regularization

    # 定义动态学习率，随着训练的步骤增加不断递减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 900, LEARNING_RATE_DECAY)
    # 定义向后传播的算法，梯度下降发，注意后面的minimize要传入global_step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 管理需要更新的变量，传入的参数是包含需要训练的变量的过程
    train_op = tf.group(train_step, variable_averages_op)

    # 正确率预测
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        # 初始所有变量
        init_op=tf.initialize_all_variables()
        sess.run(init_op)
        # 训练集输入字典
        train_x = train_dataset[:, 0]  # 每一行的features;
        train_y = train_dataset[:, 1]  # 每一行的label

        # 测试集输入字典
        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        test_feed = {x: list(test_x), y_: list(test_y)}

        for i in range(TRAINING_STEPS):
            j = 0
            while j < len(train_x):
                start = j
                end = j + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                validate_feed = {x: list(batch_x), y_: list(batch_y)}
                sess.run(train_op, feed_dict=validate_feed)
                j = end
            validate_acc = sess.run(accuracy, feed_dict=test_feed)
            print("After %d training step(s),validation accuracy using average model is %g" % (i, validate_acc))
        # 用测试集查看模型的准确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))
        #保存模型
        saver = tf.train.Saver()
        model_path = 'saved_model/model'+str(nn)+'.ckpt'
        saver.save(sess, model_path)
        nn +=1

        out = open('result.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        for i in dataset2:
            ceshi_x = i[1]  # 每一行的features;
            id = i[0]  # 每一行的label
            yy = sess.run(y,feed_dict={x:[ceshi_x]})
            s = id+','+str(np.argmax(yy))
            csv_write.writerow([s])
