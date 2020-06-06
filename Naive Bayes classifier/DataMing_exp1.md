<h2 align='center'>《数据挖掘技术概论》实验报告</h2>
<h2 align='center'>实验一  朴素贝叶斯分类实验</h2>

<h3 align='left'><b>一、实验目的</b></h3>

1. 手动实现朴素贝叶斯分类算法，深入地理解该算法的基本原理。
2. 进一步熟悉基本的程序设计方法。
3. 对带类标的数据集——泰坦尼克号船员特征及生存情况运用贝叶斯分类器进行预测，输出每个样例属于每个类别的后验概率值，并评估该分类器的准确率。

<h3 align='left'><b>二、算法描述和分析</b></h3>

&emsp;朴素贝叶斯分类法是基于贝叶斯理论的一种算法，该算法从有限的训练样本集估计出未知样本属于某一类别的后验概率，从而使得后验概率最大的类别作为该样本的最终预测类别。

1. **先验概率的计算**

    若有充足的独立同分布样本，则可以容易地估计出类先验分布概率, $D_c$表示训练集D中第c类样本组成的集合，则

    $$
    P(c) = \frac{|D_c|}{|D|}
    $$
2. **后验概率的计算**

    令$D_{c,x_i}$表示$D_c$在第i个属性熵取值为$x_i$的样本组成的集合，则条件概率$P(x_i | c)$可估计为

    $$
    P(x_i | c) = \frac{|D_{c,x_i}|}{|D_c|}
    $$

    基于属性条件独立性假设，后验概率可重写为

    $$
    P(c|x) = \frac {P(c)P(x|c)}{P(x)} = \frac {P(c)}{P(x)}\prod_{i=1}^d P(x_i | c)
    $$
3. **连续属性离散化技术**

    采用二分法处理：
    给定样本集D和连续属性a，假定a在D上出现了n个不同的取值，将这些值从小到大进行排序，记为${a^1.a^2,\dots ,a^n}$,基于划分点t将D分为子集$D^-_t$和$D^+_t$，其中$D^-_t$包含属性a上取值不大于t的样本，而$D^+_t$则包含在属性a上取值大于t的样本，考察包含n-1个元素的候选划分点集合

    $$
    T_a=\left[
            \frac{a^i\,+\,a^{i+1}}{2}\,|\,1\leq i \leq {n-1} 
        \right]
    $$
    即把区间的中位点$\frac{a^i\,+\,a^{i+1}}{2}$作为候选划分点
    再根据信息增益的最大化标准来选择划分点。
    
    信息熵是度量样本集合纯度最常用的一种指标，假定当前样本集合D中第k类样本所占比例为$p_k (k = 1,2,...|\Upsilon|)$,则D的信息熵定义为

    $$
    Ent(D) = -\sum_{k=1}^{|\Upsilon|} p_klog_2p_k .
    $$ 
    假定离散属性值a有V个可能的取值${a^1,a^2,...,a^V}$,若使用a对样本集D进行划分，则会产生V个分支节点，其中第v个分支节点包含了D中所有在属性a上取值为$a^v$的样本，记位$D^v$. 可根据上式计算$D_v$的信息熵，考虑道不同的分支节点所包含的样本数不同，给分支结点赋予权重$\frac {|D^v|}{|D|}$,即样本数越多的分支结点影响越大，于是可计算出用属性a对样本集D进行划分所获得的信息增益：

    $$
    Gain(D, a) = Ent(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
    $$
    一般而言，信息增益越大，则意味者使用属性a来进化划分所获得的"纯度提升"越大，即选择属性$a_* = arg\,max\,Gain(D,a)\,,a\in A$来进行划分

<h3 align='left'><b>三、程序实现技术技巧的介绍和分析</b></h3>

1. **数据清洗**

    1.1 加载数据集

    读取数据集，将非注释行读入保存，打乱数据集后按照70%，30%划分训练集和验证集

        def load_dataset(path,train_percentage): 
        dataset = []
        i = 0
        with open(path,'r') as file:
            dat = reader(file, delimiter=',')
            for row in dat:
                # 读取dat文件，并忽略前8行注释
                if(i>7):
                    # 将字符串类型转化为浮点型
                    row[0:4] = list(map(float, row[0:4]))
                    dataset.append(row)
                i += 1
        # 将类别标签转化为整形
        for row in dataset:
            row[3] = int(row[3])
        dataset = np.array(dataset)
        dataset = split_attribute(dataset) 
        # 打乱数据集 
        random.shuffle(dataset)
        # 划分训练集和验证集
        n_train_data = round(len(dataset)*train_percentage)
        train_data = dataset[0:n_train_data]
        val_data = dataset[n_train_data:]
        return train_data, val_data
    1.2 信息增益的计算方法

        def information_entropy(dataset):
        """
        计算给定数据集下，按照两类分的情况的信息熵
        :param dataset:数据集
        :return:信息熵
        """
        if(len(dataset) == 0):
            return 0
        positive = 0
        negative = 0
        for row in dataset:
            if(row[-1] == 1):
                positive += 1
            else:
                negative += 1
        Pr_P = positive / len(dataset)
        Pr_N = negative / len(dataset)
        Ent_ = -(Pr_P*math.log2(Pr_P) + Pr_N*math.log2(Pr_N))
        return Ent_
    
        def information_gain(dataset, index_attribute, value):
        """
        计算给定属性、按照给定值做划分的信息增益
        :param dataset:数据集
        :param index_attribute:属性序列
        :param value:当前属性下的给定取值
        :return ：信息增益
        """
        Ent_D = information_entropy(dataset)
        new_dataset_P = []
        new_dataset_N = []
        for row in dataset:
            if(row[index_attribute] > value):
                new_dataset_P.append(row)
            else:
                new_dataset_N.append(row)
        new_dataset_P = np.array(new_dataset_P)
        new_dataset_N = np.array(new_dataset_N)
        Ent_val_P = information_entropy(new_dataset_P)        
        Ent_val_N = information_entropy(new_dataset_N)
        Ent_new = (len(new_dataset_P)*Ent_val_P + len(new_dataset_N)*Ent_val_N) / len(dataset)
        return (Ent_D-Ent_new)

2. **朴素贝叶斯分类模型**
   
   2.1 先验概率计算
        
    根据训练集的数据，按照频率估计概率

        def Prior(train):
        """
        计算类别的先验概率
        :param:训练集
        :return:正例、负例的先验概率
        """
        Posi_num = 0
        neg_num = 0
        for row in train:
            if(row[-1] == 1):
                Posi_num += 1
            else:
                neg_num += 1
        return [Posi_num/len(train), neg_num/len(train)]

    2.2 后验概率的计算

        def conditional_pr(dataset, label, attribute_index, value):
        """
        给定label下，计算观测到所给特征值的条件概率
        :param: dataset：训练集
        :label:所属类别
        :attribute_index:观测值所在特征列的列号
        :value：观测值
        :return：条件概率
        """
        this_dataset = []
        this_dataset_value = []
        for row in dataset:
            if(row[-1] == label):
                this_dataset.append(row)
        for row in this_dataset:
            if(row[attribute_index] == value):
                this_dataset_value.append(row)
        return (len(this_dataset_value) / (len(this_dataset)))


        def Posteir(data, label):
        """
        根据观测值，计算样本属于某一类别的后验概率
        :param:data：样本数据
        :label:假设样本属于的某一类别
        :return:后验概率
        """
        Pr_P,Pr_N = Prior(train)
        # 当前类别下的概率
        Pr_x_c = 1
        # 另外一类的概率
        Pr_x_d = 1
        for i in range(len(data)):
            Pr_x_c = Pr_x_c*conditional_pr(train, label, i, data[i])
        for i in range(len(data)):
            Pr_x_d = Pr_x_d*conditional_pr(train, -label, i, data[i])    
        if(label == 1):
            Pr_x_c = Pr_x_c * Pr_P
            Pr_x_d = Pr_x_d * Pr_N
        else:
            Pr_x_c = Pr_x_c * Pr_N
            Pr_x_d = Pr_x_d * Pr_P
        return (Pr_x_c/(Pr_x_c+Pr_x_d)) 
    2.3 贝叶斯分类器

        def Bayes_classifier(data):
        """
        根据训练得到的贝叶斯分类器，对所给样本所属类别进行预测
        :param:data:样本
        :return: 预测类标，预测类标为1的后验概率，预测类别为-1的后验概率
        """
        result = []
        for label in [-1,1]:
            result.append(Posteir(data,label))
        if(result.index(max(result)) == 0):
            return -1,Posteir(data,1),Posteir(data,-1)
        else:
            return 1,Posteir(data,1),Posteir(data,-1)
3. **模型评估**
   
   计算准确率，$\frac {TP+TN}{TP+FP+TN+FN}$

        def validation(val_data):
        """测试模型在验证集上的效果
        :param val_data: 验证集
        :return: 模型在验证集上的准确率
        """
        # 获取预测类标
        predicted_label = []
        Pr_N = []
        Pr_P = []
        for row in val_data:
            result = Bayes_classifier(row[0:3])
            prediction = result[0]
            Pr_P.append(result[1])
            Pr_N.append(result[2])
            predicted_label.append(prediction)
        # 获取真实类标
        actual_label = [row[-1] for row in val_data]
        # 计算准确率
        accuracy = accuracy_calculation(actual_label, predicted_label)
        return round(accuracy,2),predicted_label,Pr_P,Pr_N

<h3 align='left'><b>四、实验数据和实验方法</b></h3>

1.  数据格式
    
    1.1 数据集

    一共2201行数据，包含三列特征（class age sex）和标签列

    1.2 特征描述
    |Attribute |Type |Range        |
    |----------|:---:|------------:|
    |Class     |real |[-1.87,0.965]|     
    |Age       |real |[-0.228,4.38]|    
    |Sex       |real |[-1.92,0.521]|    
    |Survived  |int  |{-1.0,1.0}   |       

2.  实验方法
   
    重复十次实验，取准确率的平均值，并将结果输出到txt文件

        if __name__ == "__main__":
            file_path = './titanic.dat'

            # 参数设置
            train_percentage = 0.3
            file_save_name = 'Bayes_Titanic_result.txt' # 结果保存路径：默认当前文件路径下
            # 训练模型
            train, val = load_dataset(file_path, train_percentage)
            result, predicted_label,Pr_P,Pr_N = validation(val)
            print(f"accuracy is: {result}%")
            #储存结果: 将每个样例所属每个类别的后验概率值输入到新的文件
            val = pd.DataFrame(val)
            val['predicted_survived'] = predicted_label
            val['Predicted_label_1'] = Pr_P
            val['Predicted_label_-1'] = Pr_N
            val.rename(columns={0:'Class', 1:'Age', 2:'Sex',3:'Survived'}, inplace = True)
            val.to_csv(file_save_name,sep='\t')
<h3 align='left'><b>五、实验结果、结果分析和实验结论</b></h3>

1. **实验结果**
   
   朴素贝叶斯分类器在该数据集上的准确率：75.6%~80.9%，平均值为78.1%，中位数为78.3%

2. **结果分析**
   
   准确率稳定在78%左右，还有较大的提升空间，可以从以下几个方面入手:
   
   2.1 增加数据量

   查阅资料，在另外的大样本数据集上的实验中，模型表现更好，准确率可达80.7%~88.1%

   2.2 引入修正
        
    采用拉普拉斯修正，在估计先验、后验概率值时进行平滑，令N表示训练集D中可能的类别数，$N_i$表示第i个属性可能的取值数，
    
    则先验概率修正为：
    $$
    \hat P(c) = \frac {|D_c|+1}{|D+N|}
    $$
    条件概率修正为：

    $$
    \hat P(x_i | c) = \frac {|D_{c,x_i}|+1}{|D_c|+N_i}
    $$
    
    最后引入拉普拉斯修正后的准确率为76.96%~79.56%，平均值为78.28%，中位数为78.46%，与引入之前无显著提高。分析后发现，训练集中的属性值在每个类中都有实例，没有概率值为0的情况，故采用拉普拉斯修正法后对模型表现提升不大。
   
   2.3 选择更多的特征变量

3. **实验结论**

    朴素贝叶斯分类法在该数据集上的表现尚可，未来可以通过选择更多的特征，增加数据量来提升准确率。同时，如果数据集中所选特征不符合“属性条件独立性”假设也会影响分类器最终的准确率。
    
<h3 align='left'><b>六、 小结</b></h3>
&emsp;本实验利用python语言搭建朴素贝叶斯分类模型，经历数据清洗、连续值离散化、训练集验证集划分、后验概率计算等步骤，最终模型的平均准确率达到78.3%，未来可以从增加样本量、增加特征值等方面来提升准确率。
<br>
&emsp;朴素贝叶斯分类器是一种简便、易于执行、在某些数据集上有着很好的应用效果的分类方法；在应用时，为了尽可能提高准确率，不仅需要选择独立的特征，还需要较大的样本量，同时属性值与类别分布不均匀时还需要引入修正以免概率估值为零。


<h3 align='left'><b>参考文献</b></h3>

1. **机器学习，周志华，清华大学出版社**
2. **Data Mining: Concepts and Techniques, Jiawei Han, Micheline kamber, Jian pei**


