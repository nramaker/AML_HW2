import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import e
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

#arrays to help us plot
accuracies = []
A_magnitudes=[]
B_magnitudes= []
iters=[]

#tuning parameters
epochs = 300
Nb = 50 # batch size
m = .5 #used to calulate step size
n = 1.0 # used to calculate step size
reporting_constant = 5 # how often do we want to track our accuracies and magnitueds
reg_constants = [ .001, .01, .1, 1]
#reg_constants = [ 1]

def train_and_predict():
    print(" ")
    print("### Opening data files")
    data = open_and_join_data()

    print(" ")
    print("### Clearning data")
    data = clean_data(data)

    #scale the data
    print(" ")
    print("### Scaling data")
    data = scale_data(data)
    
    #extract training data 80%
    print(" ")
    print("### Splitting data into training, test, and validation sets")
    train, test, validation = split_dataset(data, test_percent=10, validation_percent=10)

    #extract features and labels
    print(" ")
    print("### Extracting features and labels")
    train_features, train_labels = extract_features_labels(train)
    test_features, test_labels = extract_features_labels(test)
    valid_features, valid_labels = extract_features_labels(validation)

    print("Training Features Shape: {}".format(train_features.shape))
    print("Training Labels Shape: {}".format(train_labels.shape))
    print("Test Features Shape: {}".format(test_features.shape))
    print("Test Labels Shape: {}".format(test_labels.shape))
    print("Validation Features Shape: {}".format(valid_features.shape))
    print("Validation Labels Shape: {}".format(valid_labels.shape))

    # train_features = train_features.iloc[0:5]
    # train_labels = train_labels.iloc[0:5]
    # print(train_features)
    # print("Labels {}".format(train_labels))
    for lam in reg_constants:
        print(" ")
        print("### Fitting model with reg_constant: {}".format(lam))
        accs, a_mags, b_mags, its= fit(train_features, train_labels, valid_features, valid_labels, lam)
        accuracies.append(accs)
        A_magnitudes.append(a_mags)
        B_magnitudes.append(b_mags)
        iters.append(its)

    #check test accuracy using best classifier
    # predict_and_test_accuracy(test_features, test_labels, A, B)


def fit(train_features, train_labels, test_features, test_labels, lam):
    #Initialize A and B to 1s
    feature_count = train_features.shape[1]

    #Initialize A and B
    A = np.empty(shape=(feature_count,1))
    A.fill(1)
    B = 3

    #local measurements
    accs = []
    a_mags = []
    b_mags = []
    its = []

    #loop to train model
    for i in range(1, epochs):
        #make predictions
        predictions = predict(np.transpose(train_features.as_matrix()), A, B)
        #calculate loss
        costs = calculate_cost(predictions, train_labels.as_matrix())
        #update A and B values using gradient decent
        step_size = calc_step_size(epoch=i)
        A, B = calc_updated_coeffs(train_features, train_labels.as_matrix(), costs, Nb, A, B, step_size, lam)

        #track our errors
        if(i%reporting_constant == 0):
            # print("")
            its.append(i)
            a_mags.append(A)
            b_mags.append(B)
            accuracy = predict_and_test_accuracy(test_features,test_labels, A, B)
            accs.append(accuracy)
    return accs, a_mags, b_mags, its

def count_instances_of_value(data, value):
    df = np.where(data==value, -1, 0)
    print("There are {} instances of {} in the data".format(sum(df),value))

def predict(X, A, B):
    # print(" {} * {}".format(np.transpose(A).shape, X.shape))
    pred = np.dot(np.transpose(A), X) + B
    return pred[0]

def predict_and_test_accuracy(features, labels, A, B):
    # print("features.shape {}".format(features.shape))
    # print("A. {}".format(A))
    predictions = predict(np.transpose(features.as_matrix()), A, B)
    correct_predictions = list(map(lambda pred, truth: 1 if pred*truth >=0 else 0, predictions, labels.T ))
    accuracy = sum(correct_predictions)/float(len(predictions))
    # print("Accuracy {}".format(accuracy))
    # print("{} correct out of {} predictions".format(sum(correct_predictions),len(predictions)))
    return accuracy

#plot error
def show_plots():
    # plot accuracies that we recorded for given lambda
    plt.figure(1)

    # plt.subplot(221)
    
    for i in range(0, len(reg_constants)):
        lam = reg_constants[i]       
        its = iters[i]
        accs = accuracies[i]
        
        plt.plot(its, accs, marker='o', label = lam)
        plt.xscale('log')
        plt.yscale('linear')
        
    plt.title('Accuracies for each reg constant')
    plt.legend()
    plt.grid(True)
    
    plt.figure(2)
    feature_titles = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    plot_locs = [221, 222, 223, 224]
    plt.title("Coeffiecent magnitudes")
    for i in range(0, len(reg_constants)):
        plt.subplot(plot_locs[i])
        lam = reg_constants[i]
        plt.title("reg constant: {}".format(lam))
        a_mag = A_magnitudes[i]
        b_mag = B_magnitudes[i]
        for j in range(0, len(feature_titles)):
            its = iters[i]
            title = feature_titles[j]
            coeff = []
            #gather values for each coeffiecent
            for k in range(0, len(its)):
                instance = a_mag[k][j]
                coeff.append(instance)
            plt.plot(its, coeff, label=title)
        #TODO add b
    plt.legend()
    plt.show()

def plot_accuracies(iters, accuracies, lams):
    # plot accuracies that we recorded for given lambda
    plt.figure(1)

    # plt.subplot(221)
    
    for i in range(0, len(lams)):
        lam = lams[i]       
        print("") 
        # print("plotting {} values".format(lam))
        its = iters[i]
        # print("iterations {}".format(its))
        accs = accuracies[i]
        # print("accuracies {}".format(accs))
        
        plt.plot(its, accs, marker='o', label = lam)
        plt.xscale('log')
        plt.yscale('linear')
        
    plt.title('Accuracies for each reg constant')
    plt.legend()
    plt.grid(True)
    
    # plt.figure(2)
    # line1, = plt.plot([3,2,1], marker='x', label='Line 1')
    # line2, = plt.plot([1,2,3], marker='o', label='Line 2')

    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

def calculate_cost(predictions, truths):  #do I need a subbatch here?
    # print("truths {}".format(truths))
    # print("predictions {}".format(predictions))
    # print("truths*predictions {}".format(truths*predictions))
    costs = list(map(lambda y, gamma: max(0, 1 - (y*gamma)), truths, predictions))
    # print("costs {}".format(costs))
    return costs

def calc_updated_coeffs(features, labels, costs, batch_size, A, B, step_size, lam):

    joined = pd.DataFrame.copy(features)
    joined['cost']= costs
    joined['label'] = list(labels)
    # print("joined {}".format(joined))

    #get sub batch from data of batch_size
    selected = joined.sample(n=batch_size)
    # print("smaller batch {}".format(selected))

    selected_costs = selected['cost']
    selected_labels = selected['label']
    selected_X = selected.drop('cost', axis=1).drop('label', axis=1)

    new_A = calc_new_A(A=A.T, X=selected_X, costs=selected_costs, labels=selected_labels, eta=step_size, lam=lam)
    new_B = calc_new_B(B, costs=selected_costs, labels=selected_labels, eta=step_size, lam=lam)
    
    return new_A, new_B

def calc_new_A(A, X, costs, labels, eta, lam):
    N = len(labels)
    lamA = lam*A
    batch_grads = []
    
    x_matrix = X.as_matrix()
    cost_matrix = costs.as_matrix()
    label_matrix = labels.as_matrix()

    for i in range(0, N):
        x = x_matrix[i]
        gamma = cost_matrix[i]
        y = label_matrix[i]

        #gradient decent algorithm
        if y*gamma >= 1:
            batch_grads.append(lamA)
        else:
            batch_grads.append(lamA - np.dot(y,x))
    
    grad_avg = sum(batch_grads)/N
    # print("avg {}".format(grad_avg))
    g_zero = (lam/2)*(A*A)
    # print("g_zero {}".format(g_zero))

    update = (-1.0)*grad_avg - (g_zero)
    # print("update {}".format(update))
    new_A = A + (np.dot(eta,update))

    return new_A.T

def calc_new_B(B, costs, labels, eta, lam):
    N = len(labels)
    batch_grads = []
    
    cost_matrix = costs.as_matrix()
    label_matrix = labels.as_matrix()

    for i in range(0, N):
        gamma = cost_matrix[i]
        y = label_matrix[i]

        if y*gamma >=1:
            batch_grads.append(0)
        else:
            batch_grads.append(-1.0 * y * gamma)

    # print("b batch_grads {}".format(batch_grads))
    grad_avg = sum(batch_grads)/N

    # print("avg {}".format(grad_avg))

    update = -1.0 * grad_avg - (lam*B)
    new_B = B + (np.dot(eta, update))
    return new_B

def calc_step_size(epoch):
    return m/(epoch+n)

def open_and_join_data():
    data = []
    print("Loading samples from file: adult.data")
    df1 = pd.read_csv("adult.data", header=None)
    print("Loaded {} samples from file.".format(len(df1)))
    # return df1
    print("Loading samples from file: adult.test")
    df2 = pd.read_csv("adult.test", header=None, skiprows=1)
    print("Loaded {} samples from file.".format(len(df2)))
    data = pd.concat((df1, df2)).reset_index(drop=True)
    
    print("Full data set is {} lines.".format(len(data)))

    return data

def clean_data(data):

    print(" ")
    print("Data shape before cleaning. {}".format(data.shape))

    #set labels
    headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", 
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "orig_label" ]
    data.columns = headers 
    
    #remove rows with missing values, if any
    age = (data['age'].apply(str)!=' ?') 
    fnlwgt = (data['fnlwgt'].apply(str)!=' ?')
    education = (data['education-num'].apply(str)!=' ?')
    gain = (data['capital-gain'].apply(str)!=' ?')
    loss = (data['capital-loss'].apply(str)!=' ?')
    hours = (data['hours-per-week'].apply(str)!=' ?')
    data = data[age & fnlwgt & education & gain & loss & hours]
    
    workclass = (data["workclass"]!=' ?')
    education = (data["education"]!=' ?')
    marital = (data["marital-status"]!=' ?')
    occupation = (data["occupation"]!=' ?')
    relationship = (data["relationship"]!=' ?')
    sex = (data["sex"]!=' ?')
    race = (data["race"]!=' ?')
    native = (data["native-country"]!=' ?')
    data = data[workclass & education & marital & occupation & relationship & sex & race & native]

    #then remove columns of non continuous data
    data = data.drop("workclass", 1)
    data = data.drop("education", 1)
    data = data.drop("marital-status", 1)
    data = data.drop("occupation", 1)
    data = data.drop("relationship", 1)
    data = data.drop("race", 1)
    data = data.drop("sex", 1)
    data = data.drop("native-country", 1)

    print("Data shape after cleaning. {}".format(data.shape))
    
    #change labels to -1 and 1
    # orig_label = data.orig_label
    data['label'] = np.where(data.orig_label==" <=50K", -1, 0)
    data.loc[data['orig_label'] == " <=50K.", 'label'] = -1
    data.loc[data['orig_label'] == " >50K", 'label'] = 1
    data.loc[data['orig_label'] == " >50K.", 'label'] = 1
    data = data.drop('orig_label', 1)
    return data

def scale_data(data):
    scaler = MinMaxScaler()
    data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']] = scaler.fit_transform(data[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']])
    return data

def split_dataset(data, test_percent, validation_percent=0):
    train_percent = 100 - test_percent - validation_percent
    print("Splitting data in ratio of {}% training, {}% test, and {}% validation.".format(train_percent,test_percent,validation_percent))

    seed = 207
    ratio = train_percent/float(100)
    print("Taking {} of all data for training".format(ratio))
    train = data.sample(frac = ratio, random_state = seed)
    test = data.drop(train.index)
    valid = pd.DataFrame()

    if validation_percent>0:
        #figure out what ratio of the test set we should pull out
        ratio = validation_percent/float(test_percent+validation_percent)
        print("Taking {} of remaining test data for validation".format(ratio))
        valid = test.sample(frac = ratio, random_state = seed)
        new_test = test.drop(valid.index)
        return train, new_test, valid
    else:
        return train, test, pd.DataFrame()

def extract_features_labels(data):
    if len(data) == 0:
        return data, data
    return data.drop("label",1), data["label"]

#main entry
if __name__ == "__main__":
    print(" ##### AML HW2 SVM Classifier  ##### ")
    train_and_predict()
    show_plots()