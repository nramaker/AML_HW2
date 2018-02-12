import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import e
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

#arrays to help us plot
accuracies = []
A_arrays=[]
B_arrays= []
iters=[]

#our best results while training
best_A=[]
best_B=1
best_accuracy = 0.0
best_reg_constant=0

#tuning parameters
epochs = 200
Nb = 300 # batch size
m = 2.5 #used to calulate step size
n = 1 # used to calculate step size
reporting_constant = 1 # how often do we want to track our accuracies and magnitueds
reg_constants = [ 1, .1, .01, .001]
#reg_constants = [ .1]

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

    for lam in reg_constants:
        print(" ")
        print("### Fitting model with reg_constant: {}".format(lam))
        A, B, accs, a_mags, b_mags, its= fit(train_features, train_labels, valid_features, valid_labels, lam)
        accuracies.append(accs)
        A_arrays.append(a_mags)
        B_arrays.append(b_mags)
        iters.append(its)

    #check test accuracy using best classifier
    accuracy = predict_and_test_accuracy(test_features, test_labels, A, B)
    print("Accuracy on Test Set: {}".format(accuracy))
    print("Using A {}".format(A.T))
    print("Using B {}".format(B))
    print("Using lam {}".format(lam))


def fit(train_features, train_labels, test_features, test_labels, lam):
    #Initialize A and B
    A = np.empty(shape=(train_features.shape[1],1))
    A.fill(1)
    B = 1

    #local measurements
    accs = []
    a_mags = []
    b_mags = []
    its = []

    #loop to train model
    for i in range(1, epochs):
    
        step_size = calc_step_size(epoch=i)
        A, B, a_mags= calc_updated_coeffs(train_features, train_labels.as_matrix(), Nb, A, B, step_size, lam)

        #track our errors
        if(i%reporting_constant == 0):
            its.append(i) # record what iteration we're in
            a_mags.append(A)  # record our A vector
            b_mags.append(B) # record our B scalar
            accuracy = predict_and_test_accuracy(test_features,test_labels, A, B) # test the accuracy on the current A, B with the validation data
            accs.append(accuracy) # record our accuracy

    return A, B, accs, a_mags, b_mags, its

def predict(X, A, B):
    pred = np.dot(np.transpose(A), X) + B
    return pred[0]

def predict_and_test_accuracy(features, labels, A, B):
    predictions = predict(np.transpose(features.as_matrix()), A, B)
    correct_predictions = list(map(lambda pred, truth: 1 if pred*truth >=0 else 0, predictions, labels.T ))
    accuracy = sum(correct_predictions)/float(len(predictions))
    return accuracy

def show_plots():
    # plot accuracies that we recorded for given lambda
    plt.figure(1)
    
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
    
    # plt.figure(2)
    # plt.title("Coeffiecent magnitudes")
    # for i in range(0, len(reg_constants)):
    #     this_A = A_arrays[i]
    #     print("this_A[0] {}".format(this_A[0]))
    #     print("this_A[1] {}".format(this_A[1]))
    #     print("this_A[2] {}".format(this_A[2]))
    #     print("this_A[3] {}".format(this_A[3]))
    #     # print("a_mags {}".format(a_mags))
    #     #a_mags = np.sqrt(np.dot(this_A,this_A))
    #     plt.plot(this_A.index,this_A, label=reg_constants[i])
    # plt.legend()

    # plot of coeffients (not required)
    # plt.figure(3)
    # feature_titles = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # plot_locs = [321, 322, 323, 324]
    # plt.title("Coeffiecent magnitudes")
    
    # for i in range(0, len(reg_constants)):
    #     plt.subplot(plot_locs[i])
    #     plt.title("reg constant: {}".format(reg_constants[i]))
    #     for j in range(0, len(feature_titles)):
    #         coeff = []
    #         #gather values for each coeffiecent
    #         for k in range(0, len(iters[i])):
    #             coeff.append(A_arrays[i][k][j])
    #         plt.xscale('log')
    #         plt.plot(iters[i], coeff, label=feature_titles[j])
    #     plt.plot(iters[i], B_arrays[i], label="bias")
    # plt.legend(loc='lower center', bbox_to_anchor=(0, -1.0),
    #       ncol=3, fancybox=True, shadow=True)
    plt.show()

def calc_updated_coeffs(features, labels, batch_size, A, B, step_size, lam):

    #join the features and labels so we get matching samples
    joined = pd.DataFrame.copy(features)
    joined['label'] = list(labels)

    #get sub batch from data of batch_size
    selected = joined.sample(n=batch_size)
    
    # separate them again
    selected_labels = selected['label']
    selected_X = selected.drop('label', axis=1)

    # perform gradient descent
    new_A, new_B, a_magnitudes = calc_new_A(X=selected_X.as_matrix(), y=selected_labels.as_matrix(), a=A.T, b=B, eta=step_size, lam=lam)
    # new_B = calc_new_B(X=selected_X.as_matrix(), y=selected_labels.as_matrix(), a=A.T, b=B, eta=step_size)
    return new_A, new_B, a_magnitudes

def calc_new_A(X, y, a, b, eta, lam):
    #for every instance in our batch
    a_magnitudes = []
    accuracies = []
    for i in range(len(y)):
        #check to see if the magnitude of our cost is 
        if y[i] * (np.dot(a, X[i]) + b) >= 1:
            a -= eta * lam * a
            b -= eta * 0 
        else:
            a -= eta * (lam * a - y[i] * X[i])
            b -= eta * -y[i]
        if i % reporting_constant == 0 :
            magnitude = list(map(lambda x: np.sqrt(np.dot(x,x)),a))
            a_magnitudes.append(magnitude)
            
            
    return (a.T,b, a_magnitudes)

# def calc_new_B(X, y, a, b, eta):
#     #for every instance in our batch
#     for i in range(len(y)):
#         #check to see if the magnitude of our cost is 
#         if y[i] * (np.dot(a, X[i]) + b) >= 1:
#             b -= eta * 0 
#         else:
#             b -= eta * -y[i]
#     return b

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