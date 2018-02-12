import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import e
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

#arrays to help us plot
accuracies = []
A_arrays=[]
iters=[]

#tuning parameters
epochs = 50
Nb = 300 # batch size
m = 1.5 #used to calulate step size
n = 10 # used to calculate step size
reporting_constant = 30 # how often do we want to track our accuracies and magnitueds
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

    #our best results while training
    best_A=[]
    best_B=1
    best_accuracy = 0.0
    best_reg_constant=0
    
    for lam in reg_constants:
        print(" ")
        print("### Fitting model with reg_constant: {}".format(lam))
        A, B, best_acc, best_reg_constant, accs, a_mags, its= fit(train_features, train_labels, valid_features, valid_labels, lam)
        accuracies.append(accs)
        A_arrays.append(a_mags)
        iters.append(its)
        
        if best_acc > best_accuracy:
            best_accuracy=best_acc
            best_A = A
            best_B = B
            best_reg_constant=lam
        
        accuracy = predict_and_test_accuracy(test_features, test_labels, A, B)
        print("")
        print("Accuracy on Test Set: {}".format(accuracy))
        print("Using A {}".format(A.T))
        print("Using B {}".format(B))
        print("Using lam {}".format(lam))

    # #check test accuracy using best classifier
    accuracy = predict_and_test_accuracy(test_features, test_labels, best_A, best_B)
    print("")
    print("### Final Results")
    print("Best Accuracy on Validation Set: {}".format(best_accuracy))
    print("Accuracy on Test Set Using Same Model: {}".format(accuracy))
    print("Using A {}".format(best_A.T))
    print("Using B {}".format(best_B))
    print("Using lam {}".format(best_reg_constant))


def fit(train_features, train_labels, test_features, test_labels, lam):
    #Initialize A and B
    A = np.empty(shape=(train_features.shape[1],1))
    A.fill(1)
    B = 1

    #local measurements
    accs = []
    a_mags = []
    its = []
    
    best_accuracy=0.0
    best_A = []
    best_B = 0

    #loop to train model
    for i in range(1, epochs):
    
        step_size = calc_step_size(epoch=i)

        selected_X, selected_labels = select_sub_batch(train_features, train_labels.as_matrix(), batch_size=Nb)
        X=selected_X.as_matrix()
        y=selected_labels.as_matrix()
        a = A.T
        
        for j in range(len(y)):
            #check to see if the magnitude of our cost is 
            if y[j] * (np.dot(a, X[j]) + B) >= 1:
                a -= step_size * lam * a
                B -= step_size * 0 
            else:
                a -= step_size * (lam * a - y[j] * X[j])
                B -= step_size * -y[j]
            # at regular intervals, tests and record our process
            if j % reporting_constant == 0 :
                its.append(j+i*Nb)
                magnitude = list(map(lambda x: np.sqrt(np.dot(x,x)),a))
                a_mags.append(magnitude)
                accuracy = predict_and_test_accuracy(test_features,test_labels, a.T, B) # test the accuracy on the current A, B with the validation data
                accs.append(accuracy) # record our accuracy
                
                if accuracy > best_accuracy:
                    best_accuracy=accuracy
                    best_A = a.T
                    best_B = B
                    best_reg_constant=lam

    return best_A, best_B, best_accuracy, best_reg_constant, accs, a_mags, its

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
        plt.plot(iters[i], accuracies[i], label = reg_constants[i])
        plt.xscale('log')
        plt.yscale('linear')
        
    plt.title('Accuracies for each reg constant')
    plt.legend()
    plt.grid(True)
    
    #plot the magnitues of the coefficients
    plt.figure(2)
    plt.title("Coeffiecent magnitudes")
    for i in range(0, len(reg_constants)):
        plt.plot(iters[i],A_arrays[i], label=reg_constants[i])
    plt.legend()
    plt.show()

def select_sub_batch(features, labels, batch_size):
    #join the features and labels so we get matching samples
    joined = pd.DataFrame.copy(features)
    joined['label'] = list(labels)

    #get sub batch from data of batch_size
    selected = joined.sample(n=batch_size)
    
    # separate them again
    selected_labels = selected['label']
    selected_X = selected.drop('label', axis=1)
    return selected_X, selected_labels            

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