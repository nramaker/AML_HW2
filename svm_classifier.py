import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#variables to track error
errors = []
reg_vectors=[]

#tuning parameters
epochs = 300
Nb = 50 # batch size
reg_constants = [1]
#reg_constants = [1**-3, 1**-2, 1**-1, 1]

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

    train_features = train_features.iloc[:2]
    print(train_features)
    print("Labels {}".format(train_labels.iloc[:2]))
    for lam in reg_constants:
        print(" ")
        print("### Fitting model with reg_constant: {}".format(lam))
        A, B, errors, accuracies = fit(train_features, train_labels, valid_features, valid_labels, lam)
        # print(A, B, errors)

    #check test accuracy using best classifier
    predict_and_test_accuracy(test_features, test_labels, A, B)
    return [], []


def fit(train_features, train_labels, test_features, test_labels, lam):
    #Initialize A and B to 1s
    feature_count = train_features.shape[1]
    row_count = len(train_features)

    #Initialize A and B
    A = np.empty(shape=(feature_count,1))
    A.fill(1)
    B = np.empty(shape=(1,row_count))
    B.fill(1)

    #loop to train model
    for i in range(1, 10):
        #make predictions
        predictions = predict(np.transpose(train_features.as_matrix()), A, B, lam)
        print("Predictions {}".format(predictions))
        #calculate loss
        costs = calculate_cost(predictions, train_labels.as_matrix())
        print("Costs {}".format(costs))
        #update A and B values using gradient decent
        step_size = calc_step_size(epoch=i)
        print("Step size {}".format(step_size))
        A, B = calc_gradient(costs, Nb, A, B, step_size)

        #track our errors
        if(i%30 == 0):
            print("Epoch {}".format(i))
            #TODO compute accurracies
            #TODO record coefficient vectors
    return [0.0], [0.1], [0.0], []


def predict(X, A, B, lam):
    #return vector of 

    # print(" {} * {} + {}".format(np.transpose(A).shape, X.shape, B.shape))
    pred = np.dot(np.transpose(A), X) + B

    return pred[0]

def predict_and_test_accuracy(test_features, test_labels, A, B):
    pass

#plot error
def show_plots(errors, reg_vectors):
    print(" ")
    print("### Showing plots")
    #sum cost of each example, then average them by number of examples

def calculate_cost(predictions, truths):  #do I need a subbatch here?
    #calculate hinge loss
    costs = list(map(lambda y, gamma: max(0, 1 - (y*gamma)), truths, predictions))
    return costs

def calc_gradient(data, batch_size, A, B, step_size):
    #get sub batch from data of batch_size
    return A, B

def calc_step_size(epoch):
    m = 1.0
    n = 1.0
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

    seed = 507
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
    errors, reg_vectors = train_and_predict()
    show_plots(errors, reg_vectors)