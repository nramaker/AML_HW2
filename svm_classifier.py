import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#variables to track error
errors = []
reg_vectors=[]

#tuning parameters
epochs = 300
Nb = 5 # batch size
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

    # train_features = train_features.iloc[1:2]
    # train_labels = train_labels.iloc[1:2]
    # print(train_features)
    # print("Labels {}".format(train_labels))
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
    B = 1

    #loop to train model
    for i in range(1, epochs):
        #make predictions
        predictions = predict(np.transpose(train_features.as_matrix()), A, B)
        # print("Predictions {}".format(predictions))
        #calculate loss
        costs = calculate_cost(predictions, train_labels.as_matrix())
        # print("Costs {}".format(costs))
        #update A and B values using gradient decent
        step_size = calc_step_size(epoch=i)
        # print("Step size {}".format(step_size))
        #print("Previous X.shape {}".format(train_features.shape))
        A, B = calc_updated_coeffs(train_features, train_labels.as_matrix(), costs, Nb, A, B, step_size, lam)
        #print("New X.shape {}".format(train_features.shape))

        #track our errors
        if(i%30 == 0):
            print("############## Epoch {} ".format(i))
            print("A = {}".format( A))
            print("B = {}".format(B))
            #TODO compute accurracies
            #TODO record coefficient vectors
    return [0.0], [0.1], [0.0], []


def predict(X, A, B):
    # print(" {} * {}".format(np.transpose(A).shape, X.shape))
    pred = np.dot(np.transpose(A), X) + B

    return pred[0]

def predict_and_test_accuracy(test_features, test_labels, A, B):
    pass

def calc_accuracy(predictions, truths):
    N = len(truths)
#plot error
def show_plots(errors, reg_vectors):
    print(" ")
    print("### Showing plots")
    #sum cost of each example, then average them by number of examples

def calculate_cost(predictions, truths):  #do I need a subbatch here?
    #calculate hinge loss
    #TODO add additional regularization argument
    costs = list(map(lambda y, gamma: max(0, 1 - (y*gamma)), truths, predictions))
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

    new_A = calc_new_A(A=A, X=selected_X, costs=selected_costs, labels=selected_labels, eta=step_size, lam=lam)
    new_B = calc_new_B(B, costs=selected_costs, labels=selected_labels, eta=step_size)
    
    # joined.drop('cost', axis=1).drop('label', axis=1)
    return new_A, new_B

def calc_new_A(A, X, costs, labels, eta, lam):
    N = len(labels)

    # lamA = lam*A
    # print("lamA {}".format(lamA.T))
    # gradients = []
    # print(labels)
    # for i in range(0, len(labels.as_matrix())):
    #     label = labels.as_matrix()[i]
    #     print("label {}".format(label))
    #     cost = costs.as_matrix()[i]
    #     print("cost {}".format(cost))
    #     this_x = X.as_matrix()[i]
    #     print("this_x {}".format(this_x))
    #     cond = labels.as_matrix()[i] * costs.as_matrix()[i]  # this is the value that we test on
    #     print("condition at {} is {}".format(i, cond))

    #     if cond >= 1:
    #         print("Condition met")
    #         gradients.append(lamA)
    #     else:
    #         print("Else met")
    #         x = label*this_x
    #         print("x {}".format(x))
    #         gradients.append(lamA.T - x)
    # print("A {}".format(A))
    # print("labels {}".format(labels))
    # print("X {}".format(X.as_matrix()))
    # print("labels.shape {}".format(labels.shape))
    # print("lam {}".format(lam))
    #print("lam * A - y {}".format(lam*A -labels ))
    # print("y * cost {}".format(labels*costs))

    gradients = list(map(lambda y, cost, X: lam*A.T if y*cost>=1 else (lam*A.T - y*(X)), labels, costs, X.as_matrix()))
    
    #avg = list(map(lambda grads: sum(grads), gradients.iloc[0]))
    # avg = np.array(avg) - N
    avg = sum(gradients)/N

    # print("gradients {}".format(gradients))
    # print("average {}".format(avg))
    new_value = A - (eta*avg).T
    #print("new_value_A{}".format(new_value))
    return new_value

def calc_new_B(B, costs, labels, eta):
    gradients = list(map(lambda y, cost: 0 if y*cost>=1 else y*(-1.0), labels, costs))
    avg = sum(gradients)/len(gradients)
    new_value = B - eta*avg
    return new_value

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