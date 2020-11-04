import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def kNN(data, labels):

    testSize =  input("Please choose test size (e.g 0.25): ")

    #split data set into 75% training set and 25% testing set
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=float(testSize), random_state=0)

    distanceMetrics = ['euclidean', 'manhattan','chebyshev'] #we will later loop through these distance metrics to try knn on them

    #These patches will be used for the legend in the plots
    red_patch = mpatches.Patch(color='yellow', label='CLass 1: IH')
    blue_patch = mpatches.Patch(color='magenta', label='Class 2: EH')
    green_patch = mpatches.Patch(color='blue', label='Class 3: AA')
    star_patch1 = mpatches.Patch(color='black', label='Predicted class 1')
    star_patch2 = mpatches.Patch(color='green', label='Predicted class 2')
    star_patch3 = mpatches.Patch(color='red', label='Predicted class 3')
    h = .02  # step size in the mesh

    #asking the user how many time s/he wants to test the classification
    numberOfKs = input("For how many different K values would you like to test K-NN classification?")

    for i in range(int(numberOfKs)): #loop through inputted number of tests

        neighbours = input("Please enter K value: ") #ask user to enter the K-value
        neighbours = int(neighbours)
        OriginaltestValues = X_test.copy() #copy the values to another array before normalizing them

        #normalizing features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for metric in distanceMetrics: #for each distance metric, execute the K-nn algorithm using the corresponding metric
            print("----------------- Using ", metric,"distance metric-----------------")
            classifier = KNeighborsClassifier(n_neighbors=neighbours, metric=metric) #initialize class with the K-value and distance metric used
            classifier.fit(X_train, Y_train)

            prediction = classifier.predict(X_test) #get the class prediction on each of the test data

            for i in range(len(OriginaltestValues)):
                print("Formant test values: ", OriginaltestValues[i], "\tClassified to class: ", prediction[i]) #output results one by one

            print("\n\n\nConfusion Matrix:")
            print(confusion_matrix(Y_test, prediction)) #compute and print the confusion matrix
            print(classification_report(Y_test, prediction))

            #--------plotting the results---------
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1
            np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))


            #get each training set column and store them in their rispective formant
            formant1 = X_train[:, 0]
            formant2 =X_train[:, 1]
            formant3 = X_train[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') #using 3D plot to plot all 3 formants

            counter = 0
            for classes in Y_train: #loop through the class numbers
                if classes == 1: #if class 'IH' plot the below
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='yellow', marker='o', s=8)
                    counter += 1 #increment for next iteration
                elif classes == 2:#if class 'EH' plot the below
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='magenta', marker='o', s=8)
                    counter += 1 #increment for next iteration
                elif classes ==3:#if class 'AA' plot the below
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='b', marker='o', s=8)
                    counter+=1 #increment for next iteration


            #------------plotting the test data-------
            formant1 = X_test[:, 0]
            formant2 = X_test[:, 1]
            formant3 = X_test[:, 2]

            counter = 0
            #use the same routine implemented to plot the training data.  Instead loop for every predicted class,
            #and give different colours, and set the markers equal to a star to distingusih from tes points to training points
            for classes in prediction:
                if classes == 1:
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='black', marker='*', s=12)
                    counter += 1
                elif classes == 2:
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='green', marker='*', s=12)
                    counter += 1
                elif classes == 3:
                    ax.scatter(formant1[counter], formant2[counter], formant3[counter], c='red', marker='*', s=12)
                    counter += 1

            #set labels on the axis
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_zlabel('f3')

            #plot the legend, title and show the whole plot
            plt.legend(handles=[red_patch, blue_patch, green_patch, star_patch1, star_patch2, star_patch3])
            plt.title("K-NN (K = %i, distance metric = '%s')"% (neighbours, metric), loc = 'left')
            plt.show()



def main():

    print("Wait till data set is opened...\n\n")
    dataset = pd.read_csv("Gender Sorted Extraction.csv") #open the .csv file containing all the data

    #Give user the possibility to choose on what gender the KNN is to be executed
    print("Press 1 if you want to apply K-NN on Both genders together values")
    print("Press 2 if you want to apply K-NN on Female data only")
    print("Press 3 if you want to apply K-NN on Male data only")
    print("Press 4 if you want to apply K-NN on all the above")
    choice = input("Enter choice: ")

    if(choice == '1'): #if both genders
        print("BOTH GENDERS TOGETHER ")
        # get all formants and class numbers from data set for both genders
        BothGenders_x = dataset.iloc[:,5:8].values  # get all columns that contian the formants, and all their rispective rows
        BothGenders_y = dataset.iloc[:, 4].values  # get column 4 which contains the class number
        kNN(BothGenders_x, BothGenders_y) #apply knn on the whole dataset
    elif choice == '2':
        print("FEMALES ONLY")
        # get the formant values and class numbers for just Females
        Females_x = dataset.iloc[:75, 5:8].values#In the .csv file, the Females are in the first 75 rows. Hence get the first 75 rows of formant values
        Females_y = dataset.iloc[:75, 4].values#again get the first 75 rows for column of labels
        kNN(Females_x, Females_y)#apply knn on the females part of the dataset
    elif choice == '3':
        print("MALES ONLY")
        # get the formant values and class numbers for just Males
        Males_x = dataset.iloc[75:, 5:8].values #In the .csv file, the Males are in the last 75 rows. Hence get the last 75 rows of the formant values columns
        Males_y = dataset.iloc[75:, 4].values#again get the first 75 rows for column of labels
        kNN(Males_x, Males_y) #apply knnon the males dataset
    elif choice =='4':
        print("K-NN WILL BE APPLIED ON BOTH GENDERS TOGETHER, THEN FEMALES ONLY AND LASTLY MALES ONLY.")
        print("1. BOTH GENDERS TOGETHER ")
        BothGenders_x = dataset.iloc[:,5:8].values  # get all columns that contian the formants, and all their rispective rows
        BothGenders_y = dataset.iloc[:, 4].values  # get column 4 which contains the class number, and all the rows of that column
        kNN(BothGenders_x, BothGenders_y) #apply knn on the whole dataset

        print("2. FEMALES ONLY")
        # get the formant values and class numbers for just Females
        Females_x = dataset.iloc[:75, 5:8].values #In the .csv file, the Females are in the first 75 rows. Hence get the first 75 rows of formant values
        Females_y = dataset.iloc[:75, 4].values #again get the first 75 rows for column of labels
        kNN(Females_x, Females_y) #apply knn on the females dataset

        print("3. MALES ONLY")
        # get the formant values and class numbers for just Males
        Males_x = dataset.iloc[75:, 5:8].values#In the .csv file, the Males are in the last 75 rows. Hence get the last 75 rows of the formant values columns
        Males_y = dataset.iloc[75:, 4].values#again get the last 75 rows this time for the class number column
        kNN(Males_x, Males_y)#apply knnon the males dataset
    else:
        print("Incorrect input")

    input("Press enter to exit")

main() #call main function
