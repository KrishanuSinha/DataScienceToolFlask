from flask import Flask,render_template,url_for,request,Blueprint
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import seaborn as sns

# core/views.py
#from flask import render_template,request,Blueprint


#core = Blueprint('core',__name__)

app = Flask(__name__)


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

@app.route('/')
def home():
    return render_template('home.html')	
	
# Displaying Number of Rows and Columns in the Dataset:
@app.route('/preprocess')
def preprocess():  
    return render_template('about.html')

# Displaying Number of Rows and Columns in the Dataset:
@app.route('/classify')
def classify():  
    return render_template('services.html')	
	
# Displaying Visualization
@app.route('/visualize')
def visualize():  
    return render_template('visualization.html')		
	
@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv(r'C:\Users\YO5Q\Mymodule\YoutubeSpamMergedData.csv',encoding='latin-1')
    df = df.dropna(axis=0)
    df_data = df[["CONTENT","CLASS"]]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

@app.route("/display",methods=['POST'])
def display():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        num = request.form['Display']
        #num = [num]
        num = int(num)+ 1
        #df = pd.read_csv("CUST_PROFILE_CSV.csv")
        df = df.head(num).to_html()
    return render_template('display.html', data=df)

@app.route("/scatter",methods=['POST'])
def scatter():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    df['Date'] = pd.to_datetime(df['BIRTH_DT'])
    Current_Year = pd.to_datetime('now').year
    df['Birth_Year'] = pd.DataFrame(df['Date'].dt.year)
    df['Birth_Year']= np.where((df['Birth_Year']>2018), df['Birth_Year']-100, df['Birth_Year'])
    df['Age']= Current_Year - df['Birth_Year']
    df = df.dropna(axis=1)
    sns.regplot(y='BI_PER_ACCIDENT', x='PD_PER_ACCIDENT', data=df, fit_reg = True, color = 'blue')
    #plt.show()
    if request.method == 'POST':
     #   X = request.form['X_Values']
     #   Y = request.form['Y_Values']
      #  sns.regplot(y=Y, x=X, data=df, fit_reg = True, color = 'blue')
         plt.show()


# Finding Outliers in Continuous Numerical Features	
@app.route("/outlier",methods=['POST'])
def outlier():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        col = request.form['Outliers']
        dist = request.form['StandardDev']
        dist = int(dist)
        outliers = df[df[col] > df[col].mean() + dist * df[col].std()]
        df.to_csv('DeletedColumn_Data.csv')
        df2 = outliers.to_html()
    return render_template('display.html', data=df2)

# Applying ML to find top feature_importances
@app.route("/ApplyML",methods=['POST'])
def ApplyML():
    df = pd.read_csv('Categorized.csv', low_memory=False,index_col=False)
    df = df.iloc[:,3:]
    if request.method == 'POST':
        label = request.form['ApplyML']
        X,y = df.drop(label,axis=1),df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y)
        #select = SelectPercentile(percentile = 50)
        #select.fit(X_train, y_train)
        #X_Train_selected = select.transform(X_train)
        clf = ExtraTreesClassifier()
        clf = clf.fit(X_train, y_train)
        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        pred = clf.predict(X_test)
        f1_score(y_test,pred)
        # View a list of the features and their importance scores
        FeatureImportance = list(zip(X_train, clf.feature_importances_))
        FeatureImportance = pd.DataFrame(FeatureImportance, columns=['Features','Importance'])
        df2 = FeatureImportance.sort_values(by = ['Importance'],ascending=False).to_html()
    return render_template('display.html', data=df2)

# Finding F1 Score of Classification Model
@app.route("/Accuracy",methods=['POST'])
def Accuracy():
    df = pd.read_csv('Categorized.csv', low_memory=False,index_col=False)
    df = df.iloc[:,3:]
    if request.method == 'POST':
        label = request.form['Accuracy']
        X,y = df.drop(label,axis=1),df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y)
        #select = SelectPercentile(percentile = 50)
        #select.fit(X_train, y_train)
        #X_Train_selected = select.transform(X_train)
        clf = ExtraTreesClassifier()
        clf = clf.fit(X_train, y_train)
        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        pred = clf.predict(X_test)
        f1 = f1_score(y_test,pred)
        acc = pd.DataFrame(columns=["F1-Score"], data=[[f1]])
        df2 = acc.set_index(['F1-Score'])
        df2 = df2.to_html()
    return render_template('display.html', data=df2)


# Finding Confusion Matrix of Classification Model
@app.route("/ConfusionMatrix",methods=['POST'])
def ConfusionMatrix():
    df = pd.read_csv('Categorized.csv', low_memory=False,index_col=False)
    df = df.iloc[:,3:]
    if request.method == 'POST':
        label = request.form['ConfusionMatrix']
        X,y = df.drop(label,axis=1),df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y)
        #select = SelectPercentile(percentile = 50)
        #select.fit(X_train, y_train)
        #X_Train_selected = select.transform(X_train)
        clf = ExtraTreesClassifier()
        clf = clf.fit(X_train, y_train)
        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        pred = clf.predict(X_test)
        df_confusion = pd.crosstab(y_test,pred, rownames=['Actual'], colnames=['Predicted'])
        #acc = pd.DataFrame(columns=["F1-Score"], data=[[f1]])
        #df2 = acc.set_index(['F1-Score'])
        df2 = df_confusion.to_html()
    return render_template('display.html', data=df2)


@app.route("/GridSearch",methods=['POST'])
def GridSearch():
    df = pd.read_csv('Categorized.csv', low_memory=False,index_col=False)
    df = df.iloc[:,3:]
    if request.method == 'POST':
        label = request.form['gridsearch']
        X,y = df.drop(label,axis=1),df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        
        scl = ('scl', StandardScaler())
        rdm = ('rdm', RFECV(cv=k_fold, estimator=DecisionTreeClassifier(random_state=42), scoring='accuracy'))
        clf = ('clf', DecisionTreeClassifier(random_state=42))

        param_grid = [{'rdm__step': [0.02],
               'clf__criterion': ['gini', 'entropy']}]


        pipe = Pipeline([scl, rdm, clf])


        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=k_fold, refit='accuracy')



        gs.fit(X,y)
        best_pipe = gs.best_estimator_

    #    print('%.5f' % gs.best_score_)
    #    print('mask1: ', best_pipe.named_steps['rdm'].support_)


# fit the best estimator on train set again to be able to get a confusion matrix
        scores = []
        for X_train, y_train in k_fold.split(X, y):
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]

            best_pipe.fit(X_train, y_train)
            y_pred = best_pipe.predict(X_test)
            score = f1_score(y_test,y_pred)
            scores.append(score)
    
    # I didn't regard the fact that GridSearchCV is refitting the pipeline on the WHOLE dataset...
    # so, if this is not called, mask1 and mask2 will be different
    #best_pipe.fit(X, y)

    
    #    print('%.5f' % np.average(scores))
        acc = pd.DataFrame(columns=["Average Score"], data=[[np.average(scores)]])
        df2 = acc.set_index(['Average Score'])
        df2 = df2.to_html()
        #print('mask2: ', best_pipe.named_steps['rdm'].support_)
    return render_template('display.html', data=df2 )

# Deleting Column by Name
@app.route("/delcol",methods=['POST'])
def delcol():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        col = request.form['delcol']
        df = df.drop(col, axis=1)
        df.to_csv('DeletedColumn_Data.csv')
        df2 = df.head(100).to_html()
    return render_template('display.html', data=df2 )

# Deleting Rows by number
@app.route("/delrow",methods=['POST'])
def delrow():
    df = pd.read_csv("DeletedColumn_Data.csv",low_memory=False)
    
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':
        row = request.form['delrow']
        row = int(row) 
        df = df.drop(row, axis=0)
        df.to_csv('DeletedRows_Data.csv')
        df2 = df.head(100).to_html()
    return render_template('display.html', data=df2 )

# Cleaning up Missing Data
@app.route("/MissingData",methods=['POST'])
def MissingData():
    df = pd.read_csv('CUST_PROFILE_CSV.csv', low_memory=False)
    if request.method == 'POST':
        nan_percent = request.form['MissingData']
        nan_percent = float(nan_percent)/100
        t = []
        threshold = len(df.index) * nan_percent
        [t.append(c) for c in df.columns if sum(df[c].isnull()) >= threshold]
        df = df.drop(t, axis=1)
        df.to_csv('CleanedData.csv')
        df2 = df.head(100).to_html()
    return render_template('display.html', data=df2 )

# Impute the Missing Values
@app.route("/Imputer",methods=['POST'])
def Imputer():
    df = pd.read_csv('CleanedData.csv', low_memory=False)
    if request.method == 'POST':
        xt = DataFrameImputer().fit_transform(df)
        xt.to_csv('ImputedData.csv')
        df2 = xt.head(100).to_html()
    return render_template('display.html', data=df2)

# Convert the categorical data into Numericals
@app.route("/Categorized",methods=['POST'])
def Categorized():
    df = pd.read_csv('ImputedData.csv', low_memory=False)
    if request.method == 'POST':
        for col_name in df.columns:
            if(df[col_name].dtype == 'object'):
                df[col_name]= df[col_name].astype('category')
                df[col_name] = pd.get_dummies(df[col_name])
        #df = pd.get_dummies(df).reindex(columns=tradf.columns,fill_value=0)
        df = pd.DataFrame(df.iloc[:,3:])
        df.to_csv('Categorized.csv')
        df2 = df.head(100).to_html()
    return render_template('display.html', data=df2)


# Displaying Number of Rows and Columns in the Dataset:
@app.route("/shape",methods=['POST'])
def shape():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    Rows = len(df.index)
    Col = len(df.columns)
    df2 = pd.DataFrame(columns=["Rows", "Columns"], data=[[Rows,Col]])
    df2 = df2.set_index(['Rows','Columns'])
    if request.method == 'POST':
        #df = pd.read_csv("CUST_PROFILE_CSV.csv")
        df2 = df2.head().to_html()
    return render_template('display.html', data=df2)

# Displaying a describe    
@app.route("/describe",methods=['POST'])
def describe():
    df = pd.read_csv("CUST_PROFILE_CSV.csv")
    if request.method == 'POST':
        df = df.describe().to_html()
    return render_template('display.html', data=df)

# Percentage of NAN in the Data
@app.route("/percNaN",methods=['POST'])
def percNaN():
    df = pd.read_csv("CUST_PROFILE_CSV.csv")
    x =  df.isnull().mean()*100
    df = pd.DataFrame(x,columns=['Percentage Of NAN'])
    df = df.sort_values(by=['Percentage Of NAN'], ascending=False)
    if request.method == 'POST':
        df = df.to_html()
    return render_template('display.html', data=df)



# Displaying Column Names in the Dataset
@app.route("/ColumnNames",methods=['POST'])
def ColumnNames():
    df = pd.read_csv("CUST_PROFILE_CSV.csv",low_memory=False)
    df = pd.DataFrame(df.columns,columns=['Column Names'])
    if request.method == 'POST':
        df = df.to_html()
    return render_template('display.html', data=df)



"""@app.route("/ApplyML",methods=['POST'])
def ApplyML():
    df = pd.read_csv('Categorized.csv', low_memory=False,index_col=False)
    df = df.iloc[:,3:]
    label = 'GNDR_CD'
    X,y = df.drop(label,axis=1),df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y)
    #select = SelectPercentile(percentile = 50)
    #select.fit(X_train, y_train)
    #X_Train_selected = select.transform(X_train)
    clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
    pred = clf.predict(X_test)
    f1_score(y_test,pred)
    # View a list of the features and their importance scores
    FeatureImportance = list(zip(X_train, clf.feature_importances_))
    FeatureImportance = pd.DataFrame(FeatureImportance, columns=['Features','Importance'])
    df = FeatureImportance.sort_values(by = ['Importance'],ascending=False)
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        #vect = cv.transform(data).toarray()
        #my_prediction = clf.predict(vect)
        df = df.to_html()
    return render_template('display.html', data=df)"""

if __name__ == '__main__':
    app.run(debug=True)
