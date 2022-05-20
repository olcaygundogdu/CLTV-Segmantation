# IMPORT
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

# SET PROCEDURES
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

"""
1. Business Problem
An Auto Insurance company in the USA is facing issues in retaining its customers and wants to advertise promotional offers for its loyal customers. They are considering Customer Lifetime Value CLV as a parameter for this purpose. Customer Lifetime Value represents a customer’s value to a company over a period of time. It’s a competitive market for insurance companies, and the insurance premium isn’t the only determining factor in a customer’s decisions. CLV is a customer-centric metric, and a powerful base to build upon to retain valuable customers, increase revenue from less valuable customers, and improve the customer experience overall. Using CLV effectively can improve customer acquisition and customer retention, prevent churn, help the company to plan its marketing budget, measure the performance of their ads in more detail, and much more

2. Project Overview
The objective of the problem is to accurately predict the Customer Lifetime Value(CLV) of the customer for an Auto Insurance Company
Performed EDA to understand the relation of target variable CLV with the other features.
Statistical Analysis techniques like OLS for numerical and Mann–Whitney U and also Kruskal Wallis test for the categorical variables were performed to find the significance of the features with respect to the target.
Supervised Regression Models like Linear Regression, Ridge Regression, Lasso Regression, DecisionTree Regression, Random Forest Regression and Adaboost Regression.
Using GridSearchCV with Random Forest Regression gave the best RMSE and R^2 score values
3.Dataset Description
The dataset represents Customer lifetime value of an Auto Insurance Company in the United States, it includes over 24 features and 9134 records to analyze the lifetime value of Customer.
"""

# FUNCTİONS
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()})

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable,q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.1, q3=0.9):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# READİNG TRAİNİNG DATA AND PRODUCTİON DATA

df_train = pd.read_csv("Proje/squark_automotive_CLV_training_data.csv")
df_train.dropna(inplace=True)

prediction = pd.read_csv("Proje/squark_automotive_CLV_production_data.csv")

df = pd.concat([df_train, prediction], ignore_index=True)

df = df.drop(["Effective To Date", "Customer"], axis=1)

# Exploratory Data Analysis

check_df(df_train)

cat_cols, num_cols, cat_but_car = grab_col_names(df_train)

# Categorical Variable Analysis
for col in cat_cols:
    cat_summary(df_train, col, plot=True)

# Numerical Variable Analysis
for col in num_cols:
    num_summary(df_train, col, plot=True)

# Target Variable and Categorical Variable Analysis
for col in cat_cols:
    target_summary_with_cat(df_train, "Customer Lifetime Value", col, plot=True)

# Correlation

correlation_matrix(df, num_cols)

# FONKSİYONALŞTIRMA

def cltv_data_preb(df):

    print("Data Preprocessing...")

    df.columns = [col.upper() for col in df.columns]

    # Müşteri yakın zamanda (5 ay öncesine kadar) claimde bulunmuş mu?
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 5, "NEW_Recent Claim"] = "Yes"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 5, "NEW_Recent Claim"] = "No"
    # Ortalama poliçe fiyatı
    df["NEW_Average Policy Price"] = df["MONTHLY PREMIUM AUTO"] / df["NUMBER OF POLICIES"]
    # Açık şikayet var mı?
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] == 0, "OPEN COMPLAINTS"] = "No"
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] > 0, "OPEN COMPLAINTS"] = "Yes"
    # Son şikayet ne zaman oldu?
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "NEW_LAST_CLAİM"] = "Recently"
    df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 20), "NEW_LAST_CLAİM"] = "Medium"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 20, "NEW_LAST_CLAİM"] = "Long"
    # Aylık sigorta ödemesi durumu
    df.loc[df["MONTHLY PREMIUM AUTO"] <= 70, "NEW_PAYMENT_STATUS"] = "Underpayment"
    df.loc[(df["MONTHLY PREMIUM AUTO"] > 70) & (df["MONTHLY PREMIUM AUTO"] < 100), "NEW_PAYMENT_STATUS"] = "Medium"
    df.loc[df["MONTHLY PREMIUM AUTO"] >= 100, "NEW_PAYMENT_STATUS"] = "Overpayment"
    # Gelir durumu
    df.loc[df["INCOME"] == 0, "NEW_INCOME_STATUS"] = "No Income"
    df.loc[(df["INCOME"] > 0) & (df["INCOME"] <= 35000), "NEW_INCOME_STATUS"] = "Low Income"
    df.loc[(df["INCOME"] > 35000) & (df["INCOME"] <= 60000), "NEW_INCOME_STATUS"] = "Medium Income"
    df.loc[df["INCOME"] > 60000, "NEW_INCOME_STATUS"] = "High Income"
    # Müşteriye ödenen sigorta parası
    df.loc[df["TOTAL CLAIM AMOUNT"] <= 200, "NEW_CLAIM_STATUS"] = "Low"
    df.loc[(df["TOTAL CLAIM AMOUNT"] > 200) & (df["TOTAL CLAIM AMOUNT"] <= 550), "NEW_CLAIM_STATUS"] = "Medium"
    df.loc[df["TOTAL CLAIM AMOUNT"] > 550, "NEW_CLAIM_STATUS"] = "High"
    # Müşteriden elde edilen kar zarar miktarı
    df["NEW_PROFIT"] = (df["MONTHLY PREMIUM AUTO"] * df["MONTHS SINCE POLICY INCEPTION"])
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] > 0, "NEW_PROFIT"] = (df["MONTHLY PREMIUM AUTO"] * df["MONTHS SINCE POLICY INCEPTION"]) - df["TOTAL CLAIM AMOUNT"]
    # kazanç-kayıp durumu
    df.loc[df["NEW_PROFIT"] <= 0, "NEW_PROFIT-LOSS"] = "Loss"
    df.loc[df["NEW_PROFIT"] > 0, "NEW_PROFIT-LOSS"] = "Gain"
    # Müşterinin ödediği aylık sigorta parasının ortalama ödenen sigorta parasına oranı
    df["NEW_PAYMENT RATE"] = df["MONTHLY PREMIUM AUTO"] / df["MONTHLY PREMIUM AUTO"].mean()
    # Müşterilerin yıllık gelirlerin ortalama yıllık gelire oranı
    df["NEW_INCOME RATE"] = df["INCOME"] / df["INCOME"].mean()
    # Since Claim - Monthly premıum auto ilişkisi
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "NEW_CLAIM-INCOME"] = df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "MONTHLY PREMIUM AUTO"] / df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "MONTHLY PREMIUM AUTO"].mean()
    df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "NEW_CLAIM-INCOME"] = df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "MONTHLY PREMIUM AUTO"] / df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "MONTHLY PREMIUM AUTO"].mean()
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "NEW_CLAIM-INCOME"] = df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "MONTHLY PREMIUM AUTO"] / df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "MONTHLY PREMIUM AUTO"].mean()
    # Ödenen Sigorta Parası Ortalamadan Fazla mı?
    df.loc[df["TOTAL CLAIM AMOUNT"] > df["TOTAL CLAIM AMOUNT"].mean(), "NEW_CLAIM_CAT"] = "High"
    df.loc[df["TOTAL CLAIM AMOUNT"] < df["TOTAL CLAIM AMOUNT"].mean(), "NEW_CLAIM_CAT"] = "Low"
    # Sigorta Geri Ödemesi Bir Önceki Sözleşmeye mi ait?
    df["NEW_PREVIOUS AGREE"] = "No"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > df["MONTHS SINCE POLICY INCEPTION"], "NEW_PREVIOUS AGREE"] = "Yes"
    # Poliçe sayısı 2'den küçük mü?
    df.loc[df["NUMBER OF POLICIES"] <= 2, "NEW_FEW POLİCY"] = "Yes"
    df.loc[df["NUMBER OF POLICIES"] > 2, "NEW_FEW POLİCY"] = "No"

    df.columns = [col.upper() for col in df.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

    replace_with_thresholds(df, "CUSTOMER LIFETIME VALUE", q1=0.21, q3=0.79)
    replace_with_thresholds(df, "MONTHLY PREMIUM AUTO", q1=0.35, q3=0.65)
    replace_with_thresholds(df, "TOTAL CLAIM AMOUNT", q1=0.25, q3=0.75)
    replace_with_thresholds(df, "NEW_PROFIT", q1=0.15, q3=0.85)

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    df.columns = [col.upper() for col in df.columns]

    num_cols = [col for col in num_cols if "CUSTOMER LIFETIME VALUE" not in col]

    X_scaled = MinMaxScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    df_test = df[8099:].drop(["CUSTOMER LIFETIME VALUE"], axis=1)
    df = df[:8099]
    y = df["CUSTOMER LIFETIME VALUE"]
    X = df.drop(["CUSTOMER LIFETIME VALUE"], axis=1)

    print("Data Preprocessing Finished.")

    return df_test, X, y

# Base Models

df_test, X, y = cltv_data_preb(df)

def base_models(X, y, scoring="neg_mean_squared_error"):
    print("#### BASE MODELS ####")
    regressors = [('LR', LinearRegression()),
                   ('KNN', KNeighborsRegressor()),
                   ("CART", DecisionTreeRegressor()),
                   ("RF", RandomForestRegressor()),
                   ('Adaboost', AdaBoostRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('LightGBM', LGBMRegressor()),
                   ('CatBoost', CatBoostRegressor(verbose=False))
                   ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=10, scoring=scoring)
        print(f"######### {name} #########")
        print(f"RMSE: {round(np.mean(np.sqrt(-cv_results['test_score'])), 4)}")
        fitting = regressor.fit(X,y)
        print(f"R^2: {round(fitting.score(X,y), 4)}")

base_models(X, y)

# MODELS

# max_features: bölünmelerde göz önünde bulundurulacak değişken sayısı
# max_depth: derinlik
# min_samples_split: bir düğümün dallanmaya imkan vermek için kaç tane split olması gerektiği
# n_estimators: fit edilecek bağımsız ağaç sayısı

def best_models(X, y):

    print("Best Models...")

    rf = RandomForestRegressor(random_state=17)
    #catboost = CatBoostRegressor(random_state=17)
    #lgbm = LGBMRegressor(random_state=17)

    print("Random Forest...")
    #rf_params = {"max_depth": [2, None],
    #             "max_features": ["auto"],
    #             "min_samples_split": [4, 6, 8],
    #             "n_estimators": [300, 500, 700]}
    #rf_best_grid = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
    #print(f"Random Forest Best Params: {rf_best_grid.best_params_}")
    best_par = {"max_depth": None,
                "max_features": 'auto',
                "min_samples_split": 6,
                "n_estimators": 500}
    rf_final = rf.set_params(**best_par, random_state=17).fit(X, y)
    #print(f"Random Forest 10-CV Score: {cross_validate(rf_final, X, y, cv=10, scoring='neg_mean_squared_error')}")
    #print(f"Random Forest R^2 Score: {rf_final.score(X, y)}")

    #print("CatBoost...")
    #catboost_params = {"iterations": [300, 400, 500],
    #                   "learning_rate": [0.01, 0.05, 0.1],
    #                   "depth": [4, 6, 10]}
    #catboost_best_grid = GridSearchCV(catboost, catboost_params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
    #print(f"Random Forest Best Params: {catboost_best_grid.best_params_}")
    #catboost_final = catboost.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
    #print(f"CatBoost 10-CV Score: {cross_validate(catboost_final, X, y, cv=10, scoring='neg_mean_squared_error')}")
    #print(f"CatBoost R^2 Score: {catboost_final.score(X, y)}")

    #print("LightGBM...")
    #lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
    #               "n_estimators": [300, 500, 700],
    #               "colsample_bytree": [0.5, 0.7, 1]}
    #lightgbm_best_grid = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
    #print(f"LightGBM Best Params: {lightgbm_best_grid.best_params_}")
    #lgbm_final = lgbm.set_params(**lightgbm_best_grid.best_params_, random_state=17).fit(X, y)
    #print(f"LightGBM 10-CV Score: {cross_validate(lgbm_final, X, y, cv=10, scoring='neg_mean_squared_error')}")
    #print(f"LightGBM R^2 Score: {lgbm_final.score(X, y)}")

    print("Models Finished.")

    # return , catboost_final, lgbm_final
    return rf_final

rf_final = best_models(X, y)

def score(model):
    cv_results = cross_validate(lgbm_final, X, y, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    print(f"10-CV Score: {np.mean(np.sqrt(-cv_results['test_score']))}")
    print(f"R^2 Score: {catboost_final.score(X, y)}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:20])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

# Saving and Loading Model

joblib.dump(rf_final, "rf_final.joblib", compress=3)

model_from_disc = joblib.load("rf_final.pkl")

# Prediction

y_pred = model_from_disc.predict(df_test)

df.loc[8099:, "Customer Lifetime Value"] = y_pred

# Segments

df.loc[df["Customer Lifetime Value"] < 3500, "Segments"] = "Bronze"
df.loc[(df["Customer Lifetime Value"] >= 3500) & (df["Customer Lifetime Value"] < 5500), "Segments"] = "Silver"
df.loc[(df["Customer Lifetime Value"] >= 5500) & (df["Customer Lifetime Value"] < 8500), "Segments"] = "Gold"
df.loc[df["Customer Lifetime Value"] >= 8500, "Segments"] = "Diamond"

df.to_csv('New_Segments.csv')