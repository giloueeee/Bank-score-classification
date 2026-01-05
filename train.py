import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

raw_train_set=pd.read_csv('data/train.csv', index_col='ID').copy()
raw_test_set=pd.read_csv('data/test.csv', index_col ='ID').copy()

unrelevant_features=['Month', 'SSN', 'Customer_ID', 'Name','Monthly_Inhand_Salary', 'Credit_Mix' ]

relevant_train_set=raw_train_set.drop_duplicates()
relevant_test_set=raw_test_set.drop_duplicates()

relevant_train_set=relevant_train_set.drop(axis=1, columns=unrelevant_features)
relevant_test_set=relevant_test_set.drop(axis=1, columns=unrelevant_features)

###Age cleaning
relevant_train_set['Age']=pd.to_numeric(relevant_train_set['Age'].astype(str).str.replace("_",""))
cleaned_age_train_set= relevant_train_set[(relevant_train_set['Age'] >= 18) & (relevant_train_set['Age'] <= 60)] ###Removing rows with outliers, probably due to people lying about their age, their entire row become supicious
print(relevant_train_set['Age'].head(100))
print(relevant_train_set['Age'].dtype)
print("sum of unique ages: ", len(cleaned_age_train_set['Age'].unique()))

###Occupation cleaning
cleaned_occupation_train_set=cleaned_age_train_set
cleaned_occupation_train_set = cleaned_occupation_train_set[~cleaned_occupation_train_set['Occupation'].astype(str).str.contains("_", na=False)]
print("sum of unique occupations: ", len(cleaned_occupation_train_set['Occupation'].unique()))
print(len(cleaned_occupation_train_set))


###Annual Income cleaning
cleaned_income_train_set=cleaned_occupation_train_set
# Convert Annual_Income to numeric (handling underscores and other formatting)
cleaned_income_train_set['Annual_Income'] = pd.to_numeric(cleaned_income_train_set['Annual_Income'].astype(str).str.replace("_",""))
print("sum of rows with annual income > 2000000: ", len(cleaned_income_train_set[cleaned_income_train_set['Annual_Income']>2000000])) ####Removing rows with annual income > 2000000, unbalanced dataset
cleaned_income_train_set=cleaned_income_train_set[cleaned_income_train_set['Annual_Income']<2000000]

print("cleaned income set:", cleaned_income_train_set['Annual_Income'].head(5))
print(len(cleaned_income_train_set))


###Cleaning Num_Bank_Accounts
cleaned_bank_accounts_train_set=cleaned_income_train_set
print("more than 10 bank accounts :",len(cleaned_bank_accounts_train_set[cleaned_bank_accounts_train_set['Num_Bank_Accounts']>10]))
cleaned_bank_accounts_train_set=cleaned_bank_accounts_train_set[(cleaned_bank_accounts_train_set['Num_Bank_Accounts']<=10) & (cleaned_bank_accounts_train_set['Num_Bank_Accounts']>=0)]
print("cleaned bank accounts set:", cleaned_bank_accounts_train_set['Num_Bank_Accounts'].head(5))
print(len(cleaned_bank_accounts_train_set))


###Cleaning Num_Credit_Card
cleaned_credit_cards_set=cleaned_bank_accounts_train_set
cleaned_credit_cards_set = cleaned_credit_cards_set[cleaned_credit_cards_set['Num_Credit_Card']<=11]
print("cleaned credit cards set:", cleaned_credit_cards_set['Num_Credit_Card'].head(5))
print(len(cleaned_credit_cards_set))

###Cleaning interest rate

cleaned_interest_rate_set=cleaned_credit_cards_set

cleaned_interest_rate_set_crop_test = cleaned_interest_rate_set[cleaned_interest_rate_set['Interest_Rate']<60]
#plt.figure(figsize=(10, 6))
#sns.histplot(data=cleaned_interest_rate_set_crop_test, x='Interest_Rate',  bins=60, binrange=(0, 60))
#plt.xlim(0, 60)
#plt.show()

print("more than 35 interest rate :",len(cleaned_interest_rate_set[cleaned_interest_rate_set['Interest_Rate']>35]))
###only 1581 rows with interest rate superior to 35, probably errors of typing, we will remove them
cleaned_interest_rate_set_cropped = cleaned_interest_rate_set[cleaned_interest_rate_set['Interest_Rate']<35]
print(len(cleaned_interest_rate_set_cropped))
print("cleaned_interest_rate_cropped:", cleaned_interest_rate_set_cropped['Interest_Rate'].head(5))


###Cleaning Num_of_Loan
cleaned_numofloan = cleaned_interest_rate_set_cropped
cleaned_numofloan['Num_of_Loan'] = cleaned_numofloan["Num_of_Loan"].astype(str).str.replace("_","").astype(int)
cleaned_numofloan_croptest = cleaned_numofloan[(cleaned_numofloan['Num_of_Loan']<=20) & (cleaned_numofloan['Num_of_Loan']>=0)]

#sns.histplot(data=cleaned_numofloan, x='Num_of_Loan', bins=20, binrange=(0, 20))
#plt.xlim(0, 20)
#plt.xticks(range(0, 21, 1))
#plt.show()

cleaned_numofloan_cropped = cleaned_numofloan[(cleaned_numofloan['Num_of_Loan']<=10) & (cleaned_numofloan['Num_of_Loan']>=0)]

print("cleaned num of loan:", cleaned_numofloan_cropped['Num_of_Loan'].head(5))
print(len(cleaned_numofloan_cropped))

###Cleaning delay from due date
cleaned_delay_from_due_date = cleaned_numofloan_cropped
cleaned_delay_from_due_date = cleaned_delay_from_due_date[cleaned_delay_from_due_date["Delay_from_due_date"]>=0]

print("cleaned delay from due date:", cleaned_delay_from_due_date['Delay_from_due_date'].head(5))
print(len(cleaned_delay_from_due_date))

###Cleaning Num_of_Delayed_Payment

cleaned_payment_delay = cleaned_delay_from_due_date

print(len(cleaned_payment_delay))
cleaned_payment_delay['Num_of_Delayed_Payment'] = pd.to_numeric(cleaned_payment_delay["Num_of_Delayed_Payment"].astype(str).str.replace("_",""), errors='coerce')
cleaned_payment_delay_withoutnan = cleaned_payment_delay

#sns.histplot(data=cleaned_payment_delay_withoutnan, x='Num_of_Delayed_Payment', bins=40, binrange=(0, 80))
#plt.xlim(0, 80)
#plt.show()

cleaned_payment_delay = cleaned_payment_delay[cleaned_payment_delay["Num_of_Delayed_Payment"]<=30]
cleaned_payment_delay = cleaned_payment_delay[cleaned_payment_delay["Num_of_Delayed_Payment"].fillna(0) >= 0]
print("cleaned_payment_delay length:", len(cleaned_payment_delay))
print("cleaned payment delay:", cleaned_payment_delay['Num_of_Delayed_Payment'].head(5))


###Cleaning Changed_Credit_Limit

cleaned_changed_Credit_Limit = cleaned_payment_delay
cleaned_changed_Credit_Limit['Changed_Credit_Limit'] = cleaned_changed_Credit_Limit["Changed_Credit_Limit"].astype(str).str.replace("_","NaN")
print("cleaned_changed_Credit_Limit length:", len(cleaned_changed_Credit_Limit))
print("cleaned changed credit limit:", cleaned_changed_Credit_Limit['Changed_Credit_Limit'].head(10))

###cleaning Outstanding_Debt
cleaned_outstanding_debt = cleaned_changed_Credit_Limit
cleaned_outstanding_debt['Outstanding_Debt'] = cleaned_outstanding_debt["Outstanding_Debt"].astype(str).str.replace("_","").astype(float)
print("cleaned_outstanding_debt length:", len(cleaned_outstanding_debt))

###Credit_History_Age
###Credit_History_Age
cleaned_credit_history = cleaned_outstanding_debt
cleaned_credit_history['Credit_History_Age'] = cleaned_credit_history["Credit_History_Age"].replace("NA", pd.NA)
cleaned_credit_history = cleaned_credit_history.dropna(subset=['Credit_History_Age'])

# Extract years and months to a temporary DataFrame
years_months = cleaned_credit_history['Credit_History_Age'].astype(str).str.extract(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?')
years = pd.to_numeric(years_months[0], errors='coerce')
months = pd.to_numeric(years_months[1], errors='coerce')
# Convert to total months
cleaned_credit_history['Credit_History_Age'] = years * 12 + months

print("cleaned_credit_history_age length:", len(cleaned_credit_history))
print("cleaned credit history age:", cleaned_credit_history['Credit_History_Age'].head(10))


###Cleaning payment min amount

cleaned_payment_min_amount = cleaned_credit_history
cleaned_payment_min_amount['Payment_of_Min_Amount'] = cleaned_payment_min_amount["Payment_of_Min_Amount"].replace("NM", pd.NA)
print("cleaned_payment_min_amount length:", len(cleaned_payment_min_amount))
print("cleaned payment min amount:", cleaned_payment_min_amount['Payment_of_Min_Amount'].head(10))

###cleaned Amount_invested_monthly
cleaned_amount_invested_monthly = cleaned_payment_min_amount
cleaned_amount_invested_monthly['Amount_invested_monthly'] = cleaned_amount_invested_monthly["Amount_invested_monthly"].astype(str).str.replace("_","").astype(float)
print("cleaned amount invested monthly:", cleaned_amount_invested_monthly['Amount_invested_monthly'].head(10))
print("cleaned amount invested monthly length:", len(cleaned_amount_invested_monthly))

###cleaning Payment_Behaviour
cleaned_payment_behavior = cleaned_amount_invested_monthly
cleaned_payment_behavior['Payment_Behaviour'] = cleaned_payment_behavior["Payment_Behaviour"].replace("!@9#%8", pd.NA)
print("cleaned payment behavior length:", len(cleaned_payment_behavior))
print("cleaned payment behavior:", cleaned_payment_behavior['Payment_Behaviour'].head(10))


###cleaned Monthly_Balance

cleaned_Monthly_Balance = cleaned_payment_behavior
cleaned_Monthly_Balance['Monthly_Balance'] = cleaned_Monthly_Balance["Monthly_Balance"].astype(str).str.replace("_","").astype(float)
cleaned_Monthly_Balance = cleaned_Monthly_Balance[cleaned_Monthly_Balance["Monthly_Balance"]>0]
print("cleaned monthly balance:", cleaned_Monthly_Balance['Monthly_Balance'].head(10))
print("cleaned monthly balance length:", len(cleaned_Monthly_Balance))


output=cleaned_Monthly_Balance.copy()
output.to_csv('data/cleaned_data.csv', index=False)


cleaned_train_set=cleaned_Monthly_Balance
cleaned_test_set=relevant_test_set

X=cleaned_train_set.drop('Credit_Score', axis=1)

y=cleaned_train_set['Credit_Score']

print("y: ", y.head(10))
# Check class distribution before balancing
print("\nClass distribution before balancing:")
print(y.value_counts())

# Combine X and y to balance together
balanced_df = pd.concat([X, y], axis=1)

# Find the minimum count among all classes
min_class_count = y.value_counts().min()
print(f"\nMinimum class count: {min_class_count}")
print(f"Balancing all classes to {min_class_count} samples each")

# Sample equal number from each class
balanced_samples = []
for credit_score in y.unique():
    class_data = balanced_df[balanced_df['Credit_Score'] == credit_score]
    # Sample min_class_count rows (or all if less than min_class_count)
    sampled = class_data.sample(n=min(min_class_count, len(class_data)), random_state=42)
    balanced_samples.append(sampled)

# Combine all balanced samples
balanced_df = pd.concat(balanced_samples, ignore_index=False)

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=False)

# Split back into X and y
X = balanced_df.drop('Credit_Score', axis=1)
y = balanced_df['Credit_Score']

print("\nClass distribution after balancing:")
print(y.value_counts())
print(f"\nTotal samples after balancing: {len(y)}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

###imputation

numerical_cols = ["Changed_Credit_Limit", "Num_Credit_Inquiries", "Amount_invested_monthly", "Monthly_Balance"]

mean_Imputer=SimpleImputer(strategy="median")
X_train_numbers_imputed= pd.DataFrame(mean_Imputer.fit_transform(X_train[numerical_cols]), index=X_train.index)
X_val_numbers_imputed= pd.DataFrame(mean_Imputer.transform(X_val[numerical_cols]), index=X_val.index)

X_train = X_train.drop(columns=numerical_cols)
X_val = X_val.drop(columns=numerical_cols)

X_train_numbers_imputed.columns = numerical_cols
X_val_numbers_imputed.columns = numerical_cols

X_train = pd.concat([X_train, X_train_numbers_imputed], axis=1)
X_val = pd.concat([X_val, X_val_numbers_imputed], axis=1)

print("X_train_numbers_imputed: ", X_train_numbers_imputed.columns)
# Fill missing values with the most frequent value for each column
for col in ["Payment_of_Min_Amount", "Payment_Behaviour"]:
    mode_value = X_train[col].mode()[0] if not X_train[col].mode().empty else X_train[col].value_counts().index[0]
    X_train[col] = X_train[col].fillna(mode_value)
    X_val[col] = X_val[col].fillna(mode_value)


###categorization
### type of loan
cleaned_type_of_loan = cleaned_delay_from_due_date
cleaned_type_of_loan['Type_of_Loan'] = cleaned_type_of_loan["Type_of_Loan"]

def split_loan_types(df, col_name='Type_of_Loan'):
    """
    Split comma-separated loan types into separate binary columns.
    Each loan type becomes a column with 1 if present, 0 if not.
    """
    # Handle NaN values
    df[col_name] = df[col_name].fillna('')
    
    # Get all unique loan types from the entire dataset
    all_loan_types = set()
    for value in df[col_name]:
        if pd.notna(value) and value != '':
            # Split by comma and strip whitespace
            loan_types = [loan.strip() for loan in str(value).split(',')]
            all_loan_types.update(loan_types)
    
    # Remove empty strings
    all_loan_types = {loan for loan in all_loan_types if loan}
    
    # Create binary columns for each loan type
    for loan_type in all_loan_types:
        # Create column name: replace spaces with underscores, make it clean
        col_name_clean = loan_type.replace(' ', '_').replace('-', '_')
        df[f'Has_{col_name_clean}'] = df[col_name].apply(
            lambda x: 1 if pd.notna(x) and loan_type in str(x) else 0
        )
    
    # Drop the original Type_of_Loan column
    df = df.drop(columns=[col_name])
    
    return df, list(all_loan_types)

# First, get all unique loan types from both train and val to ensure consistency
# Combine to find all possible loan types
combined_loan_data = pd.concat([X_train[['Type_of_Loan']], X_val[['Type_of_Loan']]], axis=0)
all_loan_types = set()
for value in combined_loan_data['Type_of_Loan']:
    if pd.notna(value) and value != '':
        loan_types = [loan.lstrip('and ') for loan in str(value).split(',')]
        all_loan_types.update(loan_types)
all_loan_types = {loan for loan in all_loan_types if loan}

# Create binary columns for train set
X_train['Type_of_Loan'] = X_train['Type_of_Loan'].fillna('')
for loan_type in all_loan_types:
    col_name_clean = loan_type.replace(' ', '_').replace('-', '_')
    X_train[f'Has_{col_name_clean}'] = X_train['Type_of_Loan'].apply(
    lambda x: 1 if loan_type in [lt.strip().lstrip('and ') for lt in str(x).split(',')] else 0
    )

# Create binary columns for validation set
X_val['Type_of_Loan'] = X_val['Type_of_Loan'].fillna('')
for loan_type in all_loan_types:
    col_name_clean = loan_type.replace(' ', '_').replace('-', '_')
    X_val[f'Has_{col_name_clean}'] = X_val['Type_of_Loan'].apply(
        lambda x: 1 if loan_type in [lt.strip().lstrip('and ') for lt in str(x).split(',')] else 0
    )

X_train = X_train.drop(columns=['Type_of_Loan'])
X_val = X_val.drop(columns=['Type_of_Loan'])

print(f"Created {len(all_loan_types)} loan type columns: {list(all_loan_types)}")
print("Sample of new columns:", [col for col in X_train.columns if 'Has_' in col][:5])

print("cleaned type of loan:", cleaned_type_of_loan['Type_of_Loan'].head(5))
print(len(cleaned_type_of_loan))

OH_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
OH_train = pd.DataFrame(OH_encoder.fit_transform(X_train[["Occupation","Payment_of_Min_Amount","Payment_Behaviour"]]))
OH_val = pd.DataFrame(OH_encoder.transform(X_val[["Occupation","Payment_of_Min_Amount","Payment_Behaviour"]]))

OH_train.index = X_train.index
OH_val.index = X_val.index

X_train=X_train.drop(columns=["Occupation","Payment_of_Min_Amount","Payment_Behaviour"])
X_val=X_val.drop(columns=["Occupation","Payment_of_Min_Amount","Payment_Behaviour"])

X_train_OH = pd.concat([X_train, OH_train], axis=1)
X_val_OH = pd.concat([X_val, OH_val], axis=1)

y_Ordinal_encoder = OrdinalEncoder()
y_Ordinal_train = pd.DataFrame(y_Ordinal_encoder.fit_transform(y_train.values.reshape(-1, 1))).astype(int)
y_Ordinal_val = pd.DataFrame(y_Ordinal_encoder.transform(y_val.values.reshape(-1, 1))).astype(int)

y_Ordinal_train.index = y_train.index
y_Ordinal_val.index = y_val.index


X_train = X_train_OH
X_val = X_val_OH
y_train = y_Ordinal_train
y_val = y_Ordinal_val

###model training
accuracylist = []
precisionlist = []
recalllist = []
f1list = []
treeslist = []
depthlist = []
learning_ratelist = []

trainaccuracylist = []
trainprecisionlist = []
trainrecalllist = []
trainf1list = []
traintreeslist = []
traindepthlist = []
trainlearning_ratelist = []
counter = 0

n_estimators = [110, 125, 150, 170, 200, 300, 400, 500, 750, 1000]
max_depth = [ 13,14,15, 20, 25]
learning_rate = [0.05, 0.1, 0.15]
total_attempts = len(n_estimators) * len(max_depth) * len(learning_rate)

def train_model(X_train, y_train, X_val, y_val, n_estimators, max_depth, learning_rate):
    global counter
    counter = counter + 1
    model = XGBClassifier( 
    n_estimators=n_estimators, 
    learning_rate=learning_rate, 
    max_depth=max_depth,
    random_state=3                 
    )
    model.fit(X_train, y_train)
    predictions=model.predict(X_val)
    trainpredictions=model.predict(X_train)
    
    # Validation metrics
    accuracy=accuracy_score(y_val, predictions)
    precision=precision_score(y_val, predictions, average='weighted')
    recall=recall_score(y_val, predictions, average='weighted')
    f1=f1_score(y_val, predictions, average='weighted')
    accuracylist.append(accuracy)
    precisionlist.append(precision)
    recalllist.append(recall)
    f1list.append(f1)
    treeslist.append(n_estimators)
    depthlist.append(max_depth)
    learning_ratelist.append(learning_rate)
    
    # Training metrics
    train_accuracy=accuracy_score(y_train, trainpredictions)
    train_precision=precision_score(y_train, trainpredictions, average='weighted')
    train_recall=recall_score(y_train, trainpredictions, average='weighted')
    train_f1=f1_score(y_train, trainpredictions, average='weighted')
    trainaccuracylist.append(train_accuracy)
    trainprecisionlist.append(train_precision)
    trainrecalllist.append(train_recall)
    trainf1list.append(train_f1)
    traintreeslist.append(n_estimators)
    traindepthlist.append(max_depth)
    trainlearning_ratelist.append(learning_rate)
    
    print("attempt :", counter, "out of :", total_attempts, " accuracy: ", accuracy, "with trees: ", n_estimators, "and depth: ", max_depth)
    return accuracy, counter

for l in learning_rate:
    for n in n_estimators:
        for m in max_depth:
            train_model(X_train, y_train, X_val, y_val, n, m, l)

best_score=max(accuracylist)
min_score_index=accuracylist.index(best_score)
best_trees=treeslist[min_score_index]
best_depth=depthlist[min_score_index]
best_learning_rate=learning_ratelist[min_score_index]
best_precision=precisionlist[min_score_index]
best_recall=recalllist[min_score_index]
best_f1=f1list[min_score_index]

print("best score: ", best_score)
print("best trees: ", best_trees)
print("best depth: ", best_depth)
print("best learning rate: ", best_learning_rate)
print("best precision: ", best_precision)
print("best recall: ", best_recall)
print("best f1: ", best_f1) 
# Filter data for the best learning rate to avoid duplicate entries in pivot
best_lr_data = pd.DataFrame({
    'trees': treeslist, 
    'depth': depthlist,
    'learning_rate': learning_ratelist,
    'scores': accuracylist
})
best_lr_data = best_lr_data[best_lr_data['learning_rate'] == best_learning_rate]
heatmap_dataframe = best_lr_data.pivot(index='trees', columns='depth', values='scores')

sns.heatmap(data=heatmap_dataframe, annot=True, fmt='.5f')
plt.show()

Train_lr_data = pd.DataFrame({
    'train_trees': traintreeslist,
    'train_depth': traindepthlist,
    'learning_rate': trainlearning_ratelist,
    'scores': trainaccuracylist
})
Train_lr_data = Train_lr_data[Train_lr_data['learning_rate'] == best_learning_rate]
heatmap_dataframe = Train_lr_data.pivot(index='train_trees', columns='train_depth', values='scores')

sns.heatmap(data=heatmap_dataframe, annot=True, fmt='.5f')
plt.show()