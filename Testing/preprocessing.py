from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def data_preprocessing(data):
    
#     print(" Droping columns")
#     data = data.drop('IsFieldForeignKey', axis=1)
#     data = data.drop('IsNullValueExistInNumeric', axis=1)
#     data = data.drop('HasFieldUniqueValue', axis=1)

    
    
#     print(" Encodding")
    
    le = LabelEncoder()
    fields = pd.read_csv("Testing/ListOf_FieldDataType_Encoding.csv")
    fields = fields.drop_duplicates()
    fiel_dict = fields.set_index(['FieldDataType'])['FieldDataType_num'].to_dict()
    data['FieldDataType'] = data.apply(lambda row: fiel_dict.get((row['FieldDataType']), np.nan), axis=1)
    category_enc = pd.read_csv("Testing/listOf_category_encoding_label.csv")
    category_enc = category_enc.drop_duplicates()
    category_dict = category_enc.set_index(['category_dup'])['category_num'].to_dict()
    data['category_dup'] = data.apply(lambda row: category_dict.get((row['category_dup']), np.nan), axis=1)
    data['IsFieldNotNull'] = le.fit_transform(data['IsFieldNotNull'])

    
    
    
#     print("Replacing null values by mean value")
    
    mean_MinimumFieldLength = data['MinimumFieldLength'].mean()
    data['MinimumFieldLength'] = data['MinimumFieldLength'].fillna(mean_MinimumFieldLength)
    mean_MaximumField = data['MaximumField'].mean()
    data['MaximumField'] = data['MaximumField'].fillna(mean_MaximumField)
    mean_AverageFieldLength = data['AverageFieldLength'].mean()
    data['AverageFieldLength'] = data['AverageFieldLength'].fillna(mean_AverageFieldLength)
    
    
    
    

#     print("Range transformation for balancing data.")
    
    col = 'TotalNumberOfRecords'
    data[col] = np.where((data[col] <= 5),0,data[col])
    data[col] = np.where(((data[col] > 5) & (data[col] <=150)),1,data[col])
    data[col] = np.where(((data[col] > 150) & (data[col] <=1200)),2,data[col])
    data[col] = np.where(((data[col] > 1200) & (data[col] <=3500)),3,data[col])
    data[col] = np.where(((data[col] > 3500) & (data[col] <=8500)),4,data[col])
    data[col] = np.where(((data[col] > 8500) & (data[col] <=18500)),5,data[col])
    data[col] = np.where(((data[col] > 18500) & (data[col] <=36000)),6,data[col])
    data[col] = np.where(((data[col] > 36000) & (data[col] <=80000)),7,data[col])
    data[col] = np.where(((data[col] > 80000) & (data[col] <=220000)),8,data[col])
    data[col] = np.where((data[col] > 220000),9,data[col])

    col = 'LengthOfField'
    data[col] = np.where((data[col] <= 6),0,data[col])
    data[col] = np.where(((data[col] > 6) & (data[col] <=31)),1,data[col])
    data[col] = np.where((data[col] > 31),2,data[col])
    data[col] = data[col].fillna(-5)

    col = 'LengthOfFieldPrecision'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=9)),1,data[col])
    data[col] = np.where(((data[col] > 9) & (data[col] <=17)),2,data[col])
    data[col] = np.where((data[col] > 17),3,data[col])
    data[col] = data[col].fillna(-5)

    col = 'LengthOfFieldScale'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=4)),1,data[col])
    data[col] = np.where((data[col] > 4),2,data[col])
    data[col] = data[col].fillna(0)

    col = 'LengthOfFieldScale'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=4)),1,data[col])
    data[col] = np.where((data[col] > 4),2,data[col])
    data[col] = data[col].fillna(0)

    col = 'MinimumValueOfNumreic'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] ==1)),1,data[col])
    data[col] = np.where((data[col] > 1),2,data[col])

    col = 'MaximumValueOfNumreic'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=15)),1,data[col])
    data[col] = np.where(((data[col] > 15) & (data[col] <=7000)),2,data[col])
    data[col] = np.where(((data[col] > 7000) & (data[col] <=200000)),3,data[col])
    data[col] = np.where((data[col] > 200000),4,data[col])

    col = 'AverageValueOfNumreic'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=3)),1,data[col])
    data[col] = np.where(((data[col] > 3) & (data[col] <=85)),2,data[col])
    data[col] = np.where(((data[col] > 85) & (data[col] <=1500)),3,data[col])
    data[col] = np.where((data[col] > 1500),4,data[col])

    col = 'VarianceValueOfNumeric'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=500)),1,data[col])
    data[col] = np.where(((data[col] > 500) & (data[col] <=600000)),2,data[col])
    data[col] = np.where(((data[col] > 600000) & (data[col] <=85000000)),3,data[col])
    data[col] = np.where((data[col] > 85000000),4,data[col])

    col = 'CoefficientOfvarianceOfNumeric'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=3)),1,data[col])
    data[col] = np.where(((data[col] > 3) & (data[col] <=13)),2,data[col])
    data[col] = np.where((data[col] > 13),3,data[col])

    col = 'MinimumFieldLength'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=1)),1,data[col])
    data[col] = np.where(((data[col] > 1) & (data[col] <=4)),2,data[col])
    data[col] = np.where((data[col] > 4),3,data[col])

    col = 'MaximumField'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=1)),1,data[col])
    data[col] = np.where(((data[col] > 1) & (data[col] <=4)),2,data[col])
    data[col] = np.where(((data[col] > 4) & (data[col] <=6)),3,data[col])
    data[col] = np.where(((data[col] > 6) & (data[col] <=13)),4,data[col])
    data[col] = np.where((data[col] > 13),5,data[col])

    col = 'AverageFieldLength'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=1)),1,data[col])
    data[col] = np.where((data[col] > 1),2,data[col])

    col = 'RatioOfWhitespaceToLengthInCharacter'
    data[col] = np.where((data[col] <= 0),0,data[col])
    data[col] = np.where(((data[col] > 0) & (data[col] <=9)),1,data[col])
    data[col] = np.where((data[col] > 9),2,data[col])

#     print("Merging columns")
    
    data['TableName'] = data['TableName'] + ' ' + data['ERPFieldName']
    data = data.drop('ERPFieldName', axis = 1)
    
    data = data.rename(columns={'IsFieldNotNull': 'IsFieldNotNull_num', 'FieldDataType': 'FieldDataType_num'})

    cols_to_keep = ['ERPName','TableName', 'TotalNumberOfRecords', 'FieldDataType_num',
       'LengthOfField', 'IsFieldPrimaryKey', 'IsFieldNotNull_num',
       'FieldHasConstraints', 'LengthOfFieldPrecision', 'LengthOfFieldScale',
       'MinimumValueOfNumreic', 'MaximumValueOfNumreic',
       'AverageValueOfNumreic', 'VarianceValueOfNumeric',
       'CoefficientOfvarianceOfNumeric', 'MinimumFieldLength', 'MaximumField',
       'AverageFieldLength', 'RatioOfWhitespaceToLengthInCharacter', 'category_dup']
    
    data1 = data.drop(columns=[col for col in data.columns if col not in cols_to_keep])

    
    return data1