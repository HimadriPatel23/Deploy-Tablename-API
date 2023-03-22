import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def test_model(model, data, preprocess = False):
    
    from Testing.preprocessing import data_preprocessing
    if preprocess == True:
        # print("Preprocessing data")
        data = data_preprocessing(data)
        # print("Done pre")
    else:
        # print("Without prep")
        data = data
        # print("Done without prep")
        
    # print(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    predicted_data = model.predict(X)
    res = X.copy()
    res['category_num'] = y
    res['predicted'] = predicted_data
    res['result'] = res['predicted']
    res['result'] = np.where((res['predicted'] == res['category_num']),1,res['result'])
    res['result'] = np.where((res['predicted'] != res['category_num']), 0, res['result'])
    
    #res.to_csv('predicted_data.csv')

    category_dict = {0: 'ACC', 1: 'AP', 2: 'AR', 3: 'BOM', 4: 'CB', 5: 'CON', 6: 'CRM', 7: 'FA', 8: 'GL', 9: 'INV', 10: 'PO', 11: 'PRJ', 12: 'RMA', 13: 'SAL', 14: 'SHIP', 15: 'WO', 16: 'un'}
    res['predicted_decode'] = res.apply(lambda row: category_dict.get((row['predicted']), np.nan), axis=1)

    df = res[['TableName', 'predicted_decode']]

    formatted_data = df.to_dict('records')
    formatted_data
    
    # Dictionary to store the information for each table
    table_info = {}
    result = pd.DataFrame()

    for row in formatted_data:
        table_name, field_name = row['TableName'].split()
        category = row['predicted_decode']
        
        # Update the dictionary for the current table
        if table_name in table_info:
            table_info[table_name]['predicted_categories'].append(category)
            table_info[table_name]['category_counts'][category] = table_info[table_name]['category_counts'].get(category, 0) + 1
        else:
            table_info[table_name] = {
                'predicted_categories': [category],
                'category_counts': {category: 1}
            }

    # Process the information for each table
    for table_name in table_info:
        categories = table_info[table_name]['predicted_categories']
        category_counts = table_info[table_name]['category_counts']
        total_count = len(categories)
        
        # Calculate the percentage for each category
        percentage = {category: str(round(count/total_count * 100))+'%' for category, count in category_counts.items()}
        
        # Sort the categories based on the number of occurrences
        sorted_categories = sorted(category_counts, key=category_counts.get, reverse=True)
        
        # Extract the category with the maximum occurrence
        max_category = sorted_categories[0]
        # max_category = [1,2,3] 

        # Add the information for the current table to the result list
        # result = result.append({
        #     'TableName': table_name,
        #     'Category': max_category,
        #     'Categories': sorted_categories,
        #     'Percentage': [percentage[cat] for cat in sorted_categories]
        # }, ignore_index=True)

        categories = [f"{category}: {percentage}" for category, percentage in zip(sorted_categories, [percentage[cat] for cat in sorted_categories])]

        # temp = []
        # for i in categories:
        #     cat, per = i.split()

        #     if int()


        result = result.append({
            'TableName': table_name,
            'Category': max_category,
            'Categories': categories
        }, ignore_index=True)

    return result
        
    #result = result.style.format({'parcentage': '{:.2f}%'})
    #display(result)

    # whole_result = pd.DataFrame(columns=['ERPName', 'TableName', 'No. of Field', 'Total Records', 'True', 'False'])
    # erps_list = res['ERPName'].unique().tolist()
    # for e in erps_list:
    #     s_erp = res[res['ERPName'] == e]
    #     s_erp[['tablename', 'fieldname']] = s_erp['TableName'].str.split(' ', 1, expand=True)
    #     table_list = s_erp['tablename'].unique().tolist()
    #     for i in table_list:
    #         tbl = s_erp[s_erp['tablename'] == i]
    #         erp = tbl['ERPName'].unique()[0]
    #         table_name = i
    #         nooffiel = tbl['fieldname'].nunique()
    #         totalRecords = tbl.shape[0]
    #         right = tbl[tbl['result'] == 1]
    #         true = right.shape[0]
    #         wrong = tbl[tbl['result'] == 0]
    #         false = wrong.shape[0]
    #         row_dict = {'ERPName': erp, 'TableName': table_name, 'No. of Field': nooffiel, 'Total Records': totalRecords, 'True': true, 'False': false}
    #         whole_result = whole_result.append(row_dict, ignore_index=True)

    #return result
