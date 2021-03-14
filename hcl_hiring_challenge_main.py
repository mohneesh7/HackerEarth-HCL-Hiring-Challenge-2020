# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
This program is written for HCL ML hiring challenge on HackerEarth
This should be placed in the HCL hiring challenge folder to execute
correctly. All the code credits goes to Mohneesh except wherein
mentioned otherwise.There will be some automatically commented out
ipython magic because I did this in colab. This has a pylint
rating of 8.94, and a submission score of 147.
Hope I get hired! (ツ)
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /
# %cd '/content/drive/My Drive/HCL hiring challenge'
# %cd HCL\ ML\ Challenge

import re
import os
import decimal
import json
import html
import pandas as pd
import numpy as np
from tqdm import tqdm

# List all the files
os.chdir(f'HCL ML Challenge Dataset')
LIST_DIR = os.listdir()
print(LIST_DIR[:5]) #print first five records

# save the names of all files in a list called names.
NAMES = [i.split('.')[0] for i in LIST_DIR]
print(NAMES[:5]) #print first five records

def check_para(each_line):
    """
    This function basically checks for numbers with paranthesis and replaces
    it with the  negative of the number as perthe requirements of the challenge.
    e.g : (11345) ==> -11345
    Credit: http://tiny.cc/whyhmz
    """
    num = 0
    # we don't need empty values, we only need to check for numbers.
    if each_line == 'nan':
        return 'nan'
    # this pattern matches numbers in brackets.
    pat = re.compile(r'(\(?(\d+)\)?)')
    this = pat.search(each_line)
    if this is not None:
        num = decimal.Decimal(this.groups()[1])
    if this.groups()[0].startswith('('):
        num *= -1
    return str(int(num))


# this is the main for loop that extracts the values from the 500 files.
values = []
for i in tqdm(NAMES[:], position=0, leave=True):

    # Create an I/O object and read the data from the file.

    temp = open('{}.txt'.format(i), 'r')
    temp1 = temp.readlines()
    temp.close()

    # close the file object to avoid any exceptions.

    # Create lists to that will be used to create  a dataframe.

    l = []
    l1 = []
    l2 = []
    l3 = []

    # remove whitespaces
    for j in temp1:
        l.append(j.strip())

    # split lines based on the gap between them.
    for k in l:
        l1.append(k.split('  '))

    # filter out the None values and append
    for m in l1:
        l2.append(filter(None, m))

    # create lists of the seperated columns.
    for n in l2:
        l3.append(list(n))

    # Remove the header of the files which aren't required for this task.
    l3.remove(l3[0])

    # we don't need the  unnecessary satements part below the balance sheet.
    # (this number is set after fine-tuning)
    l3 = l3[:20]


    try:
    # add the header for the first column if it isn't present.
        if l3[0][0] != 'Notes':
            l3[0].insert(0, 'nan')

        # Try : Except block because some  files have 2 columns for years and some have only one.
        try:
            temp = pd.DataFrame(l3, columns=['zero', 'one', 'two'])
        except Exception:
            temp = pd.DataFrame(l3, columns=['zero', 'one'])

        # Replace all empty cells with nan as the evaluator dosen't accept NaN
        temp.replace(to_replace=np.nan, value='nan', inplace=True)

        # Most of the files 'STATEMENTS' OR '®' as ending to the indormation that we require.
        # So we remove the extra part.(Better done using try : except block)
        try:
            statement_index = int(temp[temp['zero'] == 'STATEMENTS'].index.values)
            temp = temp[:statement_index]
        except Exception:
            statement_index = int(temp[temp['zero'] == '®'].index.values[0])
            temp = temp[:statement_index]

        # Choosing what column to drop when we have 2 columns with 2019 and other years.
        our_req = 2019
        year = ''
        try:
            if str(our_req) == str(temp['one'].iloc[0]):
                year = 'two'
            elif str(our_req) == str(temp['two'].iloc[0]):
                year = 'one'

            # Remove all whitespaces from the cells of the dataframe.
            temp = temp.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            # Convert/Substitute all the numbers with commas to non-comma seperated numbers and
            # remove brackets and place negative symbol, as per the requirement.
            temp[temp.columns[1]] = temp[temp.columns[1]].replace(to_replace=r'[^]nan[^\(?\d\.\)?$]', value='', regex=True)
            temp[temp.columns[1]] = temp[temp.columns[1]].apply(check_para)

            # Drop the column that we don't require
            temp.drop(year, axis=1, inplace=True)

            # Create the dictionary format that the submission is to be done.
            temp_dict = dict(zip(temp['zero'][2:], temp[temp.columns[1]][2:]))

        except Exception:
            # This block is executed when there is only one column for year in our data and
            # it has 2019 in it. And makes a dictionary.
            if str(our_req) == str(temp['one'].iloc[0]):
                temp_dict = dict(zip(temp['zero'][2:], temp[temp.columns[1]][2:]))

            else:
            # This block will be executed when the data has only one column but dosen't
            # have 2019 in it. The values are replaces with nan as perthe requirements.
                temp_dict = dict(zip(temp['zero'][2:], temp[temp.columns[1]][2:]))
                temp_dict = dict.fromkeys(temp_dict, 'nan')

        temp_dict = str(temp_dict)
        values.append(temp_dict)

    except Exception:
        # This block takes in some exception in files like '-' in the data and such (total 65 cases)
        values.append('nan')
        continue

# Count all the nan's that came from the exception.
print(values.count('nan'))

print(values[:10])

# replace utf encoded symbols to html escape codes

for j in tqdm(range(len(values)), position=0, leave=True):

    if values[j] == 'nan':
        continue

    else:
        # replace all the utf-encoded currency symbol with its equivalent
        # HTML escape code as per the requirements.
        for i in values[j]:
            if i in ['$', '®', '<', '>', '¢', '£', '¥', '€', '§', '©', '₹']:
                values[j] = values[j].replace(i, html.escape(i).encode('ascii', 'xmlcharrefreplace').decode('utf-8')[:-1])

# this code snippet uses json.dumps() to get the required format for submission.
values1 = []
for i in values:
    if i == 'nan':
        values1.append('nan')
    else:
        values1.append(json.dumps(eval(i)))

# Create the submission dataframe and give appropriate headers
# (the second column header must be given a space b/w extracted and
# values manually in the csv file).
SUB = pd.DataFrame(dict(Filename=NAMES, ExtractedValues=values1))

SUB.to_csv('Results.csv', index=False)

# ignore this is just code to download the csv file from colab.
#from google.colab import files
#files.download('eleventh_formatted.csv')
