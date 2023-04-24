# Polianna

n := number of data points <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_t | time-variant features: time stamp containing year, month, day | (n, 3) |
| x_s | virtual space-variant features: ordinally encoded categorical variables describing form of policy and treaty associated with an article | (n, 2) |
| x_st | space-time-variant features: article ID that can be mapped to additional data containing text data either in a tokenized format or in a single string of characters. | (n, 1) |
| y_st | labels: annotation of text from a total of 42 (plus "uncertain") possible policy design category tags for article-level sub-task. For the text-level sub-task, the variable is equal to x_st and can be mapped to a precise annotation of text spans including words, start index and end index | (n, 43) / (n, 1) |


| Additional data | Description | Format |
| --- | ----------- | ----------- |
| x_st | A mapping of article IDs (430 articles) to their text. This must be used to dynamically, or statically, expand x_st. These consist of 12-4'958 words. | (n, 12-4'958)|
| y_st | A mapping of article IDs (430 articles) to their precise annotations. This must be used to dynamically, or statically, expand y_st. These consist of 1-736 annotations which in total are 3-2'208 single values for labels. | (n, 3-2'208)|



