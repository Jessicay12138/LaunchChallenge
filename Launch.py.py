import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def prelimDataClean():
    """
    Implements a preliminary cleaning of the datasets by reading and aggregating them together,
    keeping only the relevant information 
    in order to get big picture overview of the database we have to work with

    Returns
    -------
    None.

    """
    ###Reading Raw Data
    dfList = []
    rawDf = pd.DataFrame()
    for i in range(1,14):
        thisdf = pd.read_csv("sales_contents_"+str(i)+".csv", low_memory = False)
        thisdf.dropna(how = "all", axis = 1, inplace = True)
        dfList.append(thisdf)
        
    #aggregate all of raw data into one dataframe, joining only shared columns
    rawDf = pd.concat(dfList, axis=0, ignore_index = False, join="inner")  
    
    ###Overview of Entire Database 
    print("First 5 rows of dataframe: ", rawDf.head(5))
    print("General info of dataframe: ", rawDf.info())
    print("Describe the dataframe: ", rawDf.describe())
    
    
    #Keep only the columns we are interested in 
    rawDf = rawDf[['catalog_number', "lot_sale_year", "lot_sale_month", 'lot_sale_day', 
                   "artist_name_1", "nationality_1", "object_type", 'transaction',
                   'price_amount_1','price_currency_1']]
    
    ###Overview of Interested Fields; asking some broad questions to narrow down research focus
    print("What are the nationalities of artist represented in this dataset? " , rawDf['nationality_1'].unique())
    print("What are the top 10 most popular artist nationalities? ", rawDf['nationality_1'].value_counts()[:10].index.tolist())
    print("Who are the top 10 most popular artists? ", rawDf['artist_name_1'].value_counts()[:10].index.tolist())
    print("How many unique 'transaction' data points are there?" ,len(pd.unique(rawDf['transaction'])))
    
    ###Clean the 'catalog number' column
    ### by extracting the country of transaction
    rawDf['catalog_number'] = rawDf['catalog_number'].str[:2]
    rawDf = rawDf[rawDf['catalog_number'].apply(lambda x: isinstance(x, str))] #convert to string
    catalog_dict = {'B-':'Belgium','Br':'Britain','N-':'Netherlands','F-':'France','D-':'Germany, Austria & Switzerland',
                        'SC':'Scandinavia'}
    rawDf = rawDf.replace({'catalog_number': catalog_dict})
    
    ###Clean the 'nationality' column
    ###by creating an "others" category of all data entries with ambigious or unknown nationalities
    nationality = rawDf['nationality_1']
    other_list = ["and", "NON-UNIQUE", "or", "New", "Unknown", "UNKNOWN", ";", "\?"]
    other_reg = '|'.join(other_list)
    other = nationality.str.contains(other_reg)
    rawDf['nationality_1'] = np.where(other, 'Other', nationality.str.capitalize()) 
    
    ### Clean the 'objet_type' column
    ### by translating all other language descriptions to English
    rawDf = rawDf[rawDf['object_type'].apply(lambda x: isinstance(x, str))] #convert to string
    object_dict = {"Ã‰mail": "Other", "Ã‰mail; Ã‰mail": "Other", "Ã‰mail; Miniature": "Other",
                        "Aquarell": "Watercolor", "Aquarelle": "Watercolor","Aquarelle;Aquarelle": "Watercolor",
                        "Dentelle": "Lace", "Dessin": "Drawing", "Dessin;Dessin": "Drawing", "Dessin;Dessin;Dessin": "Drawing",
                        "MÃ©daille": "Other", "GemÃ¤lde": "Other", "Meuble": "Marble", "MosaÃ¯que": "Mosaic",
                        "Marqueterie": "Marquetry", "Miniatur": "Miniature", "Miniature;Ã‰mail": "Other", 
                        "Miniature;Dessin": "Other","Miniature;Miniature": "Other","Mosaique": "Mosaic" ,
                        "Pastel;Pastel": "Pastel", "Pastell": "Pastel", "Peinture": "Painting",
                        "Peinture;Dessin": "Other", "Peinture;Dessin;Sculpture": "Other", "Peinture;Peinture": "Painting",
                        "Peinture;Sculpture": "Other", "Sculpture;Sculpture": "Sculpture", "Skulptur": "Sculpture",
                        "Tapisserie":"Tapestry", "Watercolor;Enamel":"Other", "Zeichnung":"Drawing" }
    rawDf = rawDf.replace({'object_type': object_dict})
    
    ### Clean the 'transaction' column
    ### by translating all other language descriptions to English
    rawDf = rawDf[rawDf['transaction'].apply(lambda x: isinstance(x, str))] #convert to string
    transaction_dict = {'Non Vendu':'Not Sold','Vendu':'Sold','Verkauft':'Sold','Unbekannt':'Unknown','Inconnue':'Unknown',
                        'Unverkauft':'Unknown'}
    rawDf = rawDf.replace({'transaction': transaction_dict})

    ###Save all the cleaned dataframe into a new csv file 
    ### so we can go straight into using it next time without re-cleaning everytime
    rawDf.to_csv("prelim_clean.csv")
    # print('end of prelim_data_clean')
    
def basic_stats():
    """
    Function that graphs the basic statistics we are interested in, excluding any that have to do with the price of the artwork. 
    Including: Country of transaction frequency distribution, artist nationality frequency distribution, transaction dstatus distribution
    distribution by year, month, and day of transaction, and distribution accounting for both month and day.
    Returns
    -------
    None.

    """
    # Read Data
    df = pd.read_csv("prelim_clean.csv", low_memory = False)
    plt.style.use('seaborn-deep')
    
    # ###Country of Transaction (Catelog_number) Frequency Distribution
    # df['catalog_number'] = df['catalog_number'].astype('category')
    # #Make an "other" category for graphing
    # nation = df['catalog_number'].value_counts().plot(kind = 'pie' )
    # nation.set_title("Distribution of Transactions by Country")
    # nation.set_ylabel(" ")
    
    # ###Nationality Frequency Distribution
    # df['nationality_1'] = df['nationality_1'].astype('category')
    # #Figure out what the top 8 nationalities are:
    # print("What are the top 8 most popular artist nationalities? ", df['nationality_1'].value_counts()[:8].index.tolist())
    # #Make an "other" category for graphing
    # top_8 = ["German", "Dutch", "Italian", "Flemish", "French", "Austrian", "Other", "British"]
    # df.loc[~df['nationality_1'].isin(top_8), "nationality_1"] = "Other"
    # nation = df['nationality_1'].value_counts().plot(kind = 'pie' )
    # nation.set_title("8 Most Popular Artist Nationalities in Getty Transactions")
    # nation.set_ylabel(" ")
    
    # ###Name Frequency Distribution
    # df['artist_name_1'] = df['nationality_1'].astype('category')
    # #Figure out what the top 8 nationalities are:
    # print("What are the top 8 most popular artist nationalities? ", df['nationality_1'].value_counts()[:8].index.tolist())
    # #Make an "other" category for graphing
    # top_8 = ["German", "Dutch", "Italian", "Flemish", "French", "Austrian", "Other", "British"]
    # df.loc[~df['nationality_1'].isin(top_8), "nationality_1"] = "Other"
    # nation = df['nationality_1'].value_counts().plot(kind = 'pie' )
    # nation.set_title("8 Most Popular Artist Nationalities in Getty Transactions")
    # nation.set_ylabel(" ")
    
    
    # ###Frequency of Transaction by Year
    # df['lot_sale_year'] = df['lot_sale_year'].astype(np.int)
    # plt.hist(df['lot_sale_year'], bins = 20, density = 1)
    # plt.xlabel('Year')
    # plt.ylabel('Number of Transactions')
    # plt.title('Frequency of Transaction by Year')
    # plt.show()
    
    # ###Frequency of Transaction by Month
    # df['lot_sale_month'] = df['lot_sale_month'].astype(np.int)
    # month_graph = df['lot_sale_month'].value_counts(sort = True).plot(kind='bar')
    # month_graph.set_xlabel('Month')
    # month_graph.set_ylabel('Number of Transactions')
    # month_graph.title.set_text("Frequency of Transaction by Most Popular Month")
    
    # # ###Frequency of Transaction by Day
    # df['lot_sale_day'] = df['lot_sale_day'].astype(np.int)
    # day_graph = df['lot_sale_day'].value_counts(sort = True).plot(kind='bar')
    # day_graph.set_xlabel('Day')
    # day_graph.set_ylabel('Number of Transactions')
    # day_graph.title.set_text("Frequency of Transaction by Most Popular Day")
    
    # ###Frequency of Transaction by both month and day
    # df['lot_sale_day'].hist(by=df['lot_sale_month'], bins = 5)
    # df.pivot(columns = 'lot_sale_month').lot_sale_day.plot(kind = 'hist', stacked = True)
    # plt.legend(title = "Month", bbox_to_anchor=(1.05, 1))
    # plt.xlabel("Day")
    # plt.ylabel("Number of Transactions")
    # plt.title(('Frequency of Transaction by Month and Day'))
    
    
    ###Frequency of Transaction Status
    # #Make an "other" category for graphing
    # viable_transactions = ["Unknown", "Sold", "Bought In"]
    # df.loc[~df['transaction'].isin(viable_transactions), "transaction"] = "Other"
    # transaction = df['transaction'].value_counts().plot(kind = 'pie' )
    # transaction.set_title("Transaction Status Distribution")
    # transaction.set_ylabel(" ")
    print('end of basic_stats')

def deepDataClean():
    """
    separated prelimDataClean and deepDataClean so that there are more datapoints for year, month and day sales 
    so the distributions and frequencies graphed in basic_stats() is more representative of the entire databse'''
    
    function to clean the price and currency column by converting all currencies into USD (in today's timing')

    Returns
    -------
    None.

    """
    df = pd.read_csv("prelim_clean.csv", low_memory = False)

    #clean the dataset more to include only rows of data with a price and currency rather than NaN
    df.dropna(subset = ['price_currency_1'], inplace = True)
    df.dropna(subset = ['price_amount_1'], inplace = True)
    
    #Drop any rows where the month sold was incorrectly entered in the price column
    month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = df[~df['price_amount_1'].str.contains('|'.join(month_list))]
    
    #drop any rows where a hyphen is entered in place of a decimal point, also converting the price column to be a numeric datatype so it's easier to work with later 
    hyphen = df['price_amount_1']
    hyphenated = hyphen.str.contains('-')
    df['price_amount_1'] = np.where(hyphenated, '.', hyphen)
    df['price_amount_1'] = pd.to_numeric(df['price_amount_1'], errors = 'coerce', downcast = 'float')
    df = df.dropna(subset = ['price_amount_1'])
    df = df.replace([])

    #Converting all prices to a common currency (USD), using fixed conversion rates to be manually changed below 
    pound_us = 1.09
    euro_us = 1.10
    df['one_price'] = np.where(df['price_currency_1']== "pound", df['price_amount_1'].apply(lambda x: x *pound_us), df['price_amount_1'].apply(lambda x: x *euro_us))
    
    # Drop any rows with NaN that were created as a result of the data cleaning
    df = df[df['lot_sale_year'].notna()]
    df = df[df['lot_sale_month'].notna()]
    df = df[df['lot_sale_day'].notna()]
    df = df[df['price_amount_1'].notna()]
    
    # Save to new data files
    df.to_csv("deep_clean.csv")

def in_depth_stats():
    
    """
    Uses the cleaned datafiles from deepDataClean() to graph and regress relationships between 
    the date and price, using OLS simple linear regression and multi-linear regression
    
    """
     #importing graph extensions here so it's faster to run other functions when we don't need to graph
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    
    # ###Reading information from previous functions
    df = pd.read_csv("deep_clean.csv", low_memory = False)
    plt.style.use('seaborn-deep')
    
    # ###Price as Y Variable
    df = df[df['one_price'] < 10000]# Dropping outliers
    price = df["one_price"]
    
    
    # ### Graphing relationship between year and price 
    # plt.figure(figsize=(30, 15))
    # year = df['lot_sale_year']
    # plt.scatter(year, price)
    # m, b = np.polyfit(year, price, 1)
    # plt.plot(year, m*year+b, color = "red")
    # plt.xlabel("Year", fontsize = 25)
    # plt.ylabel("Price of Transaction in USD", fontsize = 25)
    # plt.title(('Price of Artwork Throughout the Years'), fontsize = 40)
    
    # ### Graphing relationship between month and price 
    # plt.figure(figsize=(30, 15))
    # month = df['lot_sale_month']
    # plt.scatter(month, price)
    # m, b = np.polyfit(month, price, 1)
    # plt.plot(month, m*month+b, color = "red")
    # plt.xlabel("Month", fontsize = 25)
    # plt.ylabel("Price of Transaction in USD", fontsize = 25)
    # plt.title(('Price of Artwork in Months'), fontsize = 40)
   
    ### Graphing relationship between day and price 
    # plt.figure(figsize=(30, 15))
    # day = df['lot_sale_day']
    # plt.scatter(day, price)
    # m, b = np.polyfit(day, price, 1)
    # plt.plot(day, m*day+b, color = "red")
    # plt.xlabel("Dayof Month", fontsize = 25)
    # plt.ylabel("Price of Transaction in USD", fontsize = 25)
    # plt.title(('Price of Artwork in Days of the Month'), fontsize = 40)

    #################
    ###Regressions###
    #################
    ####Using 2 methods (1. Simple Linear Regression (with OLS) and 2. Multi Linear Regressions): 
        # ###Method #1. OLS regression of price on years 
    years = df['lot_sale_year']
    months = df['lot_sale_month']
    days = df['lot_sale_day']
     
    # ###Effect of Year on Price
    # years = sm.add_constant(years)
    # model = sm.OLS(price, years).fit()
    # print(model.summary())
  
    # ###Effect of Month on Price
    # months = sm.add_constant(months)
    # model = sm.OLS(price, months).fit()
    # print(model.summary())
  
    # ###Effect of Day on Price
    # days = sm.add_constant(days)
    # model = sm.OLS(price, days).fit()
    # print(model.summary())
  
    # ###Multivariate Effect of (Year, Month and Day) on Price
    # #Multivariate regression of Price vs Effects of year, month and day
    # dates = df[['lot_sale_year','lot_sale_month','lot_sale_day' ]]
    # model = sm.OLS(price, dates).fit()
    # print(model.summary())
  
        # ###Method #2. Multiple linear regression, using year as one example as my laptop ran out of memory
    # year = years.values.reshape(-1,1) 
    # month = years.values.reshape(-1,1) 
    # day = years.values.reshape(-1,1) 
    
    # ###Effect of Year on Price
    # year_x_train, year_x_test, prices_train, prices_test = train_test_split(year, price, test_size = 0.2, random_state = 10)
    # year_lr = LinearRegression()
    # year_lr.fit(year_x_train, prices_train)
    #     ##Years Training Set
    # plt1 = plt.figure(figsize = (8, 11))
    # plt.scatter(year_x_train, prices_train)
    # plt.plot(year_x_train, year_lr.predict(year_x_train), color = 'red')
    # plt.title("Predicted Price vs Year, Training Set", fontweight = 'bold' )
    # plt.xlabel('Year')
    # plt.ylabel('Predicted Price')
    # plt.show()
    
    #     ## Years Testing Set
    # plt1 = plt.figure(figsize = (8, 11))
    # plt.scatter(year_x_test, prices_test)
    # plt.plot(year_x_test, year_lr.predict(year_x_test), color = 'red')
    # plt.title("Predicted Price vs Year, Testing Set", fontweight = 'bold' )
    # plt.xlabel('Year')
    # plt.ylabel('Predicted Price')
    # plt.show()
    
        # Print out the key statistics
    # r_sq = year_lr.score(year_x_test, prices_test)
    # print(f"R squared: {r_sq}")
    # print(f"intercept: {year_lr.intercept_}")
    # print(f"coefficients: {year_lr.coef_}")
    

def main():
    # prelimDataClean()
    # basic_stats()
    # deepDataClean()
    # in_depth_stats()

main()