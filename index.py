#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Toronto cities in the Greater Toronto Area (GTA) over the past five years for an upcoming report.


# In[ ]:


mean home prices 2023 time period


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import random 
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px


# In[ ]:


data of privious 10 years 


# In[76]:


#data of house
# major cities of gta
data_house_price = {
    'Years': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'toronto': [600000,612000,624000, 647000,657000,657000, 850000, 1140000, 1300000,1500000],
    'Vaughan': [550000, 570000, 590000, 610000, 720000, 760000, 800000, 920000, 950000, 1200000],
    'North York': [500000, 512000, 490000, 547000, 557000, 557000, 650000, 850000, 1150000, 1400000],
    'Mississauga': [500000, 520000, 530000, 550000, 570000, 600000, 620000, 640000, 670000, 1200000],
    'Markham': [700000, 720000, 750000, 770000, 790000, 820000, 850000, 880000, 920000, 850000], 
    'brampton': [400000,412000,424000, 447000,457000,457000, 650000, 714000, 830000,1000000],
    'Scarborough': [600000, 612000, 624000, 647000, 657000, 657000, 850000, 940000, 1000000, 1200000]
    
}
df = pd.DataFrame(data)
print(df)


# In[77]:


#house prices
prices = [1500000, 1200000, 1200000, 900000, 750000, 900000, 500000, 750000, 1200000, 1400000, 700000, 1250000]
# cities in GTA 
cities = ['Toronto', 'Mississauga', 'Scarborough', 'Markham', 'Brampton', 'Malton', 'Caledon', 'Ajax', 'North York', 'York', 'Georgetown', "vaughan"]
#bar chart
plt.bar( cities, prices )
# Add titles and axis labels
plt.title('average House Prices in Greater Toronto Area in Year 2023')
plt.xlabel('Cities')
plt.ylabel('Price i millions(CAD$)')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


Toroto has the highest average property price,
followed by York, Mississauga, Scarborough, and Vaughan, 
and Caledon and  the lowest average price.


# In[ ]:


influence on the housing market


# In[17]:


#house prices from in last 10 years in Toronto
prices_10years = [600000,612000,624000, 647000,657000,657000, 850000, 1140000, 1300000,1500000]
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
#line graph 
plt.plot( years, prices_10years )
# Add titles and labels
plt.title('rising price from last 10 year')
plt.xlabel('Toronto')
plt.ylabel('price in millions')
plt.show()


# In[25]:


#house prices from in last 10 years in Brampton
prices_10years_brampton = [400000,412000,424000, 447000,457000,457000, 650000, 714000, 830000,1000000]
#line graph 
plt.plot( years, prices_10years_brampton )
# Add titles and labels
plt.title('rising price from last 10 year')
plt.xlabel('Brampton')
plt.ylabel('price in millions')
plt.show()


# In[ ]:


During the COVID-19 epidemic, property values in both Toronto and the Brampton surged. 
This tendency can be ascribed to increasing demand for larger homes and cheap mortgage rates, 
emphasizing the pandemic's influence on the housing market.


# In[57]:


# House prices in missisaugga, markham, vaughan, scarbrough and york
prices_10years_scarbrough = [600000, 612000, 624000, 647000, 657000, 657000, 850000, 940000, 1000000, 1200000]
prices_10years_york = [500000, 512000, 490000, 547000, 557000, 557000, 650000, 850000,1150000,1400000]
prices_10years_mississauga = [500000, 520000, 530000, 550000, 570000, 600000, 620000, 640000, 670000, 1200000]
prices_10years_markham = [700000, 720000, 750000, 770000, 790000, 820000, 850000, 880000, 920000, 850000]
prices_10years_vaughan = [650000, 670000, 690000, 710000, 730000, 760000, 790000, 820000, 850000, 1200000]

years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

plt.plot(years, prices_10years_scarbrough, label='scarbrough')
plt.plot(years, prices_10years_york, label      ='york and north york')
plt.plot(years, prices_10years_mississauga, label='Mississauga')
plt.plot(years, prices_10years_markham, label    ='Markham')
plt.plot(years, prices_10years_vaughan, label    ='Vaughan')

plt.title('House Prices Comparison')
plt.xlabel('Years')
plt.ylabel('Price in Millions')
plt.legend()
plt.show()


# In[ ]:


similar trend can be seen in other major cities in toronto


# In[ ]:


pridiction for next 10 years 


# In[59]:


# data frame pd
df = pd.DataFrame(data)
# (Years) andvariable (toronto)
X = df['Years'].values.reshape(-1, 1)
y = df['toronto'].values

# Create a Linear Regression model
model = LinearRegression()
model.fit(X, y)
# Predict for the next 5 years
future_years = [2024, 2025, 2026, 2027, 2028]
predicted_prices = model.predict(np.array(future_years).reshape(-1, 1))

# Print the predicted prices
for year, price in zip(future_years, predicted_prices):
    print(f"Year: {year}, Predicted Price: {price}")


# In[ ]:


Based on existing data and research, it is expected that
Toronto's home prices will continue to rise steadily in the next years.
For the next five years, the following forecasts have been made:


# In[69]:


# future years in toronto
future_5year = [2024, 2025, 2026, 2027, 2028]
# house price  in toronto around figures 
toronto_house_price = [1400000, 1490000, 1590000, 1688000, 1785000]
df_future = pd.DataFrame({'Years': future_5year, 'Toronto': toronto_house_price})
fig = px.line(df_future, x='Years', y='Toronto', title='Toronto Housing Prices Prediction for Next 5 Years')
fig.show()


# In[ ]:




