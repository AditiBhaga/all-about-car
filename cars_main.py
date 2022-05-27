#importing the required libraries
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
#setting page configuration
st.set_page_config(page_title='carsapp',page_icon='🚗')
st.image('imagelogo.png')
ti='<p style ="color:red;font-size:50px;text-align:center">Bonjour(or I must say Hello)</p>'
st.markdown(ti,unsafe_allow_html=True)
hide_menu_style = """
                <style>
                #MainMenu {visibility: hidden; }
                footer {visibility: hidden;}
                </style>
                """

st.markdown(hide_menu_style,unsafe_allow_html=True)
#reading the dataset
cars=pd.read_excel("cars_final1.xlsx")
cars.head()
#exploring the dataset

cars_data=cars[['Unnamed: 0','Make', 'Model', 'Ex-Showroom_Price',
       'Displacement',  'Drivetrain', 'Fuel_Type', 'Height',
       'Length', 'Width', 'Body_Type', 'Doors', 'ARAI_Certified_Mileage',
        'Gears', 'Front_Brakes', 'Rear_Brakes',
       'Front_Suspension', 'Rear_Suspension', 'Power_Steering', 'Power_Seats', 'Keyless_Entry', 'Power', 'Torque',
       'Odometer', 'Speedometer',
       'Seating_Capacity', 'Seats_Material', 'Type', 'Start_/_Stop_Button',
       'Audiosystem',
       'Basic_Warranty', 'Bluetooth', 'Child_Safety_Locks', 'Extended_Warranty', 'FM_Radio',
        'Handbrake', 
       'Ventilation_System',
        'Drive_Modes',  'Lane_Watch_Camera/_Side_Mirror_Camera',
       'ABS_(Anti-lock_Braking_System)', 'Airbags',
        'Compression_Ratio',
       'Parking_Assistance',
       'Infotainment_Screen', 'Navigation_System',
       'Rain_Sensing_Wipers', 'Leather_Wrapped_Steering',
        ]]
# Cleaning the dataset
cars_data.isna().sum()
cars_data.describe()
#filling Null values with not known
cars_data["Make"].fillna("Not known", inplace = True) 
cars_data.isna().sum()
#filling null values with mean 
mean_val1=cars_data['Displacement'].mean()
cars_data['Displacement'].fillna(value=mean_val1,inplace=True)
mean_val2=cars_data['Height'].mean()
cars_data['Height'].fillna(value=mean_val2,inplace=True)
mean_val3=cars_data['Width'].mean()
cars_data['Width'].fillna(value=mean_val3,inplace=True)
mean_val4=cars_data['ARAI_Certified_Mileage'].mean()
cars_data['ARAI_Certified_Mileage'].fillna(value=mean_val4,inplace=True)
mean_val5=cars_data['Doors'].mean()
cars_data['Doors'].fillna(value=mean_val5,inplace=True)
mean_val6=cars_data['Seating_Capacity'].mean()
cars_data['Seating_Capacity'].fillna(value=mean_val6,inplace=True)
#removing rows with are inpractical values and less density
cars_data.drop(cars_data[cars_data['ARAI_Certified_Mileage']>= 100].index,inplace=True)
cars_data.drop(cars_data[cars_data['Ex-Showroom_Price']>=10000000].index,inplace=True)
cars_data.isna().sum()
cars_data.fillna("Not known", inplace = True)
cars_data.isna().sum()
#creating the sidebar for the app
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(None, ["Home",  "Charts","Prediction for Price","About"], 
    icons=['house', "list-task","activity", "text-center"], 
    menu_icon="cast", default_index=0, 
    orientation="Horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "pink"},
    }
)

if selected=="Home":
    st.header("Welcome to All about cars!!!")
    st.subheader('You can select charts for knowing about different specifications of cars')
    st.subheader('You can select prediction to know the price of a car with your given inputs.')
    st.write('Enjoy!!!')
if selected=="Charts":
    st.title(f"You have selected {selected}")
    st.header('Understand popular specification of the car with charts')
    st.write('Select the type of graph you want to see')
    #radio button to choose the different type of plot
    op=st.radio('plot type:' ,['Bar','Scatter'])
    if op=='Bar':
        st.write('The tallest Bar will let you know what most people like')
        selected_column1=st.selectbox('select the field one',sorted(list(cars_data.columns)))
        def countPlot():
            figure=plt.figure(figsize=(5,5))
            figure=px.bar(cars_data,y=selected_column1,color_discrete_sequence=['red']*3)
            st.plotly_chart(figure)
        countPlot()
    if op=='Scatter':
        selected_column2=st.selectbox('select the field one',sorted(list(cars_data.select_dtypes(exclude='object'))))
        selected_column3=st.selectbox('select the field two',sorted(list(cars_data.select_dtypes(exclude='object'))))
        cars_data_filter=[selected_column2,selected_column3]
        def ScatterPlot():
            fig1=plt.figure(figsize=(10,10))
            plt.scatter(x=selected_column2,y=selected_column3,data=cars_data,cmap='viridis',c='red')
            st.pyplot(fig1)
        ScatterPlot()
if selected=="About":
    st.title(f"You have selected {selected}")
    st.write('Here you can know everything and anything about cars with simple and easy to understandable graphs')
    st.write('We also predict the price of the car with your given inputs')
    st._transparent_write('''One can use this app as a review to know about the best choices of the population,
                           We all use apps and reviews of other people for anything we buy . This is the ultimate place fot it.''')
    
if selected=="Prediction for Price":
    st.title(f"{selected} \n Here we predict the price") 
    st.sidebar.header('User Input')
    numerical_cols=cars_data.select_dtypes(exclude=['object']).columns
    X = cars_data[['Height','Width','Displacement','Length','Doors','Seating_Capacity']].values
    Y = cars_data['Ex-Showroom_Price'].values
    #function to take inputs
    def user_input_features():
        Height = st.sidebar.slider('Height', 2, 2670, 1000)
        Width = st.sidebar.slider('Width', 2, 2226, 1000)
        Displacement = st.sidebar.slider('Displacement', 74, 8000, 5000)
        Length = st.sidebar.slider('Length', 5, 6092, 1000)
        Doors = st.sidebar.slider('Doors', 2, 5, 3)
        Seating_Capacity = st.sidebar.slider('Seating_Capacity', 2, 16, 5)
        data = {'Height': Height,
                'Width': Width,
                'Displacement': Displacement,
                'Length': Length,
                'Doors': Doors,
                'Seating_Capacity': Seating_Capacity}
        features = pd.DataFrame(data, index=[0])
        return features

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    df = user_input_features()
    # Main Panel
    
    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')
    
    
    # Build Regression Model
    model = RandomForestRegressor(n_estimators = 6,criterion = 'mse',random_state = 20,n_jobs = -1)
    model.fit(X,Y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)
    st.header('Prediction of Price')
    st.write(prediction)
    st.write('---')
    
    
    st.write("Thank you")
    
    
