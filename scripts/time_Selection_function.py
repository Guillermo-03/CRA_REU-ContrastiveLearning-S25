import pandas as pd


#x is the user input for time. 
def time_Selecton_function(x):
    df = pd.read_csv('Cleaned_CatTrack.csv', dtype={'Time(sec)': 'float64'})
    # print(df.info())
    while True:
        try: 
            user_time = float(x)
            break
        except ValueError:
            x = input('Please input a numerical value: ')


        #Closest match for time give within CSV 
    df['diff'] = abs(df['Time(sec)'] - user_time)
    closest_row = df.loc[df['diff'].idxmin()]
    print(f"Closest time: {closest_row['Time(sec)']},\nLatitude: {closest_row['Latitude']},\nLongitude: {closest_row['Longitude']}, \nDatetime: {closest_row['Date(GMT)']}")
   

    
       

time_Selecton_function(input('Enter a time(sec) you would like to :'))


