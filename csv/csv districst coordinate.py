


# District latitude and longitude data
district_geodata = {
   
    "Achham": (29.041, 81.301),
    "Arghakhanchi": (27.999, 83.151),
    "Baglung": (28.273, 83.589),
    "Baitadi": (29.516, 80.542),
    "Bajhang": (29.637, 81.202),
    "Bajura": (29.418, 81.308),
    "Banke": (28.050, 81.667),
    "Bara": (27.000, 85.000),
    "Bardiya": (28.280, 81.342),
    "Bhojpur": (27.174, 87.050),
    "Bhaktapur":(27.672, 85.429),
    "Chitwan": (27.529, 84.354),
    "Dadeldhura": (29.222, 80.567),
    "Dailekh": (28.848, 81.708),
    "Dang": (28.040, 82.301),
    "Darchula": (29.851, 80.548),
    "Dhading": (27.897, 84.902),
    "Dhankuta": (26.983, 87.333),
    "Dhanusha": (26.978, 86.002),
    "Dolakha": (27.679, 86.283),
    "Dolpa": (28.983, 83.733),
    "Doti": (29.260, 80.900),
    "Gorkha": (28.000, 84.632),
    "Gulmi": (28.073, 83.268),
    "Humla": (29.999, 81.851),
    "Ilam": (26.911, 87.930),
    "Jajarkot": (28.700, 82.212),
    "Jhapa": (26.533, 88.083),
    "Jumla": (29.275, 82.184),
    "Kailali": (28.680, 80.586),
    "Kalikot": (29.138, 81.600),
    "Kanchanpur": (28.970, 80.183),
    "Kapilvastu": (27.583, 83.050),
    "Kaski": (28.209, 83.985),
    "Kathmandu": (27.700, 85.333),
    "Kavrepalanchok": (27.533, 85.533),
    "Khotang": (27.186, 86.795),
    "Lalitpur": (27.667, 85.317),
    "Lamjung": (28.283, 84.283),
    "Mahottari": (26.750, 85.933),
    "Makwanpur": (27.400, 85.067),
    "Manang": (28.642, 84.023),
    "Morang": (26.447, 87.413),
    "Mugu": (29.666, 82.259),
    "Mustang": (28.976, 83.881),
    "Myagdi": (28.333, 83.567),
    "Nawalpur": (27.583, 83.800),
    "Nuwakot": (27.891, 85.194),
    "Okhaldhunga": (27.316, 86.500),
    "Palpa": (27.883, 83.550),
    "Panchthar": (27.123, 87.892),
    "Parbat": (28.237, 83.601),
    "Parsa": (27.000, 84.833),
    "Pyuthan": (28.083, 82.833),
    "Ramechhap": (27.315, 86.083),
    "Rasuwa": (28.117, 85.283),
    "Rautahat": (26.783, 85.267),
    "Rolpa": (28.333, 82.850),
    "Rukum East": (28.704, 82.468),
    "Rukum West": (28.654, 82.398),
    "Rupandehi": (27.500, 83.417),
    "Salyan": (28.383, 82.183),
    "Sankhuwasabha": (27.623, 87.294),
    "Saptari": (26.607, 86.647),
    "Sarlahi": (26.963, 85.574),
    "Sindhuli": (27.250, 85.900),
    "Sindhupalchok": (27.950, 85.717),
    "Siraha": (26.654, 86.211),
    "Solukhumbu": (27.536, 86.606),
    "Sunsari": (26.629, 87.183),
    "Surkhet": (28.600, 81.633),
    "Syangja": (28.167, 83.833),
    "Tanahun": (27.983, 84.267),
    "Taplejung": (27.358, 87.670),
    "Terhathum": (27.132, 87.557),
    "Udayapur": (26.806, 86.686)
}
import pandas as pd
import os
from tkinter import Tk, filedialog

def ask_save_location():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select Directory to Save CSV File")
    return folder_selected

# Create a DataFrame from the district latitude and longitude data
df = pd.DataFrame(list(district_geodata.items()), columns=['District', 'Latitude_Longitude'])

# Split the 'Latitude_Longitude' column into separate 'Latitude' and 'Longitude' columns
df[['Latitude', 'Longitude']] = pd.DataFrame(df['Latitude_Longitude'].tolist(), index=df.index)

# Drop the original 'Latitude_Longitude' column
df.drop(columns=['Latitude_Longitude'], inplace=True)

# Ask the user where to save the CSV file
save_directory = ask_save_location()

if save_directory:
    file_path = os.path.join(save_directory, 'district_geodata.csv')
    df.to_csv(file_path, index=False)
    print(f"CSV file 'district_geodata.csv' has been created successfully in {save_directory}.")
else:
    print("No directory selected. Operation cancelled.")
