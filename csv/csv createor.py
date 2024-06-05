import pandas as pd
from tkinter import Tk, filedialog
import os

# Dictionary of connected districts for Bagmati Province
connected_districts = {
    "Achham": ["Bajura", "Doti", "Surkhet", "Dailekh"],
    "Arghakhanchi": ["Gulmi", "Palpa", "Rupandehi", "Kapilvastu", "Pyuthan"],
    "Baglung": ["Parbat", "Myagdi", "Gulmi"],
    "Baitadi": ["Dadeldhura", "Darchula", "Bajhang"],
    "Bajhang": ["Baitadi", "Doti", "Bajura"],
    "Bajura": ["Bajhang", "Achham", "Kalikot"],
    "Banke": ["Bardiya", "Dang", "Surkhet"],
    "Bara": ["Parsa", "Rautahat", "Makwanpur"],
    "Bardiya": ["Banke", "Surkhet", "Kailali"],
    "Bhojpur": ["Khotang", "Sankhuwasabha", "Terhathum"],
    "Chitwan": ["Makwanpur", "Nawalpur", "Parsa"],
    "Dadeldhura": ["Baitadi", "Darchula", "Kanchanpur"],
    "Dailekh": ["Surkhet", "Jajarkot", "Kalikot", "Achham"],
    "Dang": ["Rolpa", "Pyuthan", "Salyan", "Banke"],
    "Darchula": ["Baitadi", "Bajhang"],
    "Dhading": ["Kathmandu", "Nuwakot", "Gorkha", "Makwanpur", "Chitwan"],
    "Dhankuta": ["Sankhuwasabha", "Terhathum", "Bhojpur", "Morang"],
    "Dhanusha": ["Sarlahi", "Mahottari", "Sindhuli", "Udayapur"],
    "Dolakha": ["Sindhupalchok", "Ramechhap"],
    "Dolpa": ["Mugu", "Jumla", "Rukum West"],
    "Doti": ["Achham", "Bajhang", "Kailali"],
    "Gorkha": ["Dhading", "Lamjung", "Syangja", "Tanahun", "Chitwan"],
    "Gulmi": ["Baglung", "Arghakhanchi", "Palpa", "Pyuthan", "Syangja"],
    "Humla": ["Mugu", "Jumla"],
    "Ilam": ["Panchthar", "Taplejung", "Jhapa", "Morang", "Dhankuta"],
    "Jajarkot": ["Dailekh", "Rukum West", "Salyan"],
    "Jhapa": ["Ilam", "Morang", "Sunsari"],
    "Jumla": ["Mugu", "Dolpa", "Jajarkot"],
    "Kailali": ["Doti", "Bardiya", "Kanchanpur", "Achham"],
    "Kalikot": ["Jumla", "Dailekh", "Jajarkot"],
    "Kanchanpur": ["Kailali", "Dadeldhura"],
    "Kapilvastu": ["Arghakhanchi", "Rupandehi"],
    "Kaski": ["Lamjung", "Manang", "Myagdi", "Parbat", "Syangja", "Tanahun"],
    "Kathmandu": ["Lalitpur", "Bhaktapur", "Nuwakot", "Dhading"],
    "Kavrepalanchok": ["Bhaktapur", "Lalitpur", "Ramechhap"],
    "Khotang": ["Okhaldhunga", "Udayapur", "Sankhuwasabha", "Solukhumbu", "Bhojpur"],
    "Lalitpur": ["Kathmandu", "Bhaktapur", "Makwanpur"],
    "Lamjung": ["Gorkha", "Kaski", "Manang", "Tanahun"],
    "Mahottari": ["Sarlahi", "Dhanusha", "Sindhuli", "Siraha"],
    "Makwanpur": ["Lalitpur", "Chitwan", "Sindhuli", "Bara"],
    "Manang": ["Kaski", "Lamjung", "Mustang"],
    "Morang": ["Ilam", "Jhapa", "Sunsari", "Dhankuta"],
    "Mugu": ["Jumla", "Dolpa", "Humla"],
    "Mustang": ["Manang", "Myagdi"],
    "Myagdi": ["Baglung", "Mustang", "Parbat"],
    "Nawalpur": ["Chitwan", "Tanahun"],
    "Nuwakot": ["Kathmandu", "Dhading", "Rasuwa"],
    "Okhaldhunga": ["Khotang", "Udayapur", "Solukhumbu"],
    "Palpa": ["Gulmi", "Rupandehi", "Syangja", "Tanahun"],
    "Panchthar": ["Ilam", "Taplejung", "Terhathum"],
    "Parbat": ["Baglung", "Myagdi", "Kaski", "Syangja"],
    "Parsa": ["Chitwan", "Makwanpur", "Bara"],
    "Pyuthan": ["Arghakhanchi", "Gulmi", "Rolpa", "Rukum East"],
    "Ramechhap": ["Kavrepalanchok", "Dolakha", "Sindhuli"],
    "Rasuwa": ["Nuwakot", "Sindhupalchok"],
    "Rautahat": ["Bara", "Sarlahi", "Sindhuli"],
    "Rolpa": ["Rukum East", "Salyan", "Pyuthan", "Dang"],
    "Rukum East": ["Rolpa", "Pyuthan", "Rukum West"],
    "Rukum West": ["Jajarkot", "Salyan", "Rukum East", "Dolpa"],
    "Rupandehi": ["Kapilvastu", "Palpa", "Nawalpur"],
    "Salyan": ["Rukum West", "Jajarkot", "Rolpa", "Dang", "Banke"],
    "Sankhuwasabha": ["Bhojpur", "Solukhumbu", "Taplejung", "Terhathum", "Dhankuta"],
    "Saptari": ["Siraha", "Udayapur"],
    "Sarlahi": ["Mahottari", "Rautahat", "Sindhuli"],
    "Sindhuli": ["Ramechhap", "Makwanpur", "Udayapur", "Sarlahi", "Dhanusha", "Rautahat"],
    "Sindhupalchok": ["Rasuwa", "Dolakha"],
    "Siraha": ["Saptari", "Udayapur", "Mahottari"],
    "Solukhumbu": ["Khotang", "Sankhuwasabha", "Okhaldhunga"],
    "Sunsari": ["Morang", "Jhapa", "Udayapur"],
    "Surkhet": ["Banke", "Dailekh", "Jajarkot", "Bardiya"],
    "Syangja": ["Palpa", "Tanahun", "Kaski", "Gulmi"],
    "Tanahun": ["Kaski", "Lamjung", "Gorkha", "Syangja", "Palpa"],
    "Taplejung": ["Panchthar", "Sankhuwasabha", "Ilam"],
    "Terhathum": ["Sankhuwasabha", "Dhankuta", "Bhojpur", "Panchthar"],
    "Udayapur": ["Saptari", "Siraha", "Sindhuli", "Khotang", "Bhojpur"]
}


# Function to generate dummy traffic data
def generate_dummy_traffic_data(from_district, to_districts):
    data = {
        "To District": to_districts,
        "Incoming Traffic": [int(1000 * (1 + i / 10)) for i in range(len(to_districts))],
        "Outgoing Traffic": [int(900 * (1 + i / 10)) for i in range(len(to_districts))]
    }
    return pd.DataFrame(data)

# Ask user for directory to save CSV files
def ask_save_location():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select Directory to Save CSV Files")
    return folder_selected

# Get the directory from the user
save_directory = ask_save_location()

# Generate and save CSV files for each district
if save_directory:
    for district, connections in connected_districts.items():
        df = generate_dummy_traffic_data(district, connections)
        file_path = os.path.join(save_directory, f'{district}_Connections.csv')
        df.to_csv(file_path, index=False)
    print(f"CSV files for each district have been created successfully in {save_directory}.")
else:
    print("No directory selected. Operation cancelled.")