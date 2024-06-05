import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

# Data for all 77 districts of Nepal
district_data = [
    {"District": "Achham", "Population": 257477, "Area": 1682},
    {"District": "Arghakhanchi", "Population": 197632, "Area": 1184},
    {"District": "Baglung", "Population": 324487, "Area": 1784},
    {"District": "Baitadi", "Population": 250898, "Area": 1506},
    {"District": "Bajhang", "Population": 195159, "Area": 3422},
    {"District": "Bajura", "Population": 108781, "Area": 2116},
    {"District": "Banke", "Population": 385840, "Area": 2257},
    {"District": "Bara", "Population": 687708, "Area": 1190},
    {"District": "Bardiya", "Population": 426576, "Area": 2025},
    {"District": "Bhaktapur", "Population": 413724, "Area": 119},
    {"District": "Bhojpur", "Population": 182459, "Area": 1506},
    {"District": "Chitwan", "Population": 579984, "Area": 2178},
    {"District": "Dadeldhura", "Population": 142094, "Area": 1254},
    {"District": "Dailekh", "Population": 261770, "Area": 1544},
    {"District": "Dang", "Population": 552583, "Area": 2955},
    {"District": "Darchula", "Population": 133274, "Area": 2363},
    {"District": "Dhading", "Population": 336067, "Area": 1926},
    {"District": "Dhankuta", "Population": 163412, "Area": 891},
    {"District": "Dhanusha", "Population": 754777, "Area": 1180},
    {"District": "Dolakha", "Population": 204229, "Area": 2191},
    {"District": "Dolpa", "Population": 36485, "Area": 7885},
    {"District": "Doti", "Population": 211746, "Area": 2310},
    {"District": "Gorkha", "Population": 271061, "Area": 3610},
    {"District": "Gulmi", "Population": 296654, "Area": 1144},
    {"District": "Humla", "Population": 50949, "Area": 5541},
    {"District": "Ilam", "Population": 290254, "Area": 1703},
    {"District": "Jajarkot", "Population": 171304, "Area": 2191},
    {"District": "Jhapa", "Population": 812650, "Area": 1606},
    {"District": "Jumla", "Population": 108921, "Area": 2531},
    {"District": "Kailali", "Population": 775709, "Area": 3245},
    {"District": "Kalikot", "Population": 136948, "Area": 1626},
    {"District": "Kanchanpur", "Population": 451248, "Area": 1543},
    {"District": "Kapilvastu", "Population": 571936, "Area": 1738},
    {"District": "Kaski", "Population": 492098, "Area": 2017},
    {"District": "Kathmandu", "Population": 1744246, "Area": 395},
    {"District": "Kavrepalanchok", "Population": 381937, "Area": 1394},
    {"District": "Khotang", "Population": 206312, "Area": 1591},
    {"District": "Lalitpur", "Population": 468132, "Area": 385},
    {"District": "Lamjung", "Population": 167724, "Area": 1396},
    {"District": "Mahottari", "Population": 627580, "Area": 1002},
    {"District": "Makwanpur", "Population": 420477, "Area": 2426},
    {"District": "Manang", "Population": 10656, "Area": 2246},
    {"District": "Morang", "Population": 965370, "Area": 1855},
    {"District": "Mugu", "Population": 55574, "Area": 3779},
    {"District": "Mustang", "Population": 133246, "Area": 3573},
    {"District": "Myagdi", "Population": 113641, "Area": 2294},
    {"District": "Nawalpur", "Population": 280406, "Area": 1226},
    {"District": "Nuwakot", "Population": 277471, "Area": 1121},
    {"District": "Okhaldhunga", "Population": 147984, "Area": 1074},
    {"District": "Palpa", "Population": 261180, "Area": 1373},
    {"District": "Panchthar", "Population": 191817, "Area": 1241},
    {"District": "Parbat", "Population": 146590, "Area": 494},
    {"District": "Parsa", "Population": 601017, "Area": 1353},
    {"District": "Pyuthan", "Population": 212484, "Area": 1306},
    {"District": "Ramechhap", "Population": 202646, "Area": 1563},
    {"District": "Rasuwa", "Population": 67940, "Area": 1545},
    {"District": "Rautahat", "Population": 686722, "Area": 1127},
    {"District": "Rolpa", "Population": 186557, "Area": 1257},
    {"District": "Rukum East", "Population": 154026, "Area": 2227},
    {"District": "Rukum West", "Population": 189805, "Area": 2357},
    {"District": "Rupandehi", "Population": 886706, "Area": 1360},
    {"District": "Salyan", "Population": 242444, "Area": 1503},
        {"District": "Sankhuwasabha", "Population": 159886, "Area": 3506},
    {"District": "Saptari", "Population": 639284, "Area": 1363},
    {"District": "Sarlahi", "Population": 769729, "Area": 1259},
    {"District": "Sindhuli", "Population": 296192, "Area": 2491},
    {"District": "Sindhupalchok", "Population": 287798, "Area": 2540},
    {"District": "Siraha", "Population": 637328, "Area": 1189},
    {"District": "Solukhumbu", "Population": 107686, "Area": 3340},
    {"District": "Sunsari", "Population": 818250, "Area": 1257},
    {"District": "Surkhet", "Population": 350804, "Area": 2426},
    {"District": "Syangja", "Population": 289148, "Area": 1164},
    {"District": "Tanahun", "Population": 323288, "Area": 1545},
    {"District": "Taplejung", "Population": 127461, "Area": 3646},
    {"District": "Terhathum", "Population": 113111, "Area": 679},
    {"District": "Udayapur", "Population": 317532, "Area": 2046}
]

class CSVGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Generator")

        self.data = pd.DataFrame(district_data)

        self.save_button = tk.Button(root, text="Save CSV", command=self.save_csv)
        self.save_button.pack()

    def save_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"CSV file saved successfully at {file_path}")
        else:
            messagebox.showinfo("Info", "No file selected. Operation cancelled.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVGeneratorApp(root)
    root.mainloop()

