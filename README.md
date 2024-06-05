
# Virus Analysis App

Virus Analysis App is a Python-based application that allows users to analyze the spread of a virus in various cities. The app uses traffic data, population data, and geographic data to visualize the spread and find the safest and shortest path between cities using the A* algorithm.


## Features

- Load city traffic data from CSV files.
- Load district geographic data (latitude and longitude) from a CSV file.
- Load population data (population and area) from a CSV file.
![image](https://github.com/SaPok5/Virus-Spread-Analysis-using-Graph-concept-and-path-finding-algo/assets/98219026/e6edb62e-74a2-4347-a9e2-290ece27ac5b)

- Analyze data to create a weighted graph based on virus infection rates.
 ![image](https://github.com/SaPok5/Virus-Spread-Analysis-using-Graph-concept-and-path-finding-algo/assets/98219026/d742cb15-b5d7-4477-b6fb-c495d7143ead)

- Visualize the graph with nodes representing cities and edges representing traffic flow.
  ![image](https://github.com/SaPok5/Virus-Spread-Analysis-using-Graph-concept-and-path-finding-algo/assets/98219026/260d5b93-41ea-41b5-b028-8ecee4684e17)

- Highlight the safest and shortest path between two cities using the A* algorithm.
  ![image](https://github.com/SaPok5/Virus-Spread-Analysis-using-Graph-concept-and-path-finding-algo/assets/98219026/f1529715-4050-4d5c-8e2a-b0560b39fbce)

- User-friendly GUI built with Tkinter.

## Requirements

- Python 3.x
- Tkinter
- pandas
- networkx
- matplotlib
- heapq (standard library)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/virus-analysis-app.git
    cd virus-analysis-app
    ```

2. Install the required Python packages:
    ```bash
    pip install pandas networkx matplotlib
    ```

## Usage

1. Run the application:
    ```bash
    python virus_analysis_app.py
    ```

2. Enter the virus infection rate (1-10).

3. Optionally enter the start city and end city for pathfinding.

4. Load the required data files:
    - Click "Load City Data" to select a folder containing CSV files with city traffic data.
    - Click "Load Geodata" to select a CSV file with district geographic data.
    - Click "Load Population Data" to select a CSV file with district population data.

5. Click "Analyze Data" to process the data, visualize the graph, and (if cities were entered) find and highlight the safest and shortest path between the start and end cities.

## Data Format

### City Traffic Data

CSV files should be named in the format `CityName_traffic.csv` and contain the following columns:

- `To District`: The destination district.
- `Incoming Traffic`: The amount of incoming traffic.
- `Outgoing Traffic`: The amount of outgoing traffic.

### District Geographic Data

CSV file with the following columns:

- `District`: The name of the district.
- `Latitude`: The latitude of the district.
- `Longitude`: The longitude of the district.

### Population Data

CSV file with the following columns:

- `District`: The name of the district.
- `Population`: The population of the district.
- `Area`: The area of the district.

## Example

### City Traffic Data (`City1_traffic.csv`)
```csv
To District,Incoming Traffic,Outgoing Traffic
City2,100,150
City3,200,100
```

### District Geographic Data (`district_geodata.csv`)
```csv
District,Latitude,Longitude
City1,34.052235,-118.243683
City2,36.778259,-119.417931
City3,40.712776,-74.005974
```

### Population Data (`population_data.csv`)
```csv
District,Population,Area
City1,1000000,503
City2,500000,400
City3,800000,783
```



