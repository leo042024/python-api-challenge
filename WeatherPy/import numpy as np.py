import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import os

# Ensure the output directory exists
if not os.path.exists('output_data'):
    os.makedirs('output_data')

# Create DataFrame with Northern Hemisphere data
northern_hemi_df = city_data_df[city_data_df['Lat'] >= 0]

def plot_linregress(x, y, xlabel='Latitude', ylabel='Max Temperature (°C)', title='Linear Regression Plot', filename='TempVLatRPlot_N_R.png'):
    x = np.array(x)
    y = np.array(y)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calculate predicted y values
    y_pred = intercept + slope * x

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', marker='o', label='Data Points')
    plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    output_path = f"output_data/{filename}"
    plt.savefig(output_path)

    # Show plot
    plt.show()

# Extracting data for linear regression
latitude = northern_hemi_df['Lat']
max_temp = northern_hemi_df['Max Temp']

# Plot linear regression for Northern Hemisphere data
plot_linregress(latitude, max_temp, title='Northern Hemisphere Latitude vs. Max Temperature')




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import os

# Ensure the output directory exists
if not os.path.exists('output_data'):
    os.makedirs('output_data')

# Create DataFrame with Northern Hemisphere data
northern_hemi_df = city_data_df[city_data_df['Lat'] >= 0]

def plot_linregress(x, y, xlabel='Latitude', ylabel='Max Temperature (°C)', title='Linear Regression Plot', filename='TempVLatRPlot_N_R.png'):
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calculate predicted y values
    y_pred = intercept + slope * x

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', marker='o', label='Data Points')
    plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    output_path = f"output_data/{filename}"
    plt.savefig(output_path)

    # Show plot
    plt.show()

# Extracting data for linear regression
latitude = northern_hemi_df['Lat']
cloudiness = northern_hemi_df['Cloudiness']

# Plot linear regression for Latitude vs. Cloudiness
plot_linregress(latitude, cloudiness, title='Northern Hemisphere Latitude vs. Cloudiness')
