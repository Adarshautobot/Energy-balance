import pandas as pd

# Load Excel file
file_path = "13.xlsx"  # Adjust path as needed
df = pd.read_excel(file_path)

# Extract relevant columns
consumption = df["Electricity Consumption (MW)"]
wind = df["Wind Generation (MW)"]
pv = df["PV Generation (MW)"]

# Calculate statistics
max_avg_demand = consumption.mean()
target_avg_renewable = 1 * max_avg_demand

mean_wind = wind.mean()
mean_pv = pv.mean()

# Calculate ratio-based factors
solar_to_wind_ratio = ((1 - 0.6) * mean_wind) / (0.6 * mean_pv)
wind_factor = target_avg_renewable / (mean_wind + solar_to_wind_ratio * mean_pv)
solar_factor = solar_to_wind_ratio * wind_factor

# Scale wind and solar generation
scaled_wind = wind * wind_factor
scaled_pv = pv * solar_factor
df["Wind Generation (MW)"] = scaled_wind
df["PV Generation (MW)"] = scaled_pv
df["Total Scaled Renewables (MW)"] = scaled_wind + scaled_pv

# Calculate total hourly energy
# Assuming the DataFrame is indexed by time and contains 15-min intervals
df['Time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='15T')  # Adjust start date as necessary
df.set_index('Time', inplace=True)

# Resample to hourly data by taking the mean of every four 15-min intervals
hourly_consumption = df['Electricity Consumption (MW)'].resample('H').mean()
hourly_wind = df['Wind Generation (MW)'].resample('H').mean()
hourly_pv = df['PV Generation (MW)'].resample('H').mean()

# Calculate total energy in MWh for each hour
total_load_energy_mwh = hourly_consumption.sum() / 4  # Convert MW to MWh (1 hour = 4 intervals)
total_wind_energy_mwh = hourly_wind.sum() / 4
total_pv_energy_mwh = hourly_pv.sum() / 4

# Output results
output_file = "scaled_renewables.xlsx"
df.to_excel(output_file, index=False)

print("Scaling complete.")
print(f"Wind factor: {wind_factor:.2f}, Solar factor: {solar_factor:.2f}")
print(f"Total Load Energy (MWh): {total_load_energy_mwh:.2f}")
print(f"Total Wind Energy (MWh): {total_wind_energy_mwh:.2f}")
print(f"Total PV Energy (MWh): {total_pv_energy_mwh:.2f}")
print(f"Output saved to: {output_file}")