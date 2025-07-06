import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
def load_data(filename='16-130%renewable.xlsx', sheet_name='Sheet1'):# modify file name here and sheet name
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        # Ensure data is loaded in 15-minute intervals 
        timestamps = df['Timestamp']
        demand_ts = df['Electricity Consumption (MW)']
        pv_ts = df['PV Generation (MW)']
        wind_ts = df['Wind Generation (MW)']
        print(f"Data loaded from '{filename}', assuming 15-minute intervals. Total data points: {len(df)}")
        return timestamps, demand_ts, pv_ts, wind_ts
    except FileNotFoundError:
        print(f"Error: Excel file '{filename}' not found.")
        exit()
    except KeyError as e:
        print(f"Error: Column '{e.args[0]}' not found in the Excel sheet. Please check column names.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        print("Please ensure your Excel file structure is as expected with numerical columns for generation/consumption.")
        exit()

# 2. Create Network
def create_network():
    net = pp.create_empty_network()
    # Bus voltage is set to 220 kV.
    bus2 = pp.create_bus(net, vn_kv=220., name="Bus 2")
    pp.create_ext_grid(net, bus=bus2, name="Grid Connection", vm_pu=1.0)
    pp.create_load(net, bus=bus2, p_mw=0, name="Load")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="PV Plant")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="Wind Farm")
    
    # Create two separate BESS (Battery Energy Storage Systems)
    pp.create_sgen(net, bus=bus2, p_mw=0, name="BESS1 Primary")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="BESS2 Backup") 

    pv_index = net.sgen.index[net.sgen.name == "PV Plant"].tolist()[0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"].tolist()[0]
    bess1_index = net.sgen.index[net.sgen.name == "BESS1 Primary"].tolist()[0]
    bess2_index = net.sgen.index[net.sgen.name == "BESS2 Backup"].tolist()[0]
    grid_index = net.ext_grid.index[0]

    return net, pv_index, wind_index, bess1_index, bess2_index, grid_index

# 3. Battery Parameters
def get_battery_params():
    battery_soc_max = 1.0
    battery_soc_min = 0.1
    # Initial SOC for both batteries
    battery1_soc_init = 1 
    battery2_soc_init = 0.1
    battery1_charge_eff = 0.894
    battery1_discharge_eff = 0.894
    battery2_charge_eff = 0.6324
    battery2_discharge_eff = 0.6324
    
    return battery_soc_max, battery_soc_min, battery1_soc_init, battery2_soc_init, \
           battery1_charge_eff, battery1_discharge_eff, battery2_charge_eff, battery2_discharge_eff

# 4. Time-Series Simulation
def run_simulation(net, demand_ts, pv_ts, wind_ts, 
                   battery1_capacity_MWh, battery1_power_MW,
                   battery2_capacity_MWh, battery2_power_MW,
                   battery_soc_max, battery_soc_min, 
                   battery1_soc_init, battery2_soc_init,
                   battery1_charge_eff, battery1_discharge_eff, 
                   battery2_charge_eff, battery2_discharge_eff):
    
    results = []
    
    battery1_soc = battery1_soc_init
    battery2_soc = battery2_soc_init

    bess1_index = net.sgen.index[net.sgen.name == "BESS1 Primary"].tolist()[0]
    bess2_index = net.sgen.index[net.sgen.name == "BESS2 Backup"].tolist()[0]
    pv_index = net.sgen.index[net.sgen.name == "PV Plant"].tolist()[0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"].tolist()[0]
    grid_index = net.ext_grid.index[0]

    battery1_throughput_energy_MWh = 0
    battery2_throughput_energy_MWh = 0
    time_step_hours = 15 / 60 # 15 minutes in hours

    for t in range(len(demand_ts)):
        # Update load and generation
        net.load.at[0, 'p_mw'] = demand_ts.iloc[t]
        net.sgen.at[pv_index, 'p_mw'] = pv_ts.iloc[t]
        net.sgen.at[wind_index, 'p_mw'] = wind_ts.iloc[t]
        excess_generation = pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t]
        battery1_power_MW_t = 0
        battery2_power_MW_t = 0
        remaining_excess = excess_generation 
        # --- Dispatch Logic ---
        if remaining_excess > 0: 
            potential_charge_bess1 = min(remaining_excess, battery1_power_MW, 
                                         (battery_soc_max - battery1_soc) * battery1_capacity_MWh / time_step_hours)
            battery1_power_MW_t = -potential_charge_bess1
            remaining_excess -= potential_charge_bess1

            # charge BESS2 
            if remaining_excess > 0:
                potential_charge_bess2 = min(remaining_excess, battery2_power_MW, 
                                             (battery_soc_max - battery2_soc) * battery2_capacity_MWh / time_step_hours)
                battery2_power_MW_t = -potential_charge_bess2 
                remaining_excess -= potential_charge_bess2

        elif remaining_excess < 0: 
            required_discharge = abs(remaining_excess)
            
            # Prioritize BESS1
            potential_discharge_bess1 = min(required_discharge, battery1_power_MW, 
                                            (battery1_soc - battery_soc_min) * battery1_capacity_MWh / time_step_hours)
            battery1_power_MW_t = potential_discharge_bess1 
            required_discharge -= potential_discharge_bess1

            #discharge BESS2 
            if required_discharge > 0:
                potential_discharge_bess2 = min(required_discharge, battery2_power_MW, 
                                                (battery2_soc - battery_soc_min) * battery2_capacity_MWh / time_step_hours)
                battery2_power_MW_t = potential_discharge_bess2 
                required_discharge -= potential_discharge_bess2
        
        # Apply battery dispatch to the network
        net.sgen.at[bess1_index, 'p_mw'] = battery1_power_MW_t
        net.sgen.at[bess2_index, 'p_mw'] = battery2_power_MW_t

        curtailed_at_t = 0 
        generator_power = 0

        try:
            pp.runpp(net)
            # Check power from the external grid. Positive is import, negative is export.
            #Assuming grid export is curtailed to 0, if not used by the batteries
            # if generator_power < 0, it means there's excess power flowing back to the grid
            generator_power_from_grid = net.res_ext_grid.at[grid_index, 'p_mw']
            
            if generator_power_from_grid < 0: # This means there's an export from the network
                curtailed_at_t = abs(generator_power_from_grid) 
                generator_power = 0  # Force export to 0 
            else:
                generator_power = generator_power_from_grid # This is grid import
            
        except Exception as e:
            print(f"Power flow did not converge at time step {t}: {e}")
            generator_power = 0
            # If power flow fails, assume any remaining excess generation that couldn't be stored or consumed is curtailed.
            curtailed_at_t = max(0, remaining_excess)


        if battery1_power_MW_t > 0: # Discharging
            energy_removed_bess1_MWh = (battery1_power_MW_t * time_step_hours) / battery1_discharge_eff
            battery1_soc -= energy_removed_bess1_MWh / battery1_capacity_MWh
            battery1_throughput_energy_MWh += battery1_power_MW_t * time_step_hours
        elif battery1_power_MW_t < 0: # Charging
            energy_stored_bess1_MWh = (abs(battery1_power_MW_t) * time_step_hours) * battery1_charge_eff
            battery1_soc += energy_stored_bess1_MWh / battery1_capacity_MWh
            battery1_throughput_energy_MWh += abs(battery1_power_MW_t) * time_step_hours
        battery1_soc = np.clip(battery1_soc, battery_soc_min, battery_soc_max)

        if battery2_power_MW_t > 0: # Discharging
            energy_removed_bess2_MWh = (battery2_power_MW_t * time_step_hours) / battery2_discharge_eff
            battery2_soc -= energy_removed_bess2_MWh / battery2_capacity_MWh
            battery2_throughput_energy_MWh += battery2_power_MW_t * time_step_hours
        elif battery2_power_MW_t < 0: # Charging
            energy_stored_bess2_MWh = (abs(battery2_power_MW_t) * time_step_hours) * battery2_charge_eff
            battery2_soc += energy_stored_bess2_MWh / battery2_capacity_MWh
            battery2_throughput_energy_MWh += abs(battery2_power_MW_t) * time_step_hours
        battery2_soc = np.clip(battery2_soc, battery_soc_min, battery_soc_max)

        results.append({
            'time': t * 15, # Convert timestep index to minutes
            'load': demand_ts.iloc[t],
            'pv_gen': pv_ts.iloc[t],
            'wind_gen': wind_ts.iloc[t],
            'battery1_power': battery1_power_MW_t,
            'battery1_soc': battery1_soc,
            'battery2_power': battery2_power_MW_t,
            'battery2_soc': battery2_soc,
            'total_battery_power': battery1_power_MW_t + battery2_power_MW_t, 
            'generator_power': generator_power,
            'curtailment_MW': curtailed_at_t # Curtailment in MW at this timestep
        })

    results_df = pd.DataFrame(results)

    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * time_step_hours
    total_generator_import_energy = results_df['generator_power'].sum() * time_step_hours

    total_simulation_duration_hours = len(demand_ts) * time_step_hours
    total_simulation_duration_days = total_simulation_duration_hours / 24

    return total_curtailment_energy_MWh, total_generator_import_energy, \
           battery1_throughput_energy_MWh, battery2_throughput_energy_MWh, \
           total_simulation_duration_days, results_df

# 6. Analysis and Visualization
def analyze_and_plot(results_df, 
                     battery1_capacity_MWh, battery1_power_MW,
                     battery2_capacity_MWh, battery2_power_MW,
                     battery1_cycles_per_day, battery2_cycles_per_day):
    
    total_load_energy = results_df['load'].sum() * (15 / 60) # Sum of MW * (15/60) hours gives MWh
    total_pv_energy = results_df['pv_gen'].sum() * (15 / 60)
    total_wind_energy = results_df['wind_gen'].sum() * (15 / 60)
    total_renewable_generation_MWh = total_pv_energy + total_wind_energy

    total_generator_import_energy = results_df['generator_power'].sum() * (15 / 60)

    total_battery1_discharge_energy = results_df['battery1_power'].apply(lambda x: max(0, x)).sum() * (15 / 60)
    total_battery1_charge_energy = results_df['battery1_power'].apply(lambda x: min(0, x)).sum() * (15 / 60)
    total_battery2_discharge_energy = results_df['battery2_power'].apply(lambda x: max(0, x)).sum() * (15 / 60)
    total_battery2_charge_energy = results_df['battery2_power'].apply(lambda x: min(0, x)).sum() * (15 / 60)

    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * (15 / 60)

    print(f"\n--- Simulation Parameters ---")
    print(f"BESS1 (Primary) Capacity: {battery1_capacity_MWh:.2f} MWh, Power: {battery1_power_MW:.2f} MW")
    print(f"BESS2 (Backup) Capacity: {battery2_capacity_MWh:.2f} MWh, Power: {battery2_power_MW:.2f} MW")
    
    print("\n--- Energy Balance ---")
    print(f"Total Load Energy: {total_load_energy:.2f} MWh")
    print(f"Total PV Energy: {total_pv_energy:.2f} MWh")
    print(f"Total Wind Energy: {total_wind_energy:.2f} MWh")
    print(f"Total Renewable Generation: {total_renewable_generation_MWh:.2f} MWh")
    print(f"Total FOSSIL FUEL Import Energy: {total_generator_import_energy:.2f} MWh")
    print(f"Total BESS1 Discharge Energy: {total_battery1_discharge_energy:.2f} MWh")
    print(f"Total BESS1 Charge Energy: {total_battery1_charge_energy:.2f} MWh")
    print(f"Total BESS2 Discharge Energy: {total_battery2_discharge_energy:.2f} MWh")
    print(f"Total BESS2 Charge Energy: {total_battery2_charge_energy:.2f} MWh")
    print(f"Total WASTED renewable Energy (Curtailment): {total_curtailment_energy_MWh:.2f} MWh")

    print("\n--- Battery Stats ---")
    print(f"BESS1 Max SoC: {results_df['battery1_soc'].max():.2f}")
    print(f"BESS1 Min SoC: {results_df['battery1_soc'].min():.2f}")
    print(f"BESS1 Cycles per day: {battery1_cycles_per_day:.2f}")
    print(f"BESS2 Max SoC: {results_df['battery2_soc'].max():.2f}")
    print(f"BESS2 Min SoC: {results_df['battery2_soc'].min():.2f}")
    print(f"BESS2 Cycles per day: {battery2_cycles_per_day:.2f}")

    print(f"\nMax fossil fuel Import Power: {results_df['generator_power'].max():.2f} MW")

    # Calculate Wasted Renewable Energy Percentage
    percentage_renewable_wasted = 0
    if total_renewable_generation_MWh > 0:
        percentage_renewable_wasted = (total_curtailment_energy_MWh / total_renewable_generation_MWh) * 100
    print(f"Percentage of Renewable Energy Wasted: {percentage_renewable_wasted:.2f} %")

    # Calculate Renewable Energy Utilization Percentage
    percentage_renewable_utilization = 0
    if total_renewable_generation_MWh > 0:
        percentage_renewable_utilization = 100 - percentage_renewable_wasted
    print(f"Percentage of Renewable Energy Utilization: {percentage_renewable_utilization:.2f} %")

    # Renewable Energy Penetration (Gross Generation)
    renewable_energy_penetration = 0
    if total_load_energy > 0:
        renewable_energy_penetration = (total_pv_energy + total_wind_energy) / total_load_energy * 100
    print(f"Renewable Energy Penetration (Gross Generation): {renewable_energy_penetration:.2f} %")


    fig, ax1 = plt.subplots(figsize=(14, 7)) # Increased figure size for better readability

    # Plot power flows
    ax1.plot(results_df['time'], results_df['load'], label='Load', color='blue', linestyle='-')
    ax1.plot(results_df['time'], results_df['pv_gen'] + results_df['wind_gen'], label='Renewables', color='green', linestyle='-')
    #ax1.plot(results_df['time'], results_df['total_battery_power'], label='Total Battery Power', color='purple', linestyle='-')
    ax1.plot(results_df['time'], results_df['battery1_power'], label='Short term storage', color='darkorange', linestyle='--')
    ax1.plot(results_df['time'], results_df['battery2_power'], label='Long term storage', color='darkviolet', linestyle='--')
    ax1.plot(results_df['time'], results_df['generator_power'], label='Fossil fuel', color='red', linestyle='-')
    ax1.plot(results_df['time'], results_df['curtailment_MW'], label='Curtailment', color='black', linestyle=':')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Power (MW)')
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1)) 

    # Plot SoC on a twin axis
    ax2 = ax1.twinx()
    #ax2.plot(results_df['time'], results_df['battery1_soc'], label='BESS1 SoC', color='cyan', linestyle='-.')
    ax2.plot(results_df['time'], results_df['battery2_soc'], label='BESS2 SoC', color='red', linestyle='-')
    ax2.set_ylabel('Battery SoC')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1)) # Adjust legend position

    plt.title("Power Balance and Battery SoCs Over Time (Two Batteries)")
    plt.tight_layout()
    plt.show()
    print("Simulation Complete")
    cost_per_mwh_opt = 5000
    cost_per_mw_opt = 1750000
    total_capital_cost_optimized_system = (fixed_battery2_capacity_MWh *cost_per_mwh_opt) + \
                                          (fixed_battery2_power_MW *cost_per_mw_opt)
    print(f"Total Estimated Capital Cost for Optimized BESS: {total_capital_cost_optimized_system:,.2f} EUR")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    timestamps, demand_ts, pv_ts, wind_ts = load_data()
    net, pv_index, wind_index, bess1_index, bess2_index, grid_index = create_network()
    
    battery_soc_max, battery_soc_min, battery1_soc_init, battery2_soc_init, \
    battery1_charge_eff, battery1_discharge_eff, \
    battery2_charge_eff, battery2_discharge_eff = get_battery_params()

    # Define fixed capacities and power ratings for the two batteries------------------------------------------
    # BESS1 (Primary SHORT TERM STORAGE)
    fixed_battery1_capacity_MWh = 82000.0
    fixed_battery1_power_MW = 20000.0 
    # BESS2 (Secondary LONG TERM STORAGE)
    fixed_battery2_capacity_MWh = 25000000
    fixed_battery2_power_MW = 18700
    
    print(f"\nRunning simulation with two fixed battery systems:")
    print(f"  Primary Battery (BESS1): Capacity = {fixed_battery1_capacity_MWh:.2f} MWh, Power = {fixed_battery1_power_MW:.2f} MW")
    print(f"  Backup Battery (BESS2): Capacity = {fixed_battery2_capacity_MWh:.2f} MWh, Power = {fixed_battery2_power_MW:.2f} MW")

    total_curtailment, total_grid_import, \
    battery1_throughput_energy_MWh, battery2_throughput_energy_MWh, \
    total_simulation_duration_days, results_df = run_simulation(
        net, demand_ts, pv_ts, wind_ts, 
        fixed_battery1_capacity_MWh, fixed_battery1_power_MW,
        fixed_battery2_capacity_MWh, fixed_battery2_power_MW,
        battery_soc_max, battery_soc_min, 
        battery1_soc_init, battery2_soc_init,
        battery1_charge_eff, battery1_discharge_eff,
        battery2_charge_eff, battery2_discharge_eff
    )

    # Calculate cycles per day for each battery
    battery1_cycles_per_day = 0
    if fixed_battery1_capacity_MWh > 0 and total_simulation_duration_days > 0:
        # A full cycle is equivalent to discharging and charging the full capacity once (throughput / (2 * capacity))
        battery1_equivalent_cycles = battery1_throughput_energy_MWh / (2 * fixed_battery1_capacity_MWh)
        battery1_cycles_per_day = battery1_equivalent_cycles / total_simulation_duration_days

    battery2_cycles_per_day = 0
    if fixed_battery2_capacity_MWh > 0 and total_simulation_duration_days > 0:
        battery2_equivalent_cycles = battery2_throughput_energy_MWh / (2 * fixed_battery2_capacity_MWh)
        battery2_cycles_per_day = battery2_equivalent_cycles / total_simulation_duration_days

    analyze_and_plot(results_df, 
                     fixed_battery1_capacity_MWh, fixed_battery1_power_MW,
                     fixed_battery2_capacity_MWh, fixed_battery2_power_MW,
                     battery1_cycles_per_day, battery2_cycles_per_day)
    
    cost_per_mwh_opt = 5000  # Cost per MWh of storage capacity (EUR)
    cost_per_mw_opt = 1750000  # Cost per MW of power capacity (EUR)
    
    # Calculate total cost for BESS2 (Backup Battery)
    total_capital_cost_optimized_system = (fixed_battery2_capacity_MWh * cost_per_mwh_opt) + \
                                          (fixed_battery2_power_MW * cost_per_mw_opt)
    print(f"Total Estimated Capital Cost for Optimized BESS (Backup): {total_capital_cost_optimized_system:,.2f} EUR")
