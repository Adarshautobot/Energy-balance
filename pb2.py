import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
def load_data(filename='14 scaled - Copy.xlsx', sheet_name='Sheet1'): #change file name and sheet name as required.
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        timestamps = df['Timestamp']
        demand_ts = df['Electricity Consumption (MW)']
        pv_ts = df['PV Generation (MW)']
        wind_ts = df['Wind Generation (MW)']
        return timestamps, demand_ts, pv_ts, wind_ts
    except FileNotFoundError:
        print(f"Error: Excel file '{filename}' not found.")
        exit()
    except KeyError as e:
        print(f"Error: Column '{e.args[0]}' not found in the Excel sheet.")
        exit()

# 2. Create Network
def create_network():
    net = pp.create_empty_network()
    bus2 = pp.create_bus(net, vn_kv=220., name="Bus 2")
    pp.create_ext_grid(net, bus=bus2, name="Grid Connection", vm_pu=1.0)
    pp.create_load(net, bus=bus2, p_mw=0, name="Load")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="PV Plant")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="Wind Farm")
    pp.create_sgen(net, bus=bus2, p_mw=0, name="BESS")

    pv_index = net.sgen.index[net.sgen.name == "PV Plant"].tolist()[0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"].tolist()[0]
    bess_index = net.sgen.index[net.sgen.name == "BESS"][0]
    grid_index = net.ext_grid.index[0]

    return net, pv_index, wind_index, bess_index, grid_index

# 3. Battery Parameters
def get_battery_params():
    battery_soc_max = 1.0
    battery_soc_min = 0.1
    battery_soc_init = 0
    battery_charge_eff = 0.8944
    battery_discharge_eff = 0.8944
    return battery_soc_max, battery_soc_min, battery_soc_init, battery_charge_eff, battery_discharge_eff

# 4. Time-Series Simulation
def run_simulation(net, demand_ts, pv_ts, wind_ts, battery_capacity_MWh,  
                   battery_soc_max, battery_soc_min, battery_soc_init,
                   battery_charge_eff, battery_discharge_eff):
    results = []
    battery_soc = battery_soc_init
    bess_index = net.sgen.index[net.sgen.name == "BESS"][0]
    grid_index = net.ext_grid.index[0]
    pv_index = net.sgen.index[net.sgen.name == "PV Plant"][0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"][0]

    total_throughput_energy = 0  

    for t in range(len(demand_ts)):
        net.load.at[0, 'p_mw'] = demand_ts.iloc[t]
        net.sgen.at[pv_index, 'p_mw'] = pv_ts.iloc[t]
        net.sgen.at[wind_index, 'p_mw'] = wind_ts.iloc[t]

        max_charge_power_MW = 20000 #change this as required.
        #------------------------------------------------------------------------------
        max_discharge_power_MW = max_charge_power_MW

        excess_generation = pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t]

        battery_power_MW = 0
        if excess_generation > 0:
            potential_charge_MW = excess_generation
            limit_by_max_power_MW = max_charge_power_MW
            limit_by_soc_space_MW = (battery_soc_max - battery_soc) * battery_capacity_MWh
            battery_power_MW = -min(potential_charge_MW, limit_by_max_power_MW, limit_by_soc_space_MW)
        elif excess_generation < 0:
            potential_discharge_MW = abs(excess_generation)
            limit_by_max_power_MW = max_discharge_power_MW
            limit_by_soc_available_MW = (battery_soc - battery_soc_min) * battery_capacity_MWh
            battery_power_MW = min(potential_discharge_MW, limit_by_max_power_MW, limit_by_soc_available_MW)

        net.sgen.at[bess_index, 'p_mw'] = battery_power_MW

        curtailed_at_t = 0

        try:
            pp.runpp(net)
            generator_power = net.res_ext_grid.at[grid_index, 'p_mw']
            if generator_power < 0:
                curtailed_at_t = abs(generator_power)
                generator_power = 0
        except Exception as e:
            print(f"Power flow did not converge at time step {t}: {e}")
            generator_power = 0
            curtailed_at_t = max(0, pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t] + net.sgen.at[bess_index, 'p_mw'])

        actual_bess_terminal_power_MW = net.sgen.at[bess_index, 'p_mw']
        energy_change_MWh = 0
        if actual_bess_terminal_power_MW > 0:
            energy_removed_from_soc_MWh = (actual_bess_terminal_power_MW * (15 / 60)) / battery_discharge_eff
            energy_change_MWh = -energy_removed_from_soc_MWh
            total_throughput_energy += energy_removed_from_soc_MWh
        elif actual_bess_terminal_power_MW < 0:
            energy_stored_in_soc_MWh = (abs(actual_bess_terminal_power_MW) * (15 / 60)) * battery_charge_eff
            energy_change_MWh = energy_stored_in_soc_MWh
            total_throughput_energy += energy_stored_in_soc_MWh

        battery_soc += energy_change_MWh / battery_capacity_MWh
        battery_soc = np.clip(battery_soc, battery_soc_min, battery_soc_max)

        results.append({
            'time': t * 15,
            'load': demand_ts.iloc[t],
            'pv_gen': pv_ts.iloc[t],
            'wind_gen': wind_ts.iloc[t],
            'battery_power': actual_bess_terminal_power_MW,
            'battery_soc': battery_soc,
            'generator_power': generator_power,
            'curtailment_MW': curtailed_at_t
        })

    results_df = pd.DataFrame(results)
    full_cycles = total_throughput_energy / (2 * battery_capacity_MWh)
    results_df.attrs['battery_cycles'] = full_cycles
    return results_df

# 6. Analysis and Visualization
def analyze_and_plot(results_df, battery_capacity_MWh):
    total_load_energy = results_df['load'].sum() * (15 / 60)
    total_pv_energy = results_df['pv_gen'].sum() * (15 / 60)
    total_wind_energy = results_df['wind_gen'].sum() * (15 / 60)
    total_renewable_generation_MWh = total_pv_energy + total_wind_energy
    total_generator_import_energy = results_df['generator_power'].sum() * (15 / 60)
    total_battery_discharge_energy = results_df['battery_power'].apply(lambda x: max(0, x)).sum() * (15 / 60)
    total_battery_charge_energy = results_df['battery_power'].apply(lambda x: min(0, x)).sum() * (15 / 60)
    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * (15 / 60)

    print(f"\n--- Simulation Parameters ---")
    print(f"Fixed Battery Capacity: {battery_capacity_MWh:.2f} MWh")

    print("\n--- Energy Balance ---")
    print(f"Total Load Energy: {total_load_energy:.2f} MWh")
    print(f"Total PV Energy: {total_pv_energy:.2f} MWh")
    print(f"Total Wind Energy: {total_wind_energy:.2f} MWh")
    print(f"Total Renewable Generation: {total_renewable_generation_MWh:.2f} MWh")
    print(f"Total total_generator_import_Energy: {total_generator_import_energy:.2f} MWh")
    print(f"Total Battery Discharge Energy: {total_battery_discharge_energy:.2f} MWh")
    print(f"Total Battery Charge Energy: {total_battery_charge_energy:.2f} MWh")
    print(f"Total WASTED renewable Energy: {total_curtailment_energy_MWh:.2f} MWh")

    print("\n--- Battery Stats ---")
    print(f"Max SoC: {results_df['battery_soc'].max():.2f}")
    print(f"Min SoC: {results_df['battery_soc'].min():.2f}")
    print(f"Total Full Cycles Used: {results_df.attrs['battery_cycles']:.2f}")

    print(f"\nMax Fossil fuel Power: {results_df['generator_power'].max():.2f} MW")

    percentage_renewable_wasted = (total_curtailment_energy_MWh / total_renewable_generation_MWh) * 100 if total_renewable_generation_MWh > 0 else 0
    print(f"Percentage of Renewable Energy Wasted: {percentage_renewable_wasted:.2f} %")

    percentage_renewable_utilization = 100 - percentage_renewable_wasted if total_renewable_generation_MWh > 0 else 0
    print(f"Percentage of Renewable Energy Utilization: {percentage_renewable_utilization:.2f} %")

    renewable_energy_penetration = (total_pv_energy + total_wind_energy) / total_load_energy * 100 if total_load_energy > 0 else 0
    print(f"Renewable Energy Penetration (Gross Generation): {renewable_energy_penetration:.2f} %")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(results_df['time'], results_df['load'], label='Load', color='blue')
    ax1.plot(results_df['time'], results_df['pv_gen'] + results_df['wind_gen'], label='Renewables', color='green')
    ax1.plot(results_df['time'], results_df['battery_power'], label='Battery Power (Discharge+ / Charge-)', color='purple')
    ax1.plot(results_df['time'], results_df['generator_power'], label='Fossil power', color='red')
    ax1.plot(results_df['time'], results_df['curtailment_MW'], label='Curtailment', color='black', linestyle=':')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Power (MW)')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(results_df['time'], results_df['battery_soc'], label='Battery SoC', color='orange',linestyle=':')
    ax2.set_ylabel('Battery SoC')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.title("Power Balance and Battery SoC Over Time")
    plt.tight_layout()
    plt.show()
    print("Simulation Complete")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    timestamps, demand_ts, pv_ts, wind_ts = load_data()
    net, pv_index, wind_index, bess_index, grid_index = create_network()
    battery_soc_max, battery_soc_min, battery_soc_init, battery_charge_eff, battery_discharge_eff = get_battery_params()

    fixed_battery_capacity_MWh = 82292 #change this as required.
    print(f"Running simulation with fixed battery capacity: {fixed_battery_capacity_MWh:.2f} MWh")

    results_df = run_simulation(net, demand_ts, pv_ts, wind_ts, fixed_battery_capacity_MWh,
                                battery_soc_max, battery_soc_min, battery_soc_init,
                                battery_charge_eff, battery_discharge_eff)

    analyze_and_plot(results_df, fixed_battery_capacity_MWh)
