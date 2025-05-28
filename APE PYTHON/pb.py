import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso

# 1. Load Data
def load_data(filename='14 scaled - Copy.xlsx', sheet_name='Sheet1'):
    try:

        df = pd.read_excel(filename, sheet_name=sheet_name)

        data_columns = ['Electricity Consumption (MW)', 'PV Generation (MW)', 'Wind Generation (MW)']
        df_data = df[data_columns]
        df_hourly = df_data.groupby(np.arange(len(df_data)) // 4).mean()
        if 'Timestamp' in df.columns and len(df) > 0:
            try:
                # Try to parse the first timestamp to get a sensible start point
                first_original_timestamp_str = str(df['Timestamp'].iloc[0]).split(' - ')[0].strip()
                start_datetime = pd.to_datetime(first_original_timestamp_str, format='%d.%m.%Y %H:%M')
            except Exception:
                start_datetime = pd.to_datetime('2024-01-01 00:00') # Fallback if parsing fails
        else:
            start_datetime = pd.to_datetime('2024-01-01 00:00') # Default fallback

        timestamps = pd.date_range(start=start_datetime, periods=len(df_hourly), freq='H')

        demand_ts = df_hourly['Electricity Consumption (MW)']
        pv_ts = df_hourly['PV Generation (MW)']
        wind_ts = df_hourly['Wind Generation (MW)']

        print(f"Data loaded and manually aggregated. Original 15-min data points: {len(df)}")
        print(f"New 60-min data points: {len(df_hourly)}")
        # --- MODIFICATION END ---

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
# MODIFICATION: Added battery_power_MW to the function signature
def run_simulation(net, demand_ts, pv_ts, wind_ts, battery_capacity_MWh, battery_power_MW,
                   battery_soc_max, battery_soc_min, battery_soc_init,
                   battery_charge_eff, battery_discharge_eff, is_pso_run=False):
    results = []
    battery_soc = battery_soc_init
    bess_index = net.sgen.index[net.sgen.name == "BESS"][0]
    grid_index = net.ext_grid.index[0]
    pv_index = net.sgen.index[net.sgen.name == "PV Plant"][0]
    wind_index = net.sgen.index[net.sgen.name == "Wind Farm"][0]

    battery_throughput_energy_MWh = 0

    # The time_step_hours is now 1.0 because data is hourly
    time_step_hours = 1.0

    for t in range(len(demand_ts)):
        net.load.at[0, 'p_mw'] = demand_ts.iloc[t]
        net.sgen.at[pv_index, 'p_mw'] = pv_ts.iloc[t]
        net.sgen.at[wind_index, 'p_mw'] = wind_ts.iloc[t]

        # MODIFICATION: Use battery_power_MW directly for max charge/discharge
        max_charge_power_MW = battery_power_MW
        max_discharge_power_MW = battery_power_MW

        excess_generation = pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t]

        bess_dispatch_MW = 0 # This variable tracks the power to be dispatched from/to BESS
        if excess_generation > 0: # Excess generation -> try to charge battery
            # Limit charge by: available excess, battery's max charge power, remaining capacity
            bess_dispatch_MW = -min(excess_generation, max_charge_power_MW, (battery_soc_max - battery_soc) * battery_capacity_MWh)
        elif excess_generation < 0: # Deficit generation -> try to discharge battery
            # Limit discharge by: absolute deficit, battery's max discharge power, available energy in battery
            bess_dispatch_MW = min(abs(excess_generation), max_discharge_power_MW, (battery_soc - battery_soc_min) * battery_capacity_MWh)

        net.sgen.at[bess_index, 'p_mw'] = bess_dispatch_MW # Assign to BESS sgen

        curtailed_at_t = 0
        try:
            pp.runpp(net)
            generator_power = net.res_ext_grid.at[grid_index, 'p_mw']
            # If generator_power is negative, it means there's excess generation being exported.
            # We assume this is curtailed if not explicitly handled as export.
            if generator_power < 0:
                curtailed_at_t = abs(generator_power) # Record this as curtailment
                generator_power = 0 # Ensure grid import is not negative (no export)
        except Exception as e:
            if not is_pso_run: # Only print power flow errors if not in PSO run
                print(f"Power flow did not converge at time step {t}: {e}")
            generator_power = 0
            # Fallback for curtailed_at_t if power flow fails, assume excess is curtailed
            curtailed_at_t = max(0, pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t] + net.sgen.at[bess_index, 'p_mw'])

        actual_bess_terminal_power_MW = net.sgen.at[bess_index, 'p_mw']
        energy_change_MWh = 0
        if actual_bess_terminal_power_MW > 0: # Battery is discharging (providing power to the grid/load)
            energy_removed = (actual_bess_terminal_power_MW * time_step_hours) / battery_discharge_eff
            energy_change_MWh = -energy_removed # SoC decreases
            battery_throughput_energy_MWh += actual_bess_terminal_power_MW * time_step_hours
        elif actual_bess_terminal_power_MW < 0: # Battery is charging (consuming power from grid/renewables)
            energy_stored = (abs(actual_bess_terminal_power_MW) * time_step_hours) * battery_charge_eff
            energy_change_MWh = energy_stored # SoC increases
            battery_throughput_energy_MWh += abs(actual_bess_terminal_power_MW) * time_step_hours

        battery_soc += energy_change_MWh / battery_capacity_MWh
        battery_soc = np.clip(battery_soc, battery_soc_min, battery_soc_max)

        results.append({
            'time': t * 60, # Time in minutes for plotting, but now represents hours * 60
            'load': demand_ts.iloc[t],
            'pv_gen': pv_ts.iloc[t],
            'wind_gen': wind_ts.iloc[t],
            'battery_power': actual_bess_terminal_power_MW,
            'battery_soc': battery_soc,
            'generator_power': generator_power, # Grid import only (export zeroed)
            'curtailment_MW': curtailed_at_t
        })

    results_df = pd.DataFrame(results)

    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * time_step_hours
    total_generator_import_energy = results_df['generator_power'].sum() * time_step_hours
    calculated_utilization_ratio = battery_throughput_energy_MWh / battery_capacity_MWh

    # Calculate total simulation duration in days for cycles per day calculation
    total_simulation_duration_hours = len(demand_ts) * time_step_hours
    total_simulation_duration_days = total_simulation_duration_hours / 24

    if not is_pso_run:
        print(f"Battery Throughput Energy: {battery_throughput_energy_MWh:.2f} MWh")
        print(f"Calculated Battery Utilization Ratio (Throughput / Capacity): {calculated_utilization_ratio:.4f}")
        print(f"Total Simulation Duration: {total_simulation_duration_days:.2f} days")

    # Return metrics needed for PSO objective function, including simulation duration in days
    return total_curtailment_energy_MWh, total_generator_import_energy, \
           calculated_utilization_ratio, total_simulation_duration_days, results_df

# 6. Analysis and Visualization
# MODIFICATION: Added battery_power_MW to the function signature
def analyze_and_plot(results_df, battery_capacity_MWh, battery_power_MW):
    # All energy sums now use time_step_hours = 1.0 implicitly due to hourly data
    total_load_energy = results_df['load'].sum()
    total_pv_energy = results_df['pv_gen'].sum()
    total_wind_energy = results_df['wind_gen'].sum()
    total_renewable_generation_MWh = total_pv_energy + total_wind_energy

    total_generator_import_energy = results_df['generator_power'].sum()

    total_battery_discharge_energy = results_df['battery_power'].apply(lambda x: max(0, x)).sum()
    total_battery_charge_energy = results_df['battery_power'].apply(lambda x: min(0, x)).sum()

    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum()

    print(f"\n--- Simulation Parameters ---")
    print(f"Optimized Battery Energy Capacity: {battery_capacity_MWh:.2f} MWh")
    print(f"Optimized Battery Power Rating: {battery_power_MW:.2f} MW") # MODIFICATION: Updated print statement

    print("\n--- Energy Balance ---")
    print(f"Total Load Energy: {total_load_energy:.2f} MWh")
    print(f"Total PV Energy: {total_pv_energy:.2f} MWh")
    print(f"Total Wind Energy: {total_wind_energy:.2f} MWh")
    print(f"Total Renewable Generation: {total_renewable_generation_MWh:.2f} MWh")
    print(f"Total Fossil fuel Energy: {total_generator_import_energy:.2f} MWh")
    print(f"Total Battery Discharge Energy: {total_battery_discharge_energy:.2f} MWh")
    print(f"Total Battery Charge Energy: {total_battery_charge_energy:.2f} MWh")
    print(f"Total Renewable Energy Wasted: {total_curtailment_energy_MWh:.2f} MWh")

    print("\n--- Battery Stats ---")
    print(f"Max SoC: {results_df['battery_soc'].max():.2f}")
    print(f"Min SoC: {results_df['battery_soc'].min():.2f}")

    print(f"\nMax Fossil fuel Power: {results_df['generator_power'].max():.2f} MW")

    percentage_renewable_wasted = 0
    if total_renewable_generation_MWh > 0:
        percentage_renewable_wasted = (total_curtailment_energy_MWh / total_renewable_generation_MWh) * 100
    print(f"Percentage of Renewable Energy Wasted: {percentage_renewable_wasted:.2f} %")

    percentage_renewable_utilization = 0
    if total_renewable_generation_MWh > 0:
        percentage_renewable_utilization = 100 - percentage_renewable_wasted
    print(f"Percentage of Renewable Energy Utilization: {percentage_renewable_utilization:.2f} %")

    renewable_energy_penetration = 0
    if total_load_energy > 0:
        renewable_energy_penetration = (total_pv_energy + total_wind_energy) / total_load_energy * 100
    print(f"Renewable Energy Penetration (Gross Generation): {renewable_energy_penetration:.2f} %")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(results_df['time'], results_df['load'], label='Load', color='blue')
    ax1.plot(results_df['time'], results_df['pv_gen'] + results_df['wind_gen'], label='Renewables', color='green')
    ax1.plot(results_df['time'], results_df['battery_power'], label='Battery Power (Discharge+ / Charge-)', color='purple')
    ax1.plot(results_df['time'], results_df['generator_power'], label='Fossil fuel power', color='red')
    ax1.plot(results_df['time'], results_df['curtailment_MW'], label='Curtailment', color='black', linestyle=':')
    ax1.set_xlabel('Time (minutes)') # Still label as minutes for consistency, but now represents hours * 60
    ax1.set_ylabel('Power (MW)')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(results_df['time'], results_df['battery_soc'], label='Battery SoC', color='orange')
    ax2.set_ylabel('Battery SoC')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.title("Power Balance and Battery SoC Over Time")
    plt.tight_layout()
    plt.show()
    print("Simulation Complete")

class BatteryOptimizer:
    def __init__(self, net, demand_ts, pv_ts, wind_ts, battery_params):
        self.net = net
        self.demand_ts = demand_ts
        self.pv_ts = pv_ts
        self.wind_ts = wind_ts
        self.battery_params = battery_params
        self.min_capacity = 1000.0  # MWh
        self.max_capacity = 650000.0 # MWh
        self.min_power_MW = 1000.0  # MW
        self.max_power_MW = 30000.0 # MW
        # Cycle constraints
        self.min_cycles_per_day = 1
        self.max_cycles_per_day = 1.5
        self.cycle_penalty =1e5
        # Utilization constraint
        self.min_utilization_threshold = 0.10 # 10%
        self.penalty_for_under_utilization = 1e6
        # Weights for objectives
        self.weight_curtailment = 0.5
        self.weight_grid_import = 0.5

        # NEW: Duration constraint for 4-hour storage
        self.target_duration_hours = 4.0
        self.duration_penalty_weight = 1e6 # Adjust this weight as needed for desired impact

    # MODIFICATION: evaluate_fitness now takes a 2-element array for [capacity, power]
    def evaluate_fitness(self, x):
        """Wrapper function for pyswarm's PSO, takes array input [capacity, power]"""
        battery_capacity_MWh = float(x[0])
        battery_power_MW = float(x[1])
        return self._fitness_function(battery_capacity_MWh, battery_power_MW)

    # MODIFICATION: _fitness_function now takes battery_capacity_MWh and battery_power_MW
    def _fitness_function(self, battery_capacity_MWh, battery_power_MW):
        """Core fitness calculation based on simulation results and penalties"""
        # Run simulation with both capacity and power
        curtailment, grid_import, utilization_ratio, sim_days, _ = run_simulation(
            self.net, self.demand_ts, self.pv_ts, self.wind_ts,
            battery_capacity_MWh, battery_power_MW, # Pass both variables
            **self.battery_params, is_pso_run=True)

        # Calculate cycles per day
        total_throughput_for_cycles = utilization_ratio * battery_capacity_MWh
        total_equivalent_cycles = 0
        if battery_capacity_MWh > 0: # Avoid division by zero
            total_equivalent_cycles = total_throughput_for_cycles / (2 * battery_capacity_MWh)

        cycles_per_day = 0
        if sim_days > 0: # Avoid division by zero
            cycles_per_day = total_equivalent_cycles / sim_days

        # Base fitness (minimize these objectives)
        fitness = (self.weight_curtailment * curtailment) + \
                  (self.weight_grid_import * grid_import)

        # 1. Penalty for battery under-utilization
        if utilization_ratio < self.min_utilization_threshold:
            fitness += self.penalty_for_under_utilization * (self.min_utilization_threshold - utilization_ratio)

        # 2. Penalty for low cycles per day or excessive cycles per day
        if cycles_per_day < self.min_cycles_per_day:
            fitness += self.cycle_penalty * (self.min_cycles_per_day - cycles_per_day)
        elif cycles_per_day > self.max_cycles_per_day:
            fitness += self.cycle_penalty * (cycles_per_day - self.max_cycles_per_day)

        # NEW: Penalty for deviation from target 4-hour duration
        current_duration_hours = 0
        if battery_power_MW > 0: # Avoid division by zero
            current_duration_hours = battery_capacity_MWh / battery_power_MW

        duration_deviation = abs(current_duration_hours - self.target_duration_hours)
        fitness += self.duration_penalty_weight * duration_deviation

        return fitness

    def optimize(self, n_particles, max_iter):
        """Run PSO optimization using pyswarm.pso"""
        swarmsize = n_particles
        omega = 0.7
        phip = 2.0
        phig = 2.0
        maxiter = max_iter
        minstep = 1e-8
        minfunc = 1e-8

        print("\n--- Starting pyswarm PSO Optimization (2D) ---")
        print(f"Searching Energy Capacity between {self.min_capacity:.2f} MWh and {self.max_capacity:.2f} MWh")
        print(f"Searching Power Rating between {self.min_power_MW:.2f} MW and {self.max_power_MW:.2f} MW")
        print(f"Constraining cycles to {self.min_cycles_per_day}-{self.max_cycles_per_day} cycles/day")
        print(f"Min utilization threshold: {self.min_utilization_threshold*100:.0f}%")
        print(f"Cycle penalty: {self.cycle_penalty:.0e}") # Print penalty in scientific notation
        print(f"Target duration: {self.target_duration_hours:.1f} hours with penalty {self.duration_penalty_weight:.0e}")


        # MODIFICATION: lb and ub are now 2-dimensional arrays
        xopt, fopt = pso(
            func=self.evaluate_fitness,
            lb=[self.min_capacity, self.min_power_MW],
            ub=[self.max_capacity, self.max_power_MW],
            swarmsize=swarmsize,
            omega=omega,
            phip=phip,
            phig=phig,
            maxiter=maxiter,
            minstep=minstep,
            minfunc=minfunc,
        )

        print("\n--- Optimization Complete ---")
        # MODIFICATION: xopt now contains two values
        print(f"Optimal Battery Energy Capacity: {xopt[0]:.2f} MWh")
        print(f"Optimal Battery Power Rating: {xopt[1]:.2f} MW")
        print(f"Minimum Fitness Value: {fopt:.2f}")

        return xopt[0], xopt[1] # Return both optimal capacity and power

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load data and create network
    print("Loading data and creating network...")
    timestamps, demand_ts, pv_ts, wind_ts = load_data()
    net, pv_index, wind_index, bess_index, grid_index = create_network()
    battery_params = {
        'battery_soc_max': 1.0,
        'battery_soc_min': 0.1,
        'battery_soc_init': 0,
        'battery_charge_eff': 0.8944,
        'battery_discharge_eff': 0.8944
    }

    # Create optimizer and run
    optimizer = BatteryOptimizer(net, demand_ts, pv_ts, wind_ts, battery_params)
    optimal_capacity, optimal_power = optimizer.optimize(n_particles=10, max_iter=1)
#-------------------------------------------------------------------------------
    # Run final simulation with optimal capacity and power
    print("\nRunning final detailed simulation with optimal capacity and power...")
    final_curtailment, final_grid_import, final_utilization, final_sim_days, final_results_df = run_simulation(
        net, demand_ts, pv_ts, wind_ts, optimal_capacity, optimal_power, # Pass both optimal values
        **battery_params, is_pso_run=False
    )
    throughput = final_utilization * optimal_capacity
    cycles_per_day = 0
    if final_sim_days > 0 and optimal_capacity > 0: # Avoid division by zero
        cycles_per_day = (throughput / (2 * optimal_capacity)) / final_sim_days
    print(f"\n--- Final Battery Cycling ---")
    print(f"Cycles per day: {cycles_per_day:.2f}")
    analyze_and_plot(final_results_df, optimal_capacity, optimal_power)