# Import necessary libraries (already present)
import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso

# 1. Load Data (No changes needed here)
def load_data(filename='16-100%renewable.xlsx', sheet_name='Sheet1'):
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        data_columns = ['Electricity Consumption (MW)', 'PV Generation (MW)', 'Wind Generation (MW)']
        df_data = df[data_columns]
        df_hourly = df_data.groupby(np.arange(len(df_data)) // 4).mean()
        if 'Timestamp' in df.columns and len(df) > 0:
            try:
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

# 2. Create Network (No changes needed here)
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

# 3. Battery Parameters (No changes needed here)
def get_battery_params():
    battery_soc_max = 1.0
    battery_soc_min = 0.1
    battery_soc_init = 1
    battery_charge_eff = 0.6324
    battery_discharge_eff = 0.6324
    return battery_soc_max, battery_soc_min, battery_soc_init, battery_charge_eff, battery_discharge_eff

# 4. Time-Series Simulation (No changes needed here for functionality, only for print statements)
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
    time_step_hours = 1.0

    for t in range(len(demand_ts)):
        net.load.at[0, 'p_mw'] = demand_ts.iloc[t]
        net.sgen.at[pv_index, 'p_mw'] = pv_ts.iloc[t]
        net.sgen.at[wind_index, 'p_mw'] = wind_ts.iloc[t]

        max_charge_power_MW = battery_power_MW
        max_discharge_power_MW = battery_power_MW

        excess_generation = pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t]

        bess_dispatch_MW = 0
        if excess_generation > 0:
            bess_dispatch_MW = -min(excess_generation, max_charge_power_MW, (battery_soc_max - battery_soc) * battery_capacity_MWh)
        elif excess_generation < 0:
            bess_dispatch_MW = min(abs(excess_generation), max_discharge_power_MW, (battery_soc - battery_soc_min) * battery_capacity_MWh)

        net.sgen.at[bess_index, 'p_mw'] = bess_dispatch_MW

        curtailed_at_t = 0
        try:
            pp.runpp(net)
            generator_power = net.res_ext_grid.at[grid_index, 'p_mw']
            if generator_power < 0:
                curtailed_at_t = abs(generator_power)
                generator_power = 0
        except Exception as e:
            if not is_pso_run:
                print(f"Power flow did not converge at time step {t}: {e}")
            generator_power = 0
            curtailed_at_t = max(0, pv_ts.iloc[t] + wind_ts.iloc[t] - demand_ts.iloc[t] + net.sgen.at[bess_index, 'p_mw'])

        actual_bess_terminal_power_MW = net.sgen.at[bess_index, 'p_mw']
        energy_change_MWh = 0
        if actual_bess_terminal_power_MW > 0:
            energy_removed = (actual_bess_terminal_power_MW * time_step_hours) / battery_discharge_eff
            energy_change_MWh = -energy_removed
            battery_throughput_energy_MWh += actual_bess_terminal_power_MW * time_step_hours
        elif actual_bess_terminal_power_MW < 0:
            energy_stored = (abs(actual_bess_terminal_power_MW) * time_step_hours) * battery_charge_eff
            energy_change_MWh = energy_stored
            battery_throughput_energy_MWh += abs(actual_bess_terminal_power_MW) * time_step_hours

        battery_soc += energy_change_MWh / battery_capacity_MWh
        battery_soc = np.clip(battery_soc, battery_soc_min, battery_soc_max)

        results.append({
            'time': t * 60,
            'load': demand_ts.iloc[t],
            'pv_gen': pv_ts.iloc[t],
            'wind_gen': wind_ts.iloc[t],
            'battery_power': actual_bess_terminal_power_MW,
            'battery_soc': battery_soc,
            'generator_power': generator_power,
            'curtailment_MW': curtailed_at_t
        })

    results_df = pd.DataFrame(results)

    total_curtailment_energy_MWh = results_df['curtailment_MW'].sum() * time_step_hours
    total_generator_import_energy = results_df['generator_power'].sum() * time_step_hours
    calculated_utilization_ratio = battery_throughput_energy_MWh / battery_capacity_MWh

    total_simulation_duration_hours = len(demand_ts) * time_step_hours
    total_simulation_duration_days = total_simulation_duration_hours / 24

    if not is_pso_run:
        print(f"Battery Throughput Energy: {battery_throughput_energy_MWh:.2f} MWh")
        print(f"Calculated Battery Utilization Ratio (Throughput / Capacity): {calculated_utilization_ratio:.4f}")
        print(f"Total Simulation Duration: {total_simulation_duration_days:.2f} days")

    return total_curtailment_energy_MWh, total_generator_import_energy, \
           calculated_utilization_ratio, total_simulation_duration_days, results_df

# 6. Analysis and Visualization (No changes needed here)
def analyze_and_plot(results_df, battery_capacity_MWh, battery_power_MW):
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
    print(f"Optimized Battery Power Rating: {battery_power_MW:.2f} MW")

    print("\n--- Energy Balance ---")
    print(f"Total Load Energy: {total_load_energy:.2f} MWh")
    print(f"Total PV Energy: {total_pv_energy:.2f} MWh")
    print(f"Total Wind Energy: {total_wind_energy:.2f} MWh")
    print(f"Total Renewable Generation: {total_renewable_generation_MWh:.2f} MWh")
    print(f"Total Fossil fuel Energy: {total_generator_import_energy:.2f} MWh")
    print(f"Total Battery Discharge Energy: {total_battery_discharge_energy:.2f} MWh")
    print(f"Total Battery Charge Energy: {total_battery_charge_energy:.2f} MWh")
    print(f"Total Curtailment Energy: {total_curtailment_energy_MWh:.2f} MWh")

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
    ax1.set_xlabel('Time (minutes)')
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

        # Define bounds for long-term storage
      
        self.min_capacity = 1000.0  
        self.max_capacity = 14600000.0 
        self.min_power_MW = 17000.0 
        self.max_power_MW = 30000.0  

        # Removed: Cycle constraints
        # self.min_cycles_per_day = 0.22
        # self.max_cycles_per_day = 0.24
        # self.cycle_penalty = 5e7

        # Utilization constraint (kept as it promotes battery usage)
        self.min_utilization_threshold = 0.05 
        self.penalty_for_under_utilization = 1e6

        # Weights for objectives (cost of curtailment/grid import)
        self.weight_curtailment = 80000
        self.weight_grid_import = 600000

        # Removed: Duration constraint for 4-hour storage
        # self.target_duration_hours = 4.0
        # self.duration_penalty_weight = 1e6

        # NEW: Capital Cost Parameters for Long-Term Storage (as specified by user)
        self.cost_per_mwh = 5000         # Cost per MWh (e.g., for hydrogen storage infrastructure)
        self.cost_per_mw = 1750000     # Cost per MW (e.g., for electrolyzer + reconversion unit)
        self.cost_penalty_weight = 1.0   # Weight for the cost in the fitness function

    def evaluate_fitness(self, x):
        """Wrapper function for pyswarm's PSO, takes array input [capacity, power]"""
        battery_capacity_MWh = float(x[0])
        battery_power_MW = float(x[1])
        return self._fitness_function(battery_capacity_MWh, battery_power_MW)

    def _fitness_function(self, battery_capacity_MWh, battery_power_MW):
        """Core fitness calculation based on simulation results and penalties"""
        curtailment, grid_import, utilization_ratio, sim_days, _ = run_simulation(
            self.net, self.demand_ts, self.pv_ts, self.wind_ts,
            battery_capacity_MWh, battery_power_MW,
            **self.battery_params, is_pso_run=True)

        # Calculate cycles per day (still calculated but not penalized)
        total_throughput_for_cycles = utilization_ratio * battery_capacity_MWh
        total_equivalent_cycles = 0
        if battery_capacity_MWh > 0:
            total_equivalent_cycles = total_throughput_for_cycles / (2 * battery_capacity_MWh)
        cycles_per_day = 0
        if sim_days > 0:
            cycles_per_day = total_equivalent_cycles / sim_days


        # Base fitness (minimize curtailment and grid import)
        fitness = (self.weight_curtailment * curtailment) + \
                  (self.weight_grid_import * grid_import)

        # NEW: Add Capital Cost to the fitness function
        # This will penalize larger systems based on their CAPEX
        fitness += self.cost_penalty_weight * (
            battery_capacity_MWh * self.cost_per_mwh +
            battery_power_MW * self.cost_per_mw
        )

        # 1. Penalty for battery under-utilization (kept)
        if utilization_ratio < self.min_utilization_threshold:
            fitness += self.penalty_for_under_utilization * (self.min_utilization_threshold - utilization_ratio)

        # Removed: Penalty for low cycles per day or excessive cycles per day
        # if cycles_per_day < self.min_cycles_per_day:
        #     fitness += self.cycle_penalty * (self.min_cycles_per_day - cycles_per_day)
        # elif cycles_per_day > self.max_cycles_per_day:
        #     fitness += self.cycle_penalty * (cycles_per_day - self.max_cycles_per_day)

        # Removed: Penalty for deviation from target 4-hour duration
        # current_duration_hours = 0
        # if battery_power_MW > 0:
        #     current_duration_hours = battery_capacity_MWh / battery_power_MW
        # duration_deviation = abs(current_duration_hours - self.target_duration_hours)
        # fitness += self.duration_penalty_weight * duration_deviation

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

        print("\n--- Starting pyswarm PSO Optimization (2D) for Long-Term Storage ---")
        print(f"Searching Energy Capacity between {self.min_capacity:.2f} MWh and {self.max_capacity:.2f} MWh")
        print(f"Searching Power Rating between {self.min_power_MW:.2f} MW and {self.max_power_MW:.2f} MW")
        print(f"Cost per MWh: {self.cost_per_mwh} EUR/MWh")
        print(f"Cost per MW: {self.cost_per_mw} EUR/MW")
        # Removed print statements for cycle and duration constraints

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
        print(f"Optimal Battery Energy Capacity: {xopt[0]:.2f} MWh")
        print(f"Optimal Battery Power Rating: {xopt[1]:.2f} MW")
        print(f"Minimum Fitness Value: {fopt:.2f}")

        return xopt[0], xopt[1]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load data and create network
    print("Loading data and creating network...")
    timestamps, demand_ts, pv_ts, wind_ts = load_data()
    net, pv_index, wind_index, bess_index, grid_index = create_network()
    battery_params = {
        'battery_soc_max': 1.0,
        'battery_soc_min': 0.1,
        'battery_soc_init': 1,
        'battery_charge_eff': 0.6324,
        'battery_discharge_eff': 0.6324
    }
    # Create optimizer and run
    optimizer = BatteryOptimizer(net, demand_ts, pv_ts, wind_ts, battery_params)
    # n_particles and max_iter should be adjusted based on desired convergence and computational time
    # For initial testing, you might use lower values (e.g., n_particles=10, max_iter=5)
    # For more robust optimization, consider higher values (e.g., n_particles=50, max_iter=100+)
    optimal_capacity, optimal_power = optimizer.optimize(n_particles=1, max_iter=1) # Increased max_iter for better convergence
#-------------------------------------------------------------------------------
    # Run final simulation with optimal capacity and power
    print("\nRunning final detailed simulation with optimal capacity and power...")
    final_curtailment, final_grid_import, final_utilization, final_sim_days, final_results_df = run_simulation(
        net, demand_ts, pv_ts, wind_ts, optimal_capacity, optimal_power,
        **battery_params, is_pso_run=False
    )
    throughput = final_utilization * optimal_capacity
    cycles_per_day = 0
    if final_sim_days > 0 and optimal_capacity > 0:
        cycles_per_day = (throughput / (2 * optimal_capacity)) / final_sim_days
    print(f"\n--- Final Battery Cycling ---")
    print(f"Cycles per day: {cycles_per_day:.2f}")

    # Calculate and print total capital cost
    total_capital_cost = (optimal_capacity * optimizer.cost_per_mwh) + \
                         (optimal_power * optimizer.cost_per_mw)
    print(f"Total Estimated Capital Cost for Optimized System: {total_capital_cost:,.2f} EUR")

    analyze_and_plot(final_results_df, optimal_capacity, optimal_power)