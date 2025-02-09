import os
import sys
import numpy as np
import math
import pandas as pd

from twelvedata import TDClient

from datetime import datetime

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)



import warnings

from functools import reduce

warnings.filterwarnings("ignore")



def get_bar_data(currency, bar_number=240, start_timestamp=-1, is_convert_to_time = True):
    # Initialize client - apikey parameter is requiered
    td = TDClient(apikey="dbc2c6a6a33840d4b2a11a371def5973")

    print("initial_bar_number = " + str(bar_number))
    # Construct the necessary time series
    ts = td.time_series(
        symbol=currency[:-3] + '/' + currency[-3:],
        interval="1h",
        outputsize=bar_number,
        timezone="Asia/Singapore",
    )

    # Returns pandas.DataFrame
    data_df = ts.as_pandas()

    data_df = data_df.iloc[::-1]

    data_df.reset_index(inplace=True)

    data_df = data_df.rename(columns = {'datetime' : 'time'})

    data_df['currency'] = currency

    data_df = data_df[['time', 'currency', 'open', 'high', 'low', 'close']]

    # print("Row number = " + str(data_df.shape[0]) + " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #
    print("here printing")
    print(data_df.iloc[-20:])

    return data_df

instrument = "ADAUSD"

run_execution = False

advanced_strategy = True

is_short = False

side = -1 if is_short else 1

out_folder = "C:\\Users\\admin\\CryptoTrading\\LeverageTrading"

initial_decision_file = os.path.join(out_folder, instrument + "_initial_decision" + ("_short" if side == -1 else "") + ".csv")
strategy_file = os.path.join(out_folder, instrument + "_strategy" + ("_short" if side == -1 else "") + ".csv")
execution_file = os.path.join(out_folder, instrument + "_execution" + ("_short" if side == -1 else "") + ".csv")

total_round = 5



#max_drawdown = 0.05 #0.05
max_drawdown = 0.05

#These two are constants, which never change for any instrument
#This is the key: In the second wave of a long trend, halve the profit rates, this will potentially increase the total profit rates
profit_rates = np.array([1.0, 0.5, 1.0, 0.5]) #Stop profit when making this percentage of profit vs actual notional (margin)

#move_stop_losses = np.array([0, 0, 0, 0])
move_stop_losses = np.array([1, 0, 1, 0])

#profit_rates = np.array([0.5, 0.25, 0.5, 0.25])

loss_rates = [0.5] * len(profit_rates)  #Always stop loss when losing half of the actual notional (margin)

entry_total_principal = 100


#ADAUSD
#entry_time = datetime(2024, 11, 7, 9, 0, 0)
#entry_time = datetime(2024, 11, 15, 7, 0, 0)

#DOGEUSD
entry_time = datetime(2024, 11, 9, 23, 0, 0)

if run_execution:
    data_df = get_bar_data(instrument, bar_number = 2000)

    data_df = data_df[data_df['time'] >= entry_time]

    print('data_df:')
    print(data_df.iloc[0:20])

    entry_price = data_df.iloc[0]['open']

else:
    #entry_price = 3.255
    #entry_price = 36.8

    entry_price = 0.705
    #entry_price = 0.3117

    #entry_price = 1
    #entry_price = 0.396
    #entry_price = 290

optimal_leverage = int(1.0/(max_drawdown*2))
#optimal_leverage = 10
half_optimal_leverage = int(optimal_leverage/2)

leverages = [optimal_leverage, optimal_leverage, half_optimal_leverage, half_optimal_leverage]



df = pd.DataFrame({"entry_price" : entry_price, "leverage" : leverages, "profit_rate" : profit_rates, "loss_rate" : loss_rates})

df['take_profit_pct'] = df['profit_rate'] / df['leverage']
df['take_loss_pct'] = df['loss_rate'] / df['leverage']

each_principal = entry_total_principal / len(leverages)
df['principal'] = each_principal


dfs = []

extra_principal = entry_total_principal

#extra_entry_amount = round(extra_principal * leverages[0] / entry_price, 3)
extra_entry_amount = int(extra_principal * leverages[0] / entry_price)


for theRound in range(total_round):

    print("")
    #print("Round " + str(theRound + 1) + '.........................')

    if theRound > 0:
        df['principal'] = df['principal_after_profit']
        df['entry_price'] = df['take_profit_price']

    df['entry_notional'] = df['principal'] * df['leverage']
    df['entry_amount'] = df['entry_notional'] / entry_price
    df['entry_amount'] = df['entry_amount'].astype(int)
    #df['entry_amount'] = df['entry_amount'].apply(lambda x: round(x, 3))

    df['take_profit_price'] = df['entry_price'] * (1 + side * df['take_profit_pct'])
    df['take_loss_price'] = df['entry_price'] * (1 - side * df['take_loss_pct'])

    df['move_stop_loss'] = move_stop_losses
    df['price_trigger_move_sl'] = np.where(
        df['move_stop_loss'] == 1,
        (df['entry_price'] + df['take_profit_price'])/2.0,
        0
    )


    for col in ['take_profit_price', 'take_loss_price', 'price_trigger_move_sl']:
        df[col] = df[col].apply(lambda x: round(x, 4))

    df['profit'] = df['principal'] * df['profit_rate']
    df['loss'] = df['principal'] * df['loss_rate']



    df['principal_after_profit'] = df['principal'] + df['profit']
    df['principal_after_loss'] = df['principal'] - df['loss']

    #for col in ['profit', 'loss', 'principal_after_profit', 'principal_after_loss', 'principal', 'entry_notional']:
    #    df[col] = df[col].astype(int)


    total_entry_amount = df['entry_amount'].sum()

    if theRound == 0:
        print("Initial trading decisions:")
        print(df)

        df.to_csv(initial_decision_file, index = False)

        print("total_entry_amount = " + str(total_entry_amount))
        print("Extra entry amount = " + str(extra_entry_amount))

        print("Final entry amount = " + str(round(total_entry_amount + extra_entry_amount, 3)))

    dfs += [df.copy()]

strategies = []

raw_strategy_dfs = []

for i in range(len(profit_rates)):

    strategy_df = pd.concat([df.iloc[i:(i + 1)] for df in dfs], axis=0)
    strategy_df.reset_index(inplace = True)
    strategy_df['index'] = list(range(1, strategy_df.shape[0] + 1))
    #strategy_df = strategy_df.drop(columns = ['index'])
    strategy_df = strategy_df.rename(columns = {'index' : 'round'})

    strategy_df['strategy'] = i+1

    strategy_df = strategy_df[['strategy'] + list(strategy_df.columns[0:-1])]

    dummy_df = pd.DataFrame(data = [[np.nan]*len(strategy_df.columns)], columns = strategy_df.columns)

    raw_strategy_dfs += [strategy_df]

    strategy_df = pd.concat([strategy_df, dummy_df])

    strategies += [strategy_df]


print("")
print("")
for i in range(len(strategies)):

    print("Strategy " + str(i+1))
    print(strategies[i].iloc[0:-1])

    print("")

all_strategies_df = pd.concat(strategies)



all_strategies_df.to_csv(strategy_file, index = False)

################################################

if not run_execution:
    sys.exit(0)

execution_columns = ['strategy', 'round', 'entry_time', 'entry_price', 'entry_principal', 'exit_time', 'exit_price', 'pnl']

execution_dfs = []

strategy_id = 0

print("Run execution")

if advanced_strategy:

    execution_columns += ['extra_principal']

    #if move_stop_losses.sum() > 0:
    execution_columns += ['price_trigger_move_sl', 'move_loss_time']

    current_round_ptrs = [0] * len(leverages)

    running_strategies = [1] * len(leverages)



    execution_data_list = [[]] * len(leverages)

    entry_times = [entry_time] * len(leverages)

    move_loss_times = [None] * len(leverages)

    principal_received_from_other_strategies = [0] * len(leverages)

    total_principal_from_stopped_strategies = 0
    principal_allocated_to_each_live_strategy = 0

    for i in range(data_df.shape[0]):

        bar_data = data_df.iloc[i]

        #has_running_strategies = False

        has_strategy_exit = False
        has_strategy_reach_profit = False

        for strategy_id in range(len(leverages)):

            if running_strategies[strategy_id] == 1:

                #if not has_running_strategies:
                 #   has_running_strategies = True

                this_entry_time = entry_times[strategy_id]
                this_strategy_df = raw_strategy_dfs[strategy_id]

                #print("read strategy df of strategy " + str(strategy_id))
                #print(this_strategy_df)

                current_round_ptr = current_round_ptrs[strategy_id]

                #print("current_round_ptr = " + str(current_round_ptr))

                if current_round_ptr < this_strategy_df.shape[0]:
                    this_round_data = this_strategy_df.iloc[current_round_ptr]

                    #print("take_profit_price = " + str(this_round_data['take_profit_price']))


                    if bar_data['high'] >= this_round_data['take_profit_price']:

                        has_strategy_reach_profit = True

                        this_round = this_round_data['round']
                        this_entry_price = this_round_data['entry_price']
                        this_exit_price = this_round_data['take_profit_price']
                        this_entry_value = this_round_data['principal']
                        pnl = this_round_data['profit']
                        this_exit_time = bar_data['time']


                        execution_data_list[strategy_id] = execution_data_list[strategy_id] + [[strategy_id, this_round, this_entry_time,
                                                                                                this_entry_price, this_entry_value, this_exit_time, this_exit_price, pnl,
                                                                                                principal_received_from_other_strategies[strategy_id], this_round_data['price_trigger_move_sl'], move_loss_times[strategy_id]]]
                        move_loss_times[strategy_id] = None

                        entry_times[strategy_id] = this_exit_time

                        current_round_ptrs[strategy_id] = current_round_ptrs[strategy_id] + 1

                        principal_received_from_other_strategies[strategy_id] = 0

                        print("")
                        print("At time " + str(this_exit_time) + ", strategy " + str(strategy_id) + " reaches next profit level")
                        print("principal_received_from_other_strategies = " + str(principal_allocated_to_each_live_strategy))

                        if principal_allocated_to_each_live_strategy > 0:
                            principal_received_from_other_strategies[strategy_id] = principal_allocated_to_each_live_strategy
                            total_principal_from_stopped_strategies -= principal_allocated_to_each_live_strategy

                            print("total_principal_from_stopped_strategies reduced to " + str(total_principal_from_stopped_strategies))

                            new_ptr = current_round_ptrs[strategy_id]

                            if new_ptr < this_strategy_df.shape[0]:
                                this_strategy_df = this_strategy_df.iloc[new_ptr:]
                                current_round_ptrs[strategy_id] = 0

                                old_principal = this_strategy_df.iloc[0]["principal"]
                                new_principal = old_principal + principal_allocated_to_each_live_strategy
                                multiplier = new_principal/old_principal

                                print("old_principal=" + str(old_principal) + ", new_principal=" + str(new_principal) + ", multiplier=" + str(multiplier))

                                for col in ['principal', 'entry_notional', 'entry_amount', 'profit', 'loss', 'principal_after_profit', 'principal_after_loss']:
                                    this_strategy_df[col] *=  multiplier

                                raw_strategy_dfs[strategy_id] = this_strategy_df

                                print("strategy df is updated to:")
                                print(this_strategy_df)


                    elif bar_data['low'] <= (this_round_data['take_loss_price'] if move_loss_times[strategy_id] is None else this_round_data['entry_price']):

                        has_strategy_exit = True

                        this_round = this_round_data['round']
                        this_entry_price = this_round_data['entry_price']
                        this_exit_price = this_round_data['take_loss_price'] if move_loss_times[strategy_id] is None else this_round_data['entry_price']
                        this_entry_value = this_round_data['principal']
                        pnl = -this_round_data['loss'] if move_loss_times[strategy_id] is None else 0
                        this_exit_time = bar_data['time']
                        execution_data_list[strategy_id] = execution_data_list[strategy_id] + [[strategy_id, this_round, this_entry_time, this_entry_price, this_entry_value,
                             this_exit_time, this_exit_price, pnl, principal_received_from_other_strategies[strategy_id], this_round_data['price_trigger_move_sl'], move_loss_times[strategy_id]]]

                        move_loss_times[strategy_id] = None

                        running_strategies[strategy_id] = 0

                        final_principal = this_entry_value + pnl

                        total_principal_from_stopped_strategies += final_principal

                        print("")
                        print("At time " + str(this_exit_time) + ", strategy " + str(strategy_id) + " exits, its left principal = " + str(final_principal) + " and total principal from stopped strategies = " + str(total_principal_from_stopped_strategies))


                        principal_received_from_other_strategies[strategy_id] = 0

                    elif move_stop_losses[strategy_id] == 1 and bar_data['high'] >= this_round_data['price_trigger_move_sl']:
                        move_loss_times[strategy_id] = bar_data['time']


        num_of_running_strategies = np.array(running_strategies).sum()

        if num_of_running_strategies == 0:
            break


        if (has_strategy_reach_profit or has_strategy_exit):

            print("")
            print("Now total_principal_from_stopped_strategies = " + str(total_principal_from_stopped_strategies))
            print("num running strategies = " + str(num_of_running_strategies))
            principal_allocated_to_each_live_strategy = int(total_principal_from_stopped_strategies / num_of_running_strategies)

            print("Number of strategies still running is " + str(num_of_running_strategies) + ", and principal allocated to each of them is " + str(principal_allocated_to_each_live_strategy))


    for execution_data in execution_data_list:
        print("execution_data:")
        print(execution_data)

        execution_df = pd.DataFrame(data=execution_data, columns=execution_columns)
        execution_dfs += [execution_df]

else:
    for strategy_df in raw_strategy_dfs:

        strategy_id += 1
        execution_data = []

        #print("")
        #print("")
        #print("strategy_id = " + str(strategy_id) + "............................")

        current_round = -1
        go_next_round = True
        this_entry_time = entry_time

        #print("strategy_df:")
        #print(strategy_df)

        for i in range(data_df.shape[0]):

            if go_next_round:
                current_round += 1

                # print("")
                # print("current_round = " + str(current_round))
                # print("strategy_df:")
                # print(strategy_df)
                # print("")

                this_round_data = strategy_df.iloc[current_round]
                go_next_round = False

            bar_data = data_df.iloc[i]

            if bar_data['high'] >= this_round_data['take_profit_price']:

                this_round = this_round_data['round']
                this_entry_price = this_round_data['entry_price']
                this_exit_price = this_round_data['take_profit_price']
                this_entry_value = this_round_data['principal']
                pnl = this_round_data['profit']
                this_exit_time = bar_data['time']
                execution_data += [[strategy_id, this_round, this_entry_time, this_entry_price, this_entry_value, this_exit_time, this_exit_price, pnl]]
                this_entry_time = this_exit_time

                #("exit_time = " + str(this_exit_time))
                #print("exit_price = " + str(this_exit_price))
                #print("pnl = " + str(pnl))

                go_next_round = True

            elif bar_data['low'] <= this_round_data['take_loss_price']:

                this_round = this_round_data['round']
                this_entry_price = this_round_data['entry_price']
                this_exit_price = this_round_data['take_loss_price']
                this_entry_value = this_round_data['principal']
                pnl = -this_round_data['loss']
                this_exit_time = bar_data['time']
                execution_data += [[strategy_id, this_round, this_entry_time, this_entry_price, this_entry_value, this_exit_time, this_exit_price, pnl]]

                #print("exit_time = " + str(this_exit_time))
                #print("exit_price = " + str(this_exit_price))
                #print("pnl = " + str(pnl))

                break


        execution_df = pd.DataFrame(data = execution_data, columns = execution_columns)

        #print("Finished.............")
        #print("execution_df:")
        #print(execution_df)

        execution_dfs += [execution_df]

print("")
print("")
print("Execution results:")

for execution_df in execution_dfs:
    print(execution_df)
    print("")



total_pnl = np.array([execution_df['pnl'].sum() for execution_df in execution_dfs]).sum()

return_rates = np.array([round(execution_df['pnl'].sum()/each_principal,2) for execution_df in execution_dfs])

print("total_pnl = " + str(total_pnl))

return_rate = round(total_pnl / entry_total_notional,2)

print("overall return_rate = " + str(return_rate))

print("Each strategy's return rate:")
print(return_rates)


final_execution_dfs = []
for execution_df in execution_dfs:

    dummy_df = pd.DataFrame(data=[[np.nan] * len(execution_df.columns)], columns=execution_df.columns)

    execution_df = pd.concat([execution_df, dummy_df])

    final_execution_dfs += [execution_df]

final_write_execution_df = pd.concat(final_execution_dfs)

final_write_execution_df.to_csv(execution_file, index = False)
















