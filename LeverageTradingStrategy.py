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

instrument = "ADAUSD"

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


out_folder = "C:\\Users\\admin\\CryptoTrading\\LeverageTrading"
initial_decision_file = os.path.join(out_folder, "initial_decision.csv")
strategy_file = os.path.join(out_folder, "strategy.csv")
execution_file = os.path.join(out_folder, "execution.csv")

total_round = 30

max_drawdown = 0.05

#These two are constants, which never change for any instrument
profit_rates = [1.0, 0.5, 1.0, 0.5] #Stop profit when making this percentage of profit vs actual notional (margin)
loss_rates = [0.5] * len(profit_rates)  #Always stop loss when losing half of the actual notional (margin)

entry_total_notional = 1000



entry_time = datetime(2024, 11, 7, 9, 0, 0)
data_df = get_bar_data(instrument, bar_number = 2000)

data_df = data_df[data_df['time'] >= entry_time]

print('data_df:')
print(data_df.iloc[0:20])

entry_price = data_df.iloc[0]['open']


#entry_price = 1.09


optimal_leverage = int(1.0/(max_drawdown*2))
half_optimal_leverage = int(optimal_leverage/2)

leverages = [optimal_leverage, optimal_leverage, half_optimal_leverage, half_optimal_leverage]



df = pd.DataFrame({"entry_price" : entry_price, "leverage" : leverages, "profit_rate" : profit_rates, "loss_rate" : loss_rates})

df['take_profit_pct'] = df['profit_rate'] / df['leverage']
df['take_loss_pct'] = df['loss_rate'] / df['leverage']

df['principal'] = entry_total_notional / len(leverages)


dfs = []


for theRound in range(total_round):

    print("")
    #print("Round " + str(theRound + 1) + '.........................')

    if theRound > 0:
        df['principal'] = df['principal_after_profit']
        df['entry_price'] = df['take_profit_price']

    df['entry_notional'] = df['principal'] * df['leverage']
    df['entry_amount'] = df['entry_notional'] / entry_price
    df['entry_amount'] = df['entry_amount'].astype(int)

    df['take_profit_price'] = df['entry_price'] * (1 + df['take_profit_pct'])
    df['take_loss_price'] = df['entry_price'] * (1 - df['take_loss_pct'])


    for col in ['take_profit_price', 'take_loss_price']:
        df[col] = df[col].apply(lambda x: round(x, 4))

    df['profit'] = df['principal'] * df['profit_rate']
    df['loss'] = df['principal'] * df['loss_rate']



    df['principal_after_profit'] = df['principal'] + df['profit']
    df['principal_after_loss'] = df['principal'] - df['loss']

    for col in ['profit', 'loss', 'principal_after_profit', 'principal_after_loss', 'principal', 'entry_notional']:
        df[col] = df[col].astype(int)


    total_entry_amount = df['entry_amount'].sum()

    if theRound == 0:
        print("Initial trading decisions:")
        print(df)

        df.to_csv(initial_decision_file, index = False)

        print("total_entry_amount = " + str(total_entry_amount))

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

execution_columns = ['strategy', 'round', 'entry_time', 'entry_price', 'entry_margin', 'exit_time', 'exit_price', 'pnl']

execution_dfs = []

strategy_id = 0

print("Run execution")
for strategy_df in raw_strategy_dfs:

    strategy_id += 1
    execution_data = []

    print("")
    print("")
    print("strategy_id = " + str(strategy_id) + "............................")

    current_round = -1
    go_next_round = True
    this_entry_time = entry_time

    #print("strategy_df:")
    #print(strategy_df)

    for i in range(data_df.shape[0]):

        if go_next_round:
            current_round += 1

            print("")
            print("current_round = " + str(current_round))
            print("strategy_df:")
            print(strategy_df)
            print("")

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

            print("exit_time = " + str(this_exit_time))
            print("exit_price = " + str(this_exit_price))
            print("pnl = " + str(pnl))

            go_next_round = True

        elif bar_data['low'] <= this_round_data['take_loss_price']:

            this_round = this_round_data['round']
            this_entry_price = this_round_data['entry_price']
            this_exit_price = this_round_data['take_loss_price']
            this_entry_value = this_round_data['principal']
            pnl = -this_round_data['loss']
            this_exit_time = bar_data['time']
            execution_data += [[strategy_id, this_round, this_entry_time, this_entry_price, this_entry_value, this_exit_time, this_exit_price, pnl]]

            print("exit_time = " + str(this_exit_time))
            print("exit_price = " + str(this_exit_price))
            print("pnl = " + str(pnl))

            break


    execution_df = pd.DataFrame(data = execution_data, columns = execution_columns)

    print("Finished.............")
    print("execution_df:")
    print(execution_df)

    execution_dfs += [execution_df]

print("Execution results:")

for execution_df in execution_dfs:
    print(execution_df)



total_pnl = np.array([execution_df['pnl'].sum() for execution_df in execution_dfs]).sum()

print("total_pnl = " + str(total_pnl))

return_rate = round(total_pnl / entry_total_notional,2)

print("return_rate = " + str(return_rate))


final_execution_dfs = []
for execution_df in execution_dfs:

    dummy_df = pd.DataFrame(data=[[np.nan] * len(execution_df.columns)], columns=execution_df.columns)

    execution_df = pd.concat([execution_df, dummy_df])

    final_execution_dfs += [execution_df]

final_write_execution_df = pd.concat(final_execution_dfs)

final_write_execution_df.to_csv(execution_file, index = False)
















