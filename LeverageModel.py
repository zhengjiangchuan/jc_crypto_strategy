import os
import sys
import numpy as np
import math
import pandas as pd

from twelvedata import TDClient





##Input
profit_pct = 1 #This is a constant


leverage = 10

market_ret_needed_to_close = profit_pct/leverage

print("market_ret_needed_to_close = " + str(market_ret_needed_to_close))

drawdown_to_force_out = 1.0/(leverage*2)
print("drawdown_to_force_out = " + str(drawdown_to_force_out))

#market_df = pd.DataFrame({"return": [0.58, 0.35, 0.38, 0.19], "drawdown" : [0.2, 0.2, 0.2, 0.25]})

#BTC
#market_df = pd.DataFrame({"return": [0.08, 0.29, 0.096, 0.075, 0.087],
#                          "drawdown" : [0.08, 0.07, 0.08, 0.1, 0.13]})

#ADA
market_df = pd.DataFrame({"return": [0.58, 0.35, 0.38, 0.19],
                          "drawdown" : [0.13, 0.1, 0.2, 0.27]})

columns = ['round_id', 'start_asset', 'end_asset', 'close_position_times', 'total_return', 'waves']
final_result = []

max_round = market_df.shape[0]

round_id = 0
i = -1
start_asset = 1

i += 1
round_id += 1

end_asset = 1
while i < max_round:



    finish_one_round = False

    #print("Debug i = " + str(i) + " row_num=" + str(market_df.shape[0]))
    data = market_df.iloc[i]

    actual_market_return = data['return']

    print("")
    print("")
    print("round_id=" + str(round_id) + " i=" + str(i) + ' actual_market_ret=' + str(round(actual_market_return,2)))

    start_asset = end_asset

    waves = str(i + 1)

    while not finish_one_round:



        profit_times = int(math.log(1+actual_market_return)/math.log(1+market_ret_needed_to_close))

        total_market_return_in_profit_period = math.pow(1+market_ret_needed_to_close, profit_times) - 1

        #remaining_market_return = actual_market_return - total_market_return_in_profit_period

        remaining_market_return = (1 + actual_market_return)/(1 + total_market_return_in_profit_period) - 1

        assert(remaining_market_return < market_ret_needed_to_close)

        actual_draw_down = (1 + remaining_market_return) * (1 - data['drawdown']) - 1

        print("        actual_market_return=" + str('%.2f' % actual_market_return) + " profit_times=" + str(profit_times) +
              " total_market_return_in_profit_period=" + str('%.2f' % total_market_return_in_profit_period) +
        " remaining_market_return=" + str('%.2f' % remaining_market_return) + " actual_draw_down=" + str('%.2f' % actual_draw_down))


        if actual_draw_down < -drawdown_to_force_out:

            print("        Force close position!")

            total_return = math.pow(1 + profit_pct, profit_times) / 2.0 - 1  # /2 means the last force closing position due to market draw down which finishes this round of profiting

            end_asset = start_asset * (1 + total_return)

            final_result += [[round_id, start_asset, end_asset, profit_times + 1, total_return, waves]]

            break

        else:

            print("")
            print("        NOT Force close position, continue!")
            i += 1
            print("        i=" + str(i))

            if i >= max_round:
                total_return = math.pow(1 + profit_pct, profit_times) - 1
                end_asset = start_asset * (1 + total_return)
                final_result += [[round_id, start_asset, end_asset, profit_times, total_return, waves]]
                print("temp final_result:")
                print(final_result)
                break
            else:
                current_drawdown = data['drawdown']
                data = market_df.iloc[i]
                actual_market_return = (1 + actual_market_return) * (1 - current_drawdown) * (1 + data['return']) - 1
                print("before waves = " + waves)
                waves = waves + "," + str(i+1)
                print("after waves = " + waves)

    i += 1
    round_id += 1


final_df = pd.DataFrame(data = final_result, columns = columns)

print("final_df")
print(final_df)









