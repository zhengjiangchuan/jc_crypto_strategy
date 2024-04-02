import numpy as np
import scipy as scp
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import warnings

from functools import reduce

warnings.filterwarnings("ignore")

# Portfolio Construction

start_date = datetime(2023, 4, 1)  # 4.1

forex_dir = "C:\\Users\\admin\\JCForex_prod"
#root_dir = "C:\\Users\\admin\\JCForex_prod\\portfolio_construction_reversalStrategy_duration1_ambiguous_prod_vegasFilterWeakerStronger_noDurationThreshold_rmCond7_relaxReqBelowVegas"

root_dir = "C:\\Users\\admin\\JCForex_prod\\portfolio_construction_TrendFollowingStrategy_allCurrency_V1"


if not os.path.exists(root_dir):
    os.makedirs(root_dir)

def which(bool_array):

    a = np.arange(len(bool_array))
    return a[bool_array]

def calc_max_drawdown(x):
    df = pd.DataFrame({'cum_pnl': x})
    df['max_cum_pnl'] = df['cum_pnl'].cummax()
    df['draw_down'] = df['max_cum_pnl'] - df['cum_pnl']

    max_draw_down = df['draw_down'].max()
    end = df['draw_down'].argmax()



    start = which(df['cum_pnl'] == df.iloc[end]['max_cum_pnl'])[0]

    return (max_draw_down, start, end)

def preprocess_time(t): #t is string

    if t[0] == "\'":
        t = t[1:]

    return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")



def construct_portfolio_for_end_date(end_date, start_date = datetime(2023, 4, 1)) :

    print("")
    print("Constructing Optimal Portfolio for end_date = " + start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d"))

    currency_df = pd.read_csv(os.path.join(forex_dir, "currency.csv"))

    currency_list = currency_df['currency'].tolist()
    #currency_list = currency_list[0:2]

    print("")
    print("Calculate performance for each currency............." + start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d"))
    summary_df = calculate_currency_performance(end_date, currency_list, sorted=False, accumulated_mode=False, start_date = start_date)

    sorted_currency_list = summary_df.iloc[:-1]['currency'].tolist()

    currencies_with_no_data = [currency for currency in currency_list if currency != "All" and currency not in sorted_currency_list]

    print("Currencies with no data: " + str(currencies_with_no_data))

    print("")
    print("Calculate performance again for each of sorted currencies" + start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d"))
    summary_df = calculate_currency_performance(end_date, sorted_currency_list, sorted=True, accumulated_mode=False, start_date = start_date)

    print("")
    print("Calculate performance for accumulated currencies..........." + start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d"))
    summary_df = calculate_currency_performance(end_date, sorted_currency_list, sorted=True, accumulated_mode=True, start_date = start_date)

    max_pnl_id = summary_df['last_cum_pnl'].argmax()

    optimal_currency_list = summary_df.iloc[:(max_pnl_id+1)]['currency'].tolist()

    optimal_currency_list = [currency[len('Until '):] for currency in optimal_currency_list]



    return optimal_currency_list + currencies_with_no_data



def calculate_currency_performance(end_date, currency_list, sorted, accumulated_mode, start_date = datetime(2023, 4, 1)):

    init_deposit = 8000  # 25000

    commission_rate = 28.17 * 2

    consider_cost = True

    use_fewer_trades = False

    if not accumulated_mode:
        currency_list += ['All']

    #print("currency_list = ")
    #print(currency_list)


    if accumulated_mode:
        final_output_folder = os.path.join(root_dir,
                                           "all_combined_accumulated_pnl_chart_" + start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d"))
    else:
        final_output_folder = os.path.join(root_dir,
                                           "all_combined_pnl_chart_" + start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d"))

    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    summary_data = []
    summary_columns = ['currency', 'last_cum_pnl', 'max_drawdown', 'max_drawdown_startid', 'max_drawdown_endid',
                       'adj_return', 'return_rate', 'drawdown_rate', 'trading_days', 'trade_num', 'trades_per_day']

    # This is even better settings for all currencies, and also even better for larger set of selected currencies (with GBPJPY EURCHF added)
    # trade_files = [os.path.join(forex_dir,
    #                             "all_pnl_chart_ratio1removeMustReject3_noSmartClose_macd_0204_notExceedGuppy3_relaxFastSlow_rejectLongTrend_simple\\all_trades.csv"),
    #                os.path.join(forex_dir,
    #                             "all_pnl_chart_ratio10removeMustReject3_noSmartClose_macd_0204_notExceedGuppy3_relaxFastSlow_rejectLongTrend_simple\\all_trades.csv")]

    # trade_files = [os.path.join(forex_dir,
    #                             "all_pnl_chart_ratio1ReversalStrategy_3_currencies2_duration1_ambiguous_prod_vegasFilterWeakerStronger_noDurationThreshold_rmCond7_relaxReqBelowVegas_t\\all_trades.csv"),
    #                os.path.join(forex_dir,
    #                             "all_pnl_chart_ratio10ReversalStrategy_3_currencies2_duration1_ambiguous_prod_vegasFilterWeakerStronger_noDurationThreshold_rmCond7_relaxReqBelowVegas_t\\all_trades.csv")]

    trade_files = [os.path.join(forex_dir,
                                "all_pnl_chart_ratio1TrendFollowingStrategy_allCurrency_V1\\all_trades.csv"),
                   os.path.join(forex_dir,
                                "all_pnl_chart_ratio10TrendFollowingStrategy_allCurrency_V1\\all_trades.csv")]

    # output_file = os.path.join(root_dir,
    #                            "all_pnl_chart_ratio10RemoveFucking2_variant10_new_filter_prod_all_1115_removeMustReject3_noSmartClose_macd_result_all.csv")
    # temp_output_file = os.path.join(root_dir,
    #                                 "all_pnl_chart_ratio10RemoveFucking2_variant10_new_filter_prod_all_1115_removeMustReject3_noSmartClose_macd_result_all_temp.csv")

    raw_trade_dfs = []

    #currencies_with_no_data = []

    for trade_file in trade_files:
        raw_trade_dfs += [pd.read_csv(trade_file)]

    for i in range(0, len(currency_list)):

        currency = currency_list[i]

        print("Process Currency " + currency + "...................." + start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d"))

        if accumulated_mode:
            selected_currencies = currency_list[0:(i + 1)]
        else:
            selected_currencies = None if currency in [None, "All"] else [currency]

        removed_currencies = None

        trade_dfs = []

        for raw_trade_df in raw_trade_dfs:

            trade_df = raw_trade_df.copy()

            trade_df = trade_df[trade_df['is_win'] != -1]

            if selected_currencies is not None:
                trade_df = trade_df[trade_df['currency'].isin(selected_currencies)]

            if removed_currencies is not None:
                trade_df = trade_df[~trade_df['currency'].isin(removed_currencies)]

            # print("trade df length = " + str(trade_df.shape[0]))

            for col in ['entry_time', 'exit_time']:
                trade_df[col] = trade_df[col].apply(lambda x: preprocess_time(x))

            #trade_df = trade_df[(trade_df['entry_time'] >= start_date) & (trade_df['exit_time'] < end_date)]

            trade_df = trade_df[(trade_df['exit_time'] >= start_date) & (trade_df['exit_time'] < end_date)]

            trade_df.reset_index(inplace=True)
            trade_df = trade_df.drop(columns=['index'])

            # trade_df['profit_loss_ratio'] = profit_loss_ratio

            trade_dfs += [trade_df]

        trade_df_small = trade_dfs[0]

        if len(trade_dfs) > 1:
            trade_df_large = trade_dfs[1]

            if use_fewer_trades:
                trade_df_small['sid'] = list(range(trade_df_small.shape[0]))
                trade_df_large['lid'] = list(range(trade_df_large.shape[0]))

                merged_df = pd.merge(trade_df_large[['lid', 'currency', 'side', 'entry_time']],
                                     trade_df_small[['sid', 'currency', 'side', 'entry_time']],
                                     on=['currency', 'side', 'entry_time'], how='left'
                                     )

                trade_df_small = trade_df_small[trade_df_small['sid'].isin(merged_df['sid'])]

                trade_df_small = trade_df_small.drop(columns=['sid'])
                trade_df_large = trade_df_large.drop(columns=['lid'])

            trade_df = pd.concat([trade_df_small,
                                  trade_df_large])  ############################################################ Choose to use one dataframe or two ##############################



        else:
            trade_df = trade_df_small


        if trade_df.shape[0] == 0:
            continue

        # if cutoff_end_date is not None:
        #     # cutoff_trade_df = trade_df[(trade_df['exit_time'] >= start_date) & (trade_df['exit_time'] < cutoff_end_date)]
        #
        #     cutoff_trade_df = trade_df[(trade_df['entry_time'] >= start_date) & (trade_df['exit_time'] < cutoff_end_date)]
        #     cutoff_trade_num = cutoff_trade_df.shape[0]

        #print("trade_df length = " + str(trade_df.shape[0]))

        trade_df = trade_df.sort_values(by=['exit_time', 'currency'])

        # trade_df = trade_df.iloc[99:] #######################//Extract from max draw down start time ***************************************************

        # trade_df = trade_df.iloc[28:] #####Temp #################################################################

        trade_df['id'] = list(range(trade_df.shape[0]))
        # trade_df['pnl'] = np.where(trade_df['is_win'] == 1, trade_df['profit_loss_ratio'],
        #                           np.where(trade_df['is_win'] == -1, 0, -1))

        # trade_df['pnl'] = trade_df['pnl']/2.0 ######################################################################################
        trade_df['cum_pnl'] = trade_df['pnl'].cumsum()

        trade_df.reset_index(inplace=True)
        trade_df = trade_df.drop(columns=['index'])

        for price_col in ['entry_price', 'exit_price']:
            trade_df[price_col] = np.where(
                trade_df['currency'].apply(lambda x: 'JPY' in x),
                trade_df[price_col].apply(lambda x: str(round(x * 1000.0) / 1000.0)),
                trade_df[price_col].apply(lambda x: str(round(x * 100000.0) / 100000.0))
            )

        trade_df['return_rate'] = trade_df['cum_pnl'] / init_deposit
        trade_df['return_rate'] = trade_df['return_rate'].apply(lambda x: str(int(round(x * 100, 0))) + "%")

        if 'entry_com_discount' in trade_df.columns:

            trade_df['adj_pnl'] = trade_df['pnl'] - (
                        commission_rate * trade_df['position'] - commission_rate * trade_df['position'] * (
                            trade_df['entry_com_discount'] + trade_df['exit_com_discount']) / 2.0)

        else:

            trade_df['adj_pnl'] = trade_df['pnl'] - commission_rate * trade_df['position']

        trade_df['adj_cum_pnl'] = trade_df['adj_pnl'].cumsum()

        # print("Fuck Bug")
        # display(trade_df.head(20))

        trade_df['adj_return_rate'] = trade_df['adj_cum_pnl'] / init_deposit
        trade_df['adj_return_rate'] = trade_df['adj_return_rate'].apply(lambda x: str(int(round(x * 100, 0))) + "%")

        if consider_cost:
            trade_df['unadj_pnl'] = trade_df['pnl']
            trade_df['unadj_cum_pnl'] = trade_df['cum_pnl']
            trade_df['unadj_return_rate'] = trade_df['return_rate']

            trade_df['pnl'] = trade_df['adj_pnl']
            trade_df['cum_pnl'] = trade_df['adj_cum_pnl']
            trade_df['return_rate'] = trade_df['adj_return_rate']

            trade_df = trade_df.drop(columns=['adj_pnl', 'adj_cum_pnl', 'adj_return_rate'])

        # print("Display here*******************************************************************************")
        # display(trade_df.iloc[:20])
        trade_df_copy = trade_df.copy()

        # trade_df_copy = trade_df_copy.sort_values(by = ['entry_time'])

        trade_df_copy = trade_df_copy.sort_values(by=['exit_time', 'currency'])
        trade_df_copy['id'] = list(range(trade_df_copy.shape[0]))
        # display(trade_df_copy[trade_df_copy['tp_num'].isnull()])

        # trade_df_copy = trade_df_copy[trade_df_copy['tp_num'].isnull()] #############################*************************************************************************************
        trade_df_copy.reset_index(inplace=True)
        trade_df_copy = trade_df_copy.drop(columns=['index'])
        # display(trade_df_copy.iloc[-51:]) ####################################################**********************************************************************************************************************************************

        #trade_df_copy.to_csv(output_file, index=False)

        entry_df = trade_df[['currency', 'side', 'position', 'id', 'entry_time']]
        entry_df = entry_df.rename(columns={
            'entry_time': 'time'
        })
        entry_df['is_entry'] = True

        exit_df = trade_df[['currency', 'side', 'position', 'id', 'exit_time']]
        exit_df = exit_df.rename(columns={
            'exit_time': 'time'
        })
        exit_df['is_entry'] = False

        temp_df = pd.concat([entry_df, exit_df])

        temp_df = temp_df.sort_values(by=['time'])

        cum_margin = 0
        cum_pnl = 0

        cum_margins = []
        cum_pnls = []

        delta_margins = []
        delta_pnls = []

        for i in range(temp_df.shape[0]):

            delta_margin = 0
            delta_pnl = 0

            data = temp_df.iloc[i]
            if data['is_entry']:
                delta_margin = trade_df.iloc[data['id']]['margin']
                cum_margin += delta_margin
            else:
                delta_pnl = trade_df.iloc[data['id']]['pnl']
                cum_pnl += delta_pnl

                delta_margin = -trade_df.iloc[data['id']]['margin']
                cum_margin += delta_margin

            delta_margins += [delta_margin]
            delta_pnls += [delta_pnl]

            cum_margins += [cum_margin]
            cum_pnls += [cum_pnl]

        temp_df['delta_margin'] = delta_margins
        temp_df['cum_margin'] = cum_margins

        temp_df['cum_margin'] = temp_df['cum_margin'].apply(lambda x: round(x, 2))

        temp_df['delta_pnl'] = delta_pnls
        temp_df['cum_pnl'] = cum_pnls

        temp_df['equity'] = temp_df['cum_pnl'] + init_deposit

        temp_df['margin_level'] = temp_df['equity'] / temp_df['cum_margin']

        temp_df['margin_level'] = np.where(
            temp_df['cum_margin'] == 0,
            np.inf,
            temp_df['margin_level']
        )

        temp_df['index'] = list(range(temp_df.shape[0]))

        temp_df['add_currency'] = np.where(
            temp_df['is_entry'], 1, -1
        )
        temp_df['total_num_currency'] = temp_df['add_currency'].cumsum()

        # print("Here Here temp_df length = " + str(temp_df.shape[0]))
        #
        # print("temp_df:")
        # print(temp_df)

        min_margin_level = temp_df['margin_level'].min()
        min_margin_level_id = temp_df['margin_level'].argmin()

        max_margin_level = temp_df[temp_df['cum_margin'] > 0]['margin_level'].max()
        max_margin_level_id = temp_df[temp_df['cum_margin'] > 0]['margin_level'].argmax()

        max_margin_level_id = temp_df[temp_df['cum_margin'] > 0].iloc[max_margin_level_id]['index']

        #     print("min_margin_level = " + str(min_margin_level))
        #     print('min_margin_level_id = ' + str(min_margin_level_id))

        #     print("max_margin_level = " + str(max_margin_level))
        #     print('max_margin_level_id = ' + str(max_margin_level_id))

        #     print("Temp df ***************************************************:")
        # display(temp_df.iloc[:20])
        # display(temp_df)

        #temp_df.to_csv(temp_output_file, index=False)

        ########################################

        if 'tp_num' in trade_df.columns:
            trade_num = trade_df[trade_df['tp_num'].isnull()].shape[0]
            # print("trade number = " + str(trade_num))

        # print("all trade number = " + str(trade_df.shape[0]))

        # trade_df.to_csv(output_file, index = False)
        # display(trade_df.sort_values(by = ['entry_time', 'currency']))

        max_draw_down, start_draw_down, end_draw_down = calc_max_drawdown(trade_df['cum_pnl'])
        last_cum_pnl = trade_df.iloc[-1]['cum_pnl']
        #     print("last_cum_pnl = " + str(last_cum_pnl))
        #     print("max drawdown = " + str(max_draw_down))
        #     print("max drawdown start id = " + str(start_draw_down))
        #     print('max drawdown end id = ' + str(end_draw_down))

        adj_ret = last_cum_pnl / max_draw_down

        # print("Final performance adj_ret = " + str(adj_ret))

        return_rate = last_cum_pnl / init_deposit
        drawdown_rate = max_draw_down / init_deposit

        #     print("return_rate = " + ("%.2f" % (return_rate * 100)) + "%")
        #     print("drawdown_rate = " + ("%.2f" % (drawdown_rate * 100)) + "%")

        ##################
        dummy_trade_df = trade_df.iloc[0:1].copy()

        for col in ['is_win', 'pnl', 'cum_pnl']:
            dummy_trade_df.at[0, col] = 0

        trade_df = pd.concat([dummy_trade_df, trade_df])
        trade_df['id'] = list(range(trade_df.shape[0]))

        # print("Plot pnl figure")
        fig = plt.figure(figsize=(20, 30))  # 10,5

        axes = fig.subplots(nrows=3, ncols=1)

        font_size = 25

        sns.lineplot(x='id', y='cum_pnl', markers='o', color='blue', data=trade_df, ax=axes[0])
        axes[0].set_title("All Cum Pnl Curve", fontsize=font_size)
        axes[0].set_xlabel(axes[0].get_xlabel(), size=font_size)
        #axes[0].set_xticklabels(axes[0].get_xticks(), size=font_size)
        axes[0].set_ylabel(axes[0].get_ylabel(), size=font_size)
        #axes[0].set_yticklabels(axes[0].get_yticks(), size=font_size)
        # axes.yaxis.set_major_locator(ticker.MultipleLocator(4))
        axes[0].xaxis.set_major_locator(ticker.MultipleLocator(20))
        axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1000))
        axes[0].axhline(0, ls='--', color='green', linewidth=1)

        axes[0].axvline(start_draw_down + 1, ls='--', color='red', linewidth=1)
        axes[0].axvline(end_draw_down + 1, ls='--', color='red', linewidth=1)
        plt.setp(axes[0].get_xticklabels(), rotation=45)

        # print("cutoff_trade_num = " + str(cutoff_trade_num))
        # if cutoff_end_date is not None:
        #     axes[0].axvline(cutoff_trade_num, ls='--', color='green', linewidth=1)

        #     sns.lineplot(x = 'index', y = 'cum_pnl', markers = 'o', color = 'blue', data = temp_df, ax = axes[1])
        #     axes[1].set_title("Cum Pnl 2")
        #     #axes.yaxis.set_major_locator(ticker.MultipleLocator(4))
        #     axes[1].xaxis.set_major_locator(ticker.MultipleLocator(40))
        #     axes[1].axhline(0, ls='--', color='green', linewidth=1)

        #     plt.setp(axes[1].get_xticklabels(), rotation = 45)

        sns.lineplot(x='index', y='cum_margin', markers='o', color='blue', data=temp_df, ax=axes[1])
        axes[1].set_title("Cum Margin", fontsize=font_size)
        axes[1].set_xlabel(axes[1].get_xlabel(), size=font_size)
        #axes[1].set_xticklabels(axes[1].get_xticks(), size=font_size)
        axes[1].set_ylabel(axes[1].get_ylabel(), size=font_size)
        #axes[1].set_yticklabels(axes[1].get_yticks(), size=font_size)
        # axes.yaxis.set_major_locator(ticker.MultipleLocator(4))
        axes[1].xaxis.set_major_locator(ticker.MultipleLocator(40))
        axes[1].axhline(0, ls='--', color='green', linewidth=1)

        plt.setp(axes[1].get_xticklabels(), rotation=45)

        sns.lineplot(x='index', y='margin_level', markers='o', color='blue', data=temp_df, ax=axes[2])
        axes[2].set_title("Margin Level", fontsize=font_size)
        axes[2].set_xlabel(axes[2].get_xlabel(), size=font_size)
        #axes[2].set_xticklabels(axes[2].get_xticks(), size=font_size)
        axes[2].set_ylabel(axes[2].get_ylabel(), size=font_size)
        #axes[2].set_yticklabels(axes[2].get_yticks(), size=font_size)
        # axes.yaxis.set_major_locator(ticker.MultipleLocator(4))
        axes[2].xaxis.set_major_locator(ticker.MultipleLocator(40))
        axes[2].axhline(0, ls='--', color='green', linewidth=1)

        plt.setp(axes[2].get_xticklabels(), rotation=45)

        plt.subplots_adjust(hspace=0.5)

        # print("cutoff_trade_num = " + str(cutoff_trade_num))
        # if cutoff_end_date is not None:
        #     axes[0].axvline(cutoff_trade_num, ls='--', color='green', linewidth=1)

        my_start_date = trade_df.iloc[0]['entry_time'].date()
        my_end_date = trade_df.iloc[-1]['entry_time'].date()

        delta = my_end_date - my_start_date

        # print("start_date = " + str(start_date))
        # print("delta = " + str(delta))
        days = delta.days
        # print("days = " + str(days))

        trading_days = np.busday_count(my_start_date, my_end_date)
        # print('trading days = ' + str(trading_days))

        trade_num = trade_df_small.shape[0]

        # print("trade number = " + str(trade_num))

        trades_per_day = trade_num / trading_days

        # print("Number of trades per day = " + str(trades_per_day))

        currency_name = "Until " + currency if accumulated_mode else currency

        summary_data += [[currency_name, round(last_cum_pnl, 2), round(max_draw_down, 2), start_draw_down, end_draw_down,
                          round(adj_ret, 2), ("%.2f" % (return_rate * 100)) + "%", ("%.2f" % (drawdown_rate * 100)) + "%",
                          trading_days, trade_num, round(trades_per_day, 2)]]

        if sorted:
            fig.savefig(os.path.join(final_output_folder, currency_name + ".png"))

        plt.close(fig)

    # print(final_output_folder)
    summary_df = pd.DataFrame(data=summary_data, columns=summary_columns)

    # display(summary_df)
    # print("accumulated_mode = " + str(accumulated_mode))

    temp_summary_df = summary_df.iloc[0:-1] if not accumulated_mode else summary_df

    # print("temp_summary_df here")
    # display(temp_summary_df)

    if not accumulated_mode:
        temp_summary_df = temp_summary_df.sort_values(by=['adj_return'], ascending=False)

    # print("temp_summary_df")
    # display(temp_summary_df)

    summary_df = pd.concat([temp_summary_df, summary_df.iloc[-1:]]) if not accumulated_mode else temp_summary_df

    #print(summary_df)

    if sorted:
        file_name = "summary_result.csv" if not accumulated_mode else "accumulated_summary_result.csv"
        summary_df.to_csv(os.path.join(final_output_folder, file_name), index=False)


    return summary_df


#start_dates = [datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1)]
#end_dates = [datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1), datetime(2024, 2, 4)] #5 months rolling forward

#1 month rolling forward
#start_dates = [datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1)]
#end_dates = [datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1), datetime(2024, 2, 4)] #5 months rolling forward

#2 months rolling forward
# start_dates = [datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1)]
# end_dates = [datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1), datetime(2024, 2, 4)] #5 months rolling forward

#3 months rolling forward
# start_dates = [datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1)]
# end_dates = [datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1), datetime(2024, 2, 4)] #5 months rolling forward

#4 months rolling forward
#start_dates = [datetime(2023, 4, 1), datetime(2023, 5, 1), datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1)]
#end_dates = [datetime(2023, 8, 1), datetime(2023, 9, 1), datetime(2023, 10, 1), datetime(2023, 11, 1), datetime(2023, 12, 1), datetime(2024, 1, 1), datetime(2024, 2, 4)] #5 months rolling forward



start_dates = [datetime(2023,4,1)]
end_dates = [datetime(2024,3,30)]

columns = ['by_date', 'optimal_currency_list']
final_data = []

optimal_portfolio = None

for start_date, end_date in list(zip(start_dates, end_dates)) :

    optimal_currency_list = construct_portfolio_for_end_date(end_date=end_date, start_date = start_date)

    print("")
    print("Optimal currency list by date " + start_date.strftime("%Y%m%d") + "- " + end_date.strftime("%Y%m%d"))
    print(optimal_currency_list)

    if optimal_portfolio is None:
        optimal_portfolio = set(optimal_currency_list)
    else:
        optimal_portfolio = optimal_portfolio.intersection(set(optimal_currency_list))

    final_data += [[start_date.strftime("%Y%m%d") + "-" +end_date.strftime("%Y%m%d"), ','.join(["\'" + currency + "\'" for currency in optimal_currency_list])]]


final_data += [["All", ','.join(["\'" + currency + "\'" for currency in list(optimal_portfolio)])]]

portfolio_df = pd.DataFrame(data = final_data, columns = columns)

#portfolio_df.to_csv(os.path.join(root_dir, "optimal_portfolio_by_date.csv"), index = False)

portfolio_df.to_csv(os.path.join(root_dir, "optimal_portfolio_by_start_end_date.csv"), index = False)

#portfolio_df.to_csv(os.path.join(root_dir, "optimal_portfolio_by_start_end_date_4month.csv"), index = False)

print("")
print("Optimal Portfolio By date:")

print(portfolio_df)

print("")
print("Optimal portfolio is:")
print(optimal_portfolio)

print("Number of selected currencies = " + str(len(optimal_portfolio)))



