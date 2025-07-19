class StrategyExecution:

    def __init__(self, side, leverage, take_profit_pct, take_loss_pct, strategy_id, execution_id, strategy_entry_time,
                 strategy_entry_price,
                 execution_entry_time, execution_entry_price, strategy_entry_value, execution_entry_value, default_leverage = 10, prod_size=0):
        self.active = True
        self.side = side  # 1 means long  -1 means short
        self.leverage = leverage
        self.take_profit_pct = take_profit_pct
        self.take_loss_pct = take_loss_pct
        self.strategy_id = strategy_id
        self.execution_id = execution_id
        self.strategy_entry_time = strategy_entry_time
        self.strategy_entry_price = strategy_entry_price
        self.execution_entry_time = execution_entry_time
        self.execution_entry_price = execution_entry_price
        self.strategy_entry_value = strategy_entry_value  # This is actual notioanl value (margin value, not leveraged)
        self.execution_entry_value = execution_entry_value

        self.default_leverage = default_leverage

        self.execution_exit_time = None
        self.execution_exit_price = -1
        self.execution_exit_value = -1

        self.prod_size = prod_size  # This is leveraged size (enlarged size)

        self.pnl_rate = 0
        self.pnl = 0

        self.prod_strategy_entry_price = 0
        self.prod_execution_entry_price = 0
        self.prod_strategy_entry_value = 0
        self.prod_execution_entry_value = 0

        self.prod_execution_exit_price = -1
        self.prod_execution_exit_value = -1

        self.prod_pnl_rate = 0
        self.prod_pnl = 0

        self.initialize()

    def __str__(self):

        array = ["side="+str(self.side), "leverage="+str(self.leverage),
                 "take_profit_pct="+str(self.take_profit_pct), "take_loss_pct="+str(self.take_loss_pct),
                 "strategy_id="+str(self.strategy_id), "execution_id="+str(self.execution_id),
                 "strategy_entry_time="+str(self.strategy_entry_time), "strategy_entry_price="+str(self.strategy_entry_price),
                 "execution_entry_time="+str(self.execution_entry_time), "execution_entry_price="+str(self.execution_entry_price),
                 "strategy_entry_value="+str(self.strategy_entry_value), "execution_entry_value="+str(self.execution_entry_value),
                 "prod_size="+str(self.prod_size)
                 ]

        return "Execution [" + ','.join(array) + ']'


    def initialize(self):
        self.pnl_rate = 0
        self.pnl = 0

        self.take_profit_price = self.execution_entry_price * (1 + self.side * self.take_profit_pct)
        self.take_loss_price = self.execution_entry_price * (1 - self.side * self.take_loss_pct)

    def set_prod_strategy_entry_price(self, prod_entry_price):
        self.prod_strategy_entry_price = prod_entry_price
        self.prod_execution_entry_price = prod_entry_price

        self.prod_strategy_entry_value = prod_entry_price * self.prod_size / self.default_leverage
        self.prod_execution_entry_value = self.prod_strategy_entry_value

    def exit_execution(self, execution_exit_time, execution_exit_price, is_signal_exit, is_extra_execution):
        return_rate = self.side * (execution_exit_price - self.execution_entry_price) / self.execution_entry_price

        self.pnl_rate = return_rate * self.leverage
        self.pnl = self.execution_entry_value * self.pnl_rate

        self.execution_exit_price = execution_exit_price
        self.execution_exit_value = self.execution_entry_value + self.pnl
        self.execution_exit_time = execution_exit_time

        self.active = (not is_signal_exit) and self.pnl > 0 and (not is_extra_execution)

    def exit_execution_prod(self, prod_execution_exit_price):
        prod_return_rate = self.side * (
                    prod_execution_exit_price - self.prod_execution_entry_price) / self.prod_execution_entry_price

        self.prod_pnl_rate = prod_return_rate * self.leverage
        self.prod_pnl = self.prod_execution_entry_value * self.pnl_rate

        self.prod_execution_exit_price = prod_execution_exit_price
        self.prod_execution_exit_value = self.prod_execution_entry_value + self.prod_pnl

    def calc_increased_size_when_take_profit(self):
        #return self.prod_size * self.take_profit_pct * self.leverage
        return self.prod_size * (self.prod_execution_entry_price/self.take_profit_price * (1 + self.take_profit_pct * self.leverage) - 1)

    def update_to_next_execution(self, entry_time, increased_size):
        self.execution_id = self.execution_id + 1
        self.execution_entry_time = entry_time
        self.execution_entry_price = self.execution_exit_price
        self.execution_entry_value = self.execution_exit_value
        self.prod_size = self.prod_size + increased_size

        self.prod_execution_entry_price = self.prod_execution_exit_price
        self.prod_execution_entry_value = self.prod_execution_exit_value

        self.initialize()