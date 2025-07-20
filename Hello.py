import os
from enum import Enum,auto
print("Hello world")

class OrderType(Enum):
    TAKE_PROFIT_EXIT = auto()
    STOP_LOSS_EXIT = auto()
    STOP_ENTER = auto()


orderType: OrderType = OrderType.TAKE_PROFIT_EXIT

print("orderType: " + str(orderType))