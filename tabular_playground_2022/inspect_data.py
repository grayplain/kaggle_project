import pandas as pd
import numpy as np


class inspect_data:

    def __init__(self, val):
        self.huga = val

    def hello(self, abc):
        print(abc)

    def assign_test(self, val):
        self.huga = val

    def use_test(self):
        print(self.huga)

print("this.")

ins = inspect_data()
# ins = inspect_data("kon!")
ins.hello("test method.")
# ins.assign_test(10)
ins.use_test()