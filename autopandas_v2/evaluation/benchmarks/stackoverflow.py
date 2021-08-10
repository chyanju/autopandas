from io import StringIO

import pandas as pd
import numpy as np
from autopandas_v2.evaluation.benchmarks.base import Benchmark


class PandasBenchmarks:

    # https://stackoverflow.com/questions/68243146/replace-zero-with-value-of-an-other-column-using-pandas
    class LCF_2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'ref': {0: 8400, 1: 3840, 2: 7400, 3: 5200}, 'Name': {0: 'John', 1: 'Peter', 2: 'David', 3: 'Karen'}, 'id': {0: 0, 1: 414, 2: 612, 3: 0}, 'Score': {0: 12, 1: 0, 2: 64, 3: 0}})
            ]
            self.output = pd.DataFrame({'ref': {0: 8400, 1: 3840, 2: 7400, 3: 5200}, 'Name': {0: 'John', 1: 'Peter', 2: 'David', 3: 'Karen'}, 'id': {0: 8400, 1: 414, 2: 612, 3: 5200}, 'Score': {0: 12, 1: 0, 2: 64, 3: 0}})
            self.funcs = ['df.loc_getitem', 'df.mask', 'df.eq']
            self.seqs = [[0, 0, 2, 0, 1]]

    # https://stackoverflow.com/questions/68231104/extract-part-of-a-3-d-dataframe
    class LCF_4(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    np.arange(1, 25).reshape((-1, 8)),
                    columns=pd.MultiIndex.from_product((['d1', 'd2'], list('ABCD')))
                )
            ]
            self.output = pd.DataFrame(
                np.asarray([[1,2,5,6],[9,10,13,14],[17,18,21,22]]),
                columns=pd.MultiIndex.from_product((['d1', 'd2'], list('AB')))
            )
            self.funcs = ['df.loc_getitem', 'df.columns', 'df.isin']
            self.seqs = [[1, 2, 0]]

    # https://stackoverflow.com/questions/68193521/concatenate-values-and-column-names-in-a-data-frame-to-create-a-new-data-frame
    class LCF_9(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'Value': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}, 'col1': {0: 'aa', 1: 'ba', 2: 'ca', 3: 'da', 4: 'ea'}, 'col2': {0: 'ab', 1: 'bb', 2: 'cb', 3: 'db', 4: 'eb'}, 'col3': {0: 'ac', 1: 'bc', 2: 'cc', 3: 'dc', 4: 'ec'}})
            ]
            self.output = pd.DataFrame({'Value': {0: 'a_col1', 1: 'b_col1', 2: 'c_col1', 3: 'd_col1', 4: 'e_col1', 5: 'a_col2', 6: 'b_col2', 7: 'c_col2', 8: 'd_col2', 9: 'e_col2', 10: 'a_col3', 11: 'b_col3', 12: 'c_col3', 13: 'd_col3', 14: 'e_col3'}, 'Col 1': {0: 'aa', 1: 'ba', 2: 'ca', 3: 'da', 4: 'ea', 5: 'ab', 6: 'bb', 7: 'cb', 8: 'db', 9: 'eb', 10: 'ac', 11: 'bc', 12: 'cc', 13: 'dc', 14: 'ec'}})
            self.funcs = ['df.melt', 'df.add', 'df.drop']
            self.seqs = [[0, 1, 2]]

    # https://stackoverflow.com/questions/67917573/replace-nan-with-sign-only-in-specefic-condition-python-pandas
    class LCF_23(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'L1': {0: 1.0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}, 'D1': {0: 'ABC', 1: np.nan, 2: '4.1', 3: '1.8', 4: np.nan}, 'L2': {0: 1.1, 1: 1.7, 2: np.nan, 3: 3.2, 4: 1.6}, 'D2': {0: '4.1', 1: np.nan, 2: np.nan, 3: 'PQR', 4: np.nan}, 'L3': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}})
            ]
            self.output = pd.DataFrame({'L1': {0: 1.0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}, 'D1': {0: 'ABC', 1: np.nan, 2: '4.1', 3: '1.8', 4: np.nan}, 'L2': {0: '1.1', 1: '1.7', 2: '-', 3: '3.2', 4: '1.6'}, 'D2': {0: '4.1', 1: '-', 2: '-', 3: 'PQR', 4: '-'}, 'L3': {0: '-', 1: '-', 2: '-', 3: '-', 4: '-'}})
            self.funcs = ['df.loc_getitem', 'df.notna', 'df.cumsum', 'df.eq', 'df.mask']
            self.seqs = [[0, 1, 2, 3, 4]]

    # https://stackoverflow.com/questions/67870585/python-dataframe-create-index-column-based-on-other-id-column
    class LCF_28(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'ID': {0: '000afb96ded6677c', 1: '000afb96ded6677c', 2: '000afb96ded6677c', 3: '000afb96ded6677c', 4: '000afb96ded6677c', 5: 'ffea14e87a4e1269', 6: 'ffea14e87a4e1269', 7: 'ffea14e87a4e1269', 8: 'fff455057ad492da', 9: 'fff5fc66c1fd66c2'}, 'Price': {0: 1514.5, 1: 13.0, 2: 611.0, 3: 723.0, 4: 2065.0, 5: 2286.0, 6: 1150.0, 7: 80.0, 8: 650.0, 9: 450.0}})
            ]
            self.output = pd.DataFrame({'ID': {0: '000afb96ded6677c', 1: '000afb96ded6677c', 2: '000afb96ded6677c', 3: '000afb96ded6677c', 4: '000afb96ded6677c', 5: 'ffea14e87a4e1269', 6: 'ffea14e87a4e1269', 7: 'ffea14e87a4e1269', 8: 'fff455057ad492da', 9: 'fff5fc66c1fd66c2'}, 'Price': {0: 1514.5, 1: 13.0, 2: 611.0, 3: 723.0, 4: 2065.0, 5: 2286.0, 6: 1150.0, 7: 80.0, 8: 650.0, 9: 450.0}, 'ID_2': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4}})
            self.funcs = ['df.rank', 'df.loc_getitem', 'df.astype']
            self.seqs = [[1, 0, 2]]

    # https://stackoverflow.com/questions/67845362/sort-pandas-df-subset-of-rows-within-a-group-by-specific-column
    class LCF_30(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'A': {0: 'z', 1: 'z', 2: 'x', 3: 'x', 4: 'u', 5: 'u', 6: 'y', 7: 'y'}, 'B': {0: 'k', 1: 'k', 2: 't', 3: 't', 4: 'c', 5: 'c', 6: 't', 7: 't'}, 'C': {0: 's', 1: 's', 2: 'r', 3: 'r', 4: 'r', 5: 'r', 6: 's', 7: 's'}, 'D': {0: 7, 1: 6, 2: 2, 3: 1, 4: 8, 5: 9, 6: 5, 7: 2}, 'E': {0: 'd', 1: 'l', 2: 'e', 3: 'x', 4: 'f', 5: 'h', 6: 'l', 7: 'o'}})
            ]
            self.output = pd.DataFrame({'A': {0: 'z', 1: 'z', 2: 'x', 3: 'x', 4: 'u', 5: 'u', 6: 'y', 7: 'y'}, 'B': {0: 'k', 1: 'k', 2: 't', 3: 't', 4: 'c', 5: 'c', 6: 't', 7: 't'}, 'C': {0: 's', 1: 's', 2: 'r', 3: 'r', 4: 'r', 5: 'r', 6: 's', 7: 's'}, 'D': {0: 6, 1: 7, 2: 1, 3: 2, 4: 8, 5: 9, 6: 2, 7: 5}, 'E': {0: 'l', 1: 'd', 2: 'x', 3: 'e', 4: 'f', 5: 'h', 6: 'o', 7: 'l'}})
            self.funcs = ['df.sort_values']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/26837998/pandas-replace-nan-with-blank-empty-string
    class LCF_45(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'1': {0: 'a', 1: 'b', 2: 'c'}, '2': {0: np.nan, 1: 'l', 2: np.nan}, '3': {0: 'read', 1: 'unread', 2: 'read'}})
            ]
            self.output = pd.DataFrame({'1': {0: 'a', 1: 'b', 2: 'c'}, '2': {0: "", 1: 'l', 2: ""}, '3': {0: 'read', 1: 'unread', 2: 'read'}})
            self.funcs = ['df.fillna']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/67257898/how-to-add-a-value-to-a-new-column-by-referencing-the-values-in-a-column
    class LCF_47(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}, 'reason': {0: 'x1', 1: 'x1', 2: 'x4', 3: 'x3', 4: 'x1', 5: 'x1', 6: 'x4', 7: 'x4'}, 'x1': {0: 100, 1: 15, 2: 10, 3: 12, 4: 80, 5: 15, 6: 90, 7: 12}, 'x2': {0: 15, 1: 16, 2: 50, 3: 15, 4: 15, 5: 19, 6: 40, 7: 85}, 'x3': {0: 10, 1: 14, 2: 40, 3: 60, 4: 10, 5: 84, 6: 90, 7: 60}, 'x4': {0: 20, 1: 10, 2: 30, 3: 5, 4: 20, 5: 10, 6: 30, 7: 50}, 'x5': {0: 25, 1: 10, 2: 25, 3: 1, 4: 25, 5: 10, 6: 25, 7: 10}})
            ]
            self.output = pd.DataFrame({'id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}, 'reason': {0: 'x1', 1: 'x1', 2: 'x4', 3: 'x3', 4: 'x1', 5: 'x1', 6: 'x4', 7: 'x4'}, 'x1': {0: 100, 1: 15, 2: 10, 3: 12, 4: 80, 5: 15, 6: 90, 7: 12}, 'x2': {0: 15, 1: 16, 2: 50, 3: 15, 4: 15, 5: 19, 6: 40, 7: 85}, 'x3': {0: 10, 1: 14, 2: 40, 3: 60, 4: 10, 5: 84, 6: 90, 7: 60}, 'x4': {0: 20, 1: 10, 2: 30, 3: 5, 4: 20, 5: 10, 6: 30, 7: 50}, 'x5': {0: 25, 1: 10, 2: 25, 3: 1, 4: 25, 5: 10, 6: 25, 7: 10}, 'xy': {0: 100, 1: 15, 2: 30, 3: 60, 4: 80, 5: 15, 6: 30, 7: 50}})
            self.funcs = ['df.apply', 'df.loc_getitem']
            self.seqs = [[1, 1, 0]]

    # https://stackoverflow.com/questions/67246859/how-to-convert-rows-into-columns-and-filter-using-the-id
    class LCF_50(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'customer_id': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}, 'key_id': {0: 777, 1: 888, 2: 999, 3: 777, 4: 888}, 'quantity': {0: 3, 1: 2, 2: 3, 3: 6, 4: 1}})
            ]
            self.output = pd.DataFrame({'777': {0: 3, 1: 6}, '888': {0: 2, 1: 1}, '999': {0: 3, 1: 0}})
            self.funcs = ['df.pivot_table', 'df.fillna']
            self.seqs = [[0, 1]]

    # https://stackoverflow.com/questions/67053308/how-to-replace-zero-by-one-for-particular-row-in-data-frame
    class LCF_56(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'DP1': {'OP1': 43239.0, 'OP2': 146.0, 'OP3': 266279.0, 'OP4': 360547.0, 'OP5': 380497.0, 'OP6': 6151.0, 'OP7': 142026.0, 'OP8': 76860.0, 'OP9': 6210.0, 'OP10': np.nan, 'Total': 1281955.0, 'Variance': 160244.0, "Mack's SIgma": 400.0}, 'DP2': {'OP1': 46962.0, 'OP2': 73.0, 'OP3': 1189.0, 'OP4': 56943.0, 'OP5': 17946.0, 'OP6': 16525.0, 'OP7': 21999.0, 'OP8': 102580.0, 'OP9': np.nan, 'OP10': np.nan, 'Total': 264217.0, 'Variance': 37745.0, "Mack's SIgma": 194.0}, 'DP3': {'OP1': 55858.0, 'OP2': 16647.0, 'OP3': 1.0, 'OP4': 142271.0, 'OP5': 19376.0, 'OP6': 17046.0, 'OP7': 820.0, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 252019.0, 'Variance': 42003.0, "Mack's SIgma": 205.0}, 'DP4': {'OP1': 9128.0, 'OP2': 5596.0, 'OP3': 10939.0, 'OP4': 38217.0, 'OP5': 0.0, 'OP6': 11532.0, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 75412.0, 'Variance': 15082.0, "Mack's SIgma": 123.0}, 'DP5': {'OP1': 30372.0, 'OP2': 1493.0, 'OP3': 17799.0, 'OP4': 1141.0, 'OP5': 3974.0, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 54779.0, 'Variance': 13695.0, "Mack's SIgma": 117.0}, 'DP6': {'OP1': 5932.0, 'OP2': 7175.0, 'OP3': 4702.0, 'OP4': 6757.0, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 24566.0, 'Variance': 89.0, "Mack's SIgma": 90.0}, 'DP7': {'OP1': 667.0, 'OP2': 45.0, 'OP3': 235.0, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 947.0, 'Variance': 474.0, "Mack's SIgma": 22.0}, 'DP8': {'OP1': 663.0, 'OP2': 438.0, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 1101.0, 'Variance': 1101.0, "Mack's SIgma": 33.0}, 'DP9': {'OP1': 0.0, 'OP2': np.nan, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 0.0, 'Variance': np.nan, "Mack's SIgma": np.nan}, 'DP10': {'OP1': np.nan, 'OP2': np.nan, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 0.0, 'Variance': 0.0, "Mack's SIgma": 0.0}})
            ]
            self.output = pd.DataFrame({'DP1': {'OP1': 43239.0, 'OP2': 146.0, 'OP3': 266279.0, 'OP4': 360547.0, 'OP5': 380497.0, 'OP6': 6151.0, 'OP7': 142026.0, 'OP8': 76860.0, 'OP9': 6210.0, 'OP10': np.nan, 'Total': 1281955.0, 'Variance': 160244.0, "Mack's Sigma": 400.0}, 'DP2': {'OP1': 46962.0, 'OP2': 73.0, 'OP3': 1189.0, 'OP4': 56943.0, 'OP5': 17946.0, 'OP6': 16525.0, 'OP7': 21999.0, 'OP8': 102580.0, 'OP9': np.nan, 'OP10': np.nan, 'Total': 264217.0, 'Variance': 37745.0, "Mack's Sigma": 194.0}, 'DP3': {'OP1': 55858.0, 'OP2': 16647.0, 'OP3': 1.0, 'OP4': 142271.0, 'OP5': 19376.0, 'OP6': 17046.0, 'OP7': 820.0, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 252019.0, 'Variance': 42003.0, "Mack's Sigma": 205.0}, 'DP4': {'OP1': 9128.0, 'OP2': 5596.0, 'OP3': 10939.0, 'OP4': 38217.0, 'OP5': 0.0, 'OP6': 11532.0, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 75412.0, 'Variance': 15082.0, "Mack's Sigma": 123.0}, 'DP5': {'OP1': 30372.0, 'OP2': 1493.0, 'OP3': 17799.0, 'OP4': 1141.0, 'OP5': 3974.0, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 54779.0, 'Variance': 13695.0, "Mack's Sigma": 117.0}, 'DP6': {'OP1': 5932.0, 'OP2': 7175.0, 'OP3': 4702.0, 'OP4': 6757.0, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 24566.0, 'Variance': 89.0, "Mack's Sigma": 90.0}, 'DP7': {'OP1': 667.0, 'OP2': 45.0, 'OP3': 235.0, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 947.0, 'Variance': 474.0, "Mack's Sigma": 22.0}, 'DP8': {'OP1': 663.0, 'OP2': 438.0, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 1101.0, 'Variance': 1101.0, "Mack's Sigma": 33.0}, 'DP9': {'OP1': 0.0, 'OP2': np.nan, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 0.0, 'Variance': 474.0, "Mack's Sigma": np.nan}, 'DP10': {'OP1': np.nan, 'OP2': np.nan, 'OP3': np.nan, 'OP4': np.nan, 'OP5': np.nan, 'OP6': np.nan, 'OP7': np.nan, 'OP8': np.nan, 'OP9': np.nan, 'OP10': np.nan, 'Total': 0.0, 'Variance': 0.0, "Mack's Sigma": 0.0}})
            self.funcs = ['df.loc_getitem', 'df.min']
            self.seqs = [[0, 1]]

#     # https://stackoverflow.com/questions/11881165
#     class SO_11881165_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame({"a": [5, 6, 7, 8, 9], "b": [10, 11, 12, 13, 14]})]
#             self.output = self.inputs[0].loc[[0, 2, 4]]
#             self.funcs = ['df.loc_getitem']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/11941492/
#     # same thing
#     class SO_11941492_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             df = pd.DataFrame({'group1': ['a', 'a', 'a', 'b', 'b', 'b'],
#                                'group2': ['c', 'c', 'd', 'd', 'd', 'e'],
#                                'value1': [1.1, 2, 3, 4, 5, 6],
#                                'value2': [7.1, 8, 9, 10, 11, 12]
#                                })
#             df = df.set_index(['group1', 'group2'])
#             self.inputs = [df]
#             self.output = df.xs('a', level=0)
#             self.funcs = ['df.xs']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/13647222
#     class SO_13647222_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'series': {0: 'A', 1: 'B', 2: 'C', 3: 'A', 4: 'B', 5: 'C', 6: 'A', 7: 'B',
#                                          8: 'C', 9: 'A', 10: 'B', 11: 'C', 12: 'A', 13: 'B', 14: 'C'},
#                               'step': {0: '100', 1: '100', 2: '100', 3: '101', 4: '101', 5: '101', 6: '102', 7: '102',
#                                        8: '102', 9: '103', 10: '103', 11: '103', 12: '104', 13: '104', 14: '104'},
#                               'value': {0: '1000', 1: '1001', 2: '1002', 3: '1003', 4: '1004', 5: '1005', 6: '1006',
#                                         7: '1007',
#                                         8: '1008', 9: '1009', 10: '1010', 11: '1011', 12: '1012', 13: '1013',
#                                         14: '1014'}})
#             ]
#             self.output = self.inputs[0].pivot(columns='series', values='value', index='step')
#             self.funcs = ['df.pivot']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/18172851/
#     class SO_18172851_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             df = pd.DataFrame({'daysago': {'2007-03-31': 62, '2007-03-10': 83, '2007-02-10': 111, '2007-01-13': 139,
#                                            '2006-12-23': 160, '2006-11-09': 204, '2006-10-22': 222, '2006-09-29': 245,
#                                            '2006-09-16': 258, '2006-08-30': 275, '2006-02-11': 475, '2006-01-13': 504,
#                                            '2006-01-02': 515, '2005-12-06': 542, '2005-11-29': 549, '2005-11-22': 556,
#                                            '2005-11-01': 577, '2005-10-20': 589, '2005-09-27': 612, '2005-09-07': 632,
#                                            '2005-06-12': 719, '2005-05-29': 733, '2005-05-02': 760, '2005-04-02': 790,
#                                            '2005-03-13': 810, '2004-11-09': 934},
#                                'line_race': {'2007-03-31': 111, '2007-03-10': 211, '2007-02-10': 29, '2007-01-13': 110,
#                                              '2006-12-23': 210, '2006-11-09': 39, '2006-10-22': 28, '2006-09-29': 49,
#                                              '2006-09-16': 311, '2006-08-30': 48, '2006-02-11': 45, '2006-01-13': 0,
#                                              '2006-01-02': 0, '2005-12-06': 0, '2005-11-29': 0, '2005-11-22': 0,
#                                              '2005-11-01': 0, '2005-10-20': 0, '2005-09-27': 0, '2005-09-07': 0,
#                                              '2005-06-12': 0, '2005-05-29': 0, '2005-05-02': 0, '2005-04-02': 0,
#                                              '2005-03-13': 0, '2004-11-09': 0},
#                                'rw': {'2007-03-31': 0.99999, '2007-03-10': 0.97, '2007-02-10': 0.9,
#                                       '2007-01-13': 0.8806780000000001, '2006-12-23': 0.793033, '2006-11-09': 0.636655,
#                                       '2006-10-22': 0.581946, '2006-09-29': 0.518825, '2006-09-16': 0.48622600000000005,
#                                       '2006-08-30': 0.446667, '2006-02-11': 0.16459100000000002,
#                                       '2006-01-13': 0.14240899999999998, '2006-01-02': 0.1348,
#                                       '2005-12-06': 0.11780299999999999, '2005-11-29': 0.113758,
#                                       '2005-11-22': 0.10985199999999999, '2005-11-01': 0.098919, '2005-10-20': 0.093168,
#                                       '2005-09-27': 0.083063, '2005-09-07': 0.075171, '2005-06-12': 0.04869,
#                                       '2005-05-29': 0.045404, '2005-05-02': 0.039679, '2005-04-02': 0.03416,
#                                       '2005-03-13': 0.030914999999999998, '2004-11-09': 0.016647}})
#             df['rating'] = range(2, 28)
#             df['wrating'] = df['rw'] * df['rating']
#             df = df[['daysago', 'line_race', 'rating', 'rw', 'wrating']]
#             self.inputs = [df, lambda a: a.line_race != 0]
#             self.output = self.inputs[0].loc[lambda a: a.line_race != 0]
#             self.funcs = ['df.loc_getitem']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/49583055
#     class SO_49583055_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame({'ID': {0: 20, 1: 21, 2: 22, 3: 32, 4: 31, 5: 33},
#                                          'admit': {0: pd.Timestamp('2018-03-04 00:00:00'),
#                                                    1: pd.Timestamp('2018-02-02 00:00:00'),
#                                                    2: pd.Timestamp('2018-02-05 00:00:00'),
#                                                    3: pd.Timestamp('2018-01-02 00:00:00'),
#                                                    4: pd.Timestamp('2018-01-15 00:00:00'),
#                                                    5: pd.Timestamp('2018-01-20 00:00:00')},
#                                          'discharge': {0: pd.Timestamp('2018-03-06 00:00:00'),
#                                                        1: pd.Timestamp('2018-02-06 00:00:00'),
#                                                        2: pd.Timestamp('2018-02-23 00:00:00'),
#                                                        3: pd.Timestamp('2018-02-03 00:00:00'),
#                                                        4: pd.Timestamp('2018-01-18 00:00:00'),
#                                                        5: pd.Timestamp('2018-01-24 00:00:00')},
#                                          'discharge_location': {0: 'Home1', 1: 'Home2', 2: 'Home3', 3: 'Home4',
#                                                                 4: 'Home5',
#                                                                 5: 'Home6'},
#                                          'first': {0: 11, 1: 10, 2: 9, 3: 8, 4: 12, 5: 7}})]
#             self.output = self.inputs[0].sort_values(by=['ID', 'first', 'admit'], ascending=[True, False, True])
#             self.funcs = ['df.sort_values']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/49592930
#     # ok I didn't uniqify the timestamps because that would change the actual output
#     class SO_49592930_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:30:00'): 0.0,
#                                                    pd.Timestamp('2014-05-21 10:00:00'): 10.0,
#                                                    pd.Timestamp('2014-05-21 10:30:00'): 3.0,
#                                                    pd.Timestamp('2017-07-10 22:30:00'): 18.3,
#                                                    pd.Timestamp('2017-07-10 23:00:00'): 7.6,
#                                                    pd.Timestamp('2017-07-10 23:30:00'): 2.0}}),
#                            pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:00:00'): 1.0,
#                                                    pd.Timestamp('2014-05-21 10:00:00'): 13.0,
#                                                    pd.Timestamp('2017-07-10 21:00:00'): 1.6,
#                                                    pd.Timestamp('2017-07-10 22:00:00'): 32.1,
#                                                    pd.Timestamp('2017-07-10 23:00:00'): 7.7}})
#                            ]
#             self.output = self.inputs[0].combine_first(self.inputs[1])
#             self.funcs = ['df.combine_first']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/49572546
#     class SO_49572546_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(
#                     {'C1': {1: 100, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107},
#                      'C2': {1: 201, 2: 202, 3: 203, 4: 204, 5: 205, 6: 206, 7: 207},
#                      'C3': {1: 301, 2: 302, 3: 303, 4: 304, 5: 305, 6: 306, 7: 307}}),
#                 pd.DataFrame(
#                     {'C1': {2: '1002', 3: 'v1', 4: 'v4', 7: '1007'}, 'C2': {2: '2002', 3: 'v2', 4: 'v5', 7: '2007'},
#                      'C3': {2: '3002', 3: 'v3', 4: 'v6', 7: '3007'}})
#             ]
#             self.output = self.inputs[1].combine_first(self.inputs[0])
#             self.funcs = ['df.combine_first']
#             self.seqs = [[0]]

#     class SO_12860421_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(columns=['X', 'Y', 'Z'],
#                              index=[4, 5, 6, 7],
#                              data=[['X1', 'Y2', 'Z3'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z2']]
#                              ),

#                 pd.Series.nunique
#             ]
#             self.output = self.inputs[0].pivot_table(values='X', index='Y', columns='Z', aggfunc=pd.Series.nunique)

#             self.funcs = ['df.pivot_table']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/13261175
#     class SO_13261175_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             df = pd.DataFrame({'name': ['A', 'B', 'A', 'B'], 'type': [11, 11, 12, 12],
#                                'date': ['2012-01-01', '2012-01-01', '2012-02-01', '2012-02-01'], 'value': [4, 5, 6, 7]})

#             pt = df.pivot_table(values='value', index='name', columns=['type', 'date'])
#             self.inputs = [df]
#             self.output = pt
#             self.funcs = ['df.pivot_table']
#             self.seqs = [[0]]

#     # https://stackoverflow.com/questions/13793321
#     class SO_13793321_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame([[11, 12, 13]], columns=[10, 1, 2]),
#                 pd.DataFrame([[11, 37, 38], [34, 19, 39]], columns=[10, 3, 4])
#             ]
#             self.output = self.inputs[0].merge(self.inputs[1], on=10)
#             self.funcs = ['df.merge']
#             self.seqs = [[0]]

#     class SO_14085517_depth1(Benchmark):
#         def __init__(self):
#             super().__init__()
#             text = '''\
# SEGM1\tDESC\tDistribuzione Ponderata\tRotazioni a volume
# AD2\tACCADINAROLO\t74.040\t140249.693409
# AD1\tZYMIL AMALAT Z\t90.085\t321529.053570
# FUN\tSPECIALMALAT S\t88.650\t120711.182177
# NORM5\tSTD INNAROLO\t49.790\t162259.216710
# NORM4\tSTD P.NAROLO\t52.125\t1252174.695695
# NORM3\tSTD PLNAROLO\t54.230\t213257.829615
# NORM1\tBONTA' MALAT B\t79.280\t520454.366419
# NORM6\tDA STD RILGARD\t35.290\t554927.497875
# NORM7\tOVANE VT.MANTO\t15.040\t466232.639628
# NORM2\tWEIGHT MALAT W\t79.170\t118628.572692
# '''
#             from io import StringIO
#             a = pd.read_csv(StringIO(text), delimiter='\t',
#                             index_col=(0, 1), )
#             self.inputs = [a]
#             self.output = a.sort_values(['SEGM1', 'Distribuzione Ponderata'], ascending=[True, False])
#             self.seqs = [[0]]
#             self.funcs = ['df.sort_values']

#     class SO_11418192_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame(data=[[5, 7], [6, 8], [-1, 9], [-2, 10]], columns=['a', 'b']),
#                            lambda x: x['a'] > 1, 'a > 1']
#             t = self.inputs[0]
#             self.output = t[t.apply(lambda x: x['a'] > 1, axis=1)]
#             self.funcs = ['df.apply', 'df.__getitem__']
#             self.seqs = [[0, 1]]

#     class SO_49567723_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'id': {0: 255, 1: 91, 2: 347, 3: 30, 4: 68, 5: 159, 6: 32, 7: 110, 8: 225, 9: 257},
#                               'valueA': {0: 1141, 1: 1130, 2: 830, 3: 757, 4: 736, 5: 715, 6: 713, 7: 683, 8: 638,
#                                          9: 616}}),
#                 pd.DataFrame({'id': {0: 255, 1: 91, 2: 5247, 3: 347, 4: 30, 5: 68,
#                                      6: 159, 7: 32, 8: 110, 9: 225, 10: 257,
#                                      11: 917, 12: 211, 13: 25},
#                               'valueB': {0: 1231, 1: 1170, 2: 954, 3: 870, 4: 757,
#                                          5: 736, 6: 734, 7: 713, 8: 683, 9: 644,
#                                          10: 616, 11: 585, 12: 575, 13: 530}}),
#                 'valueA != valueB'
#             ]
#             self.output = self.inputs[0].merge(self.inputs[1], on=['id']).query('valueA != valueB')
#             self.funcs = ['df.merge', 'df.query']
#             self.seqs = [[0, 1]]

#     # https://stackoverflow.com/questions/49987108
#     class SO_49987108_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#                               'COL': [23, np.nan, np.nan, np.nan, np.nan, 21, np.nan, np.nan, np.nan, 25, np.nan,
#                                       np.nan]}).set_index('ID'),
#                 int
#             ]
#             self.output = self.inputs[0].fillna(method='ffill').astype(int)
#             self.seqs = [[0, 1]]
#             self.funcs = ['df.fillna', 'df.astype']

#     # https://stackoverflow.com/questions/13261691
#     # (there's also another potential q/a pair in this question)
#     # or this could just be done with a sort????
#     class SO_13261691_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(
#                     {'date': ['3/9/12', '3/10/12', '4/9/12', '9/9/12', '11/9/12', '30/9/12', '31/10/12', '1/11/12'],
#                      'score': [100, 99, 102, 103, 111, 98, 103, 104]}, index=pd.MultiIndex.from_tuples(
#                         [('A', 'John1'), ('B', 'John2'), ('B', 'Jane'), ('A', 'Peter'), ('C', 'Josie'),
#                          ('A', 'Rachel'),
#                          ('B', 'Kate'), ('C', 'David')], names=['team', 'name']))
#             ]
#             self.output = self.inputs[0].stack().unstack()
#             self.funcs = ['df.stack', 'df.unstack']
#             self.seqs = [[0, 1]]

#     class SO_13659881_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(
#                     columns=['ip', 'useragent'],
#                     index=[0, 1, 2, 3],
#                     data=[['192.168.0.1', 'a'], ['192.168.0.1', 'a'], ['192.168.0.1', 'b'], ['192.168.0.2', 'b']]
#                 )
#             ]
#             self.output = self.inputs[0].groupby(['ip', 'useragent']).size()
#             self.funcs = ['df.groupby', 'dfgroupby.size']
#             self.seqs = [[0, 1]]

#     # https://stackoverflow.com/questions/13807758
#     class SO_13807758_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             df1 = pd.DataFrame([[10], [11], [12], [14], [16], [18]])
#             df1[::3] = np.nan
#             self.inputs = [
#                 df1
#             ]
#             self.output = self.inputs[0].dropna().reset_index(drop=True)
#             self.funcs = ['df.dropna', 'df.reset_index']
#             self.seqs = [[0, 1]]

#     # http://stackoverflow.com/questions/34365578/dplyr-filtering-based-on-two-variables
#     class SO_34365578_depth2(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'Group': {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'},
#                               'Id': {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16},
#                               'Var1': {0: 'good', 1: 'good', 2: 'bad', 3: 'good', 4: 'good', 5: 'bad'},
#                               'Var2': {0: 20, 1: 26, 2: 29, 3: 23, 4: 23, 5: 28}}),
#                 "Group == \"A\"",
#                 'sum',
#             ]
#             self.output = self.inputs[0].query('Group == "A"').pivot_table(index='Group', columns='Var1', values='Var2',
#                                                                            aggfunc='sum')
#             self.funcs = ['df.query', 'df.pivot_table']
#             self.seqs = [[0, 1]]

#       # https://stackoverflow.com/questions/10982266
#     class SO_10982266_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame(
#                 [['08:01:08', 'C', 'PXA', 20100101, 4000, 'A', 57.8, 60],
#                  ['08:01:11', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
#                  ['08:01:12', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
#                  ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
#                  ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
#                  ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
#                  ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60]],
#                 columns=['time', 'contract', 'ticker', 'expiry', 'strike', 'quote', 'price', 'volume'],
#                 index=[0, 1, 2, 3, 4, 5, 6]
#             )]
#             self.output = pd.DataFrame(
#                 [['08:01:08', 57.8, 60], ['08:01:11', 58.4, 60], ['08:01:12', 58.0, 60], ['08:01:16', 58.2, 60],
#                  ['08:01:21', 58.2, 60]],
#                 columns=['time', 'price', 'volume'],
#                 index=[0, 1, 2, 3, 4]
#             )
#             self.funcs = ['df.groupby', 'dfgroupby.mean', 'df.__getitem__']
#             self.seqs = [[0, 1, 2]]
#             # original answer for input a:
#             # pd.DataFrame([{'time': k,
#             #                'price': (v.price * v.volume).sum() / v.volume.sum(),
#             #                'volume': v.volume.mean()}
#             #               for k, v in a.groupby(['time'])],
#             #              columns=['time', 'price', 'volume'])

#     # https://stackoverflow.com/questions/11811392
#     class SO_11811392_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [pd.DataFrame(
#                 columns=['one', 'two', 'three', 'four', 'five'],
#                 index=[0, 1],
#                 data=[[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
#             )]
#             # original has a tolist at the end, but we don't support that
#             self.output = self.inputs[0].T.reset_index().values
#             self.funcs = ['df.T', 'dfgroupby.reset_index', 'df.values']
#             self.seqs = [[0, 1, 2]]

#     # https://stackoverflow.com/questions/49581206
#     class SO_49581206_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'A': {('col1', 'no'): 2, ('col1', 'yes'): 8, ('col2', 'no'): 2, ('col2', 'yes'): 6},
#                               'B': {('col1', 'no'): 0, ('col1', 'yes'): 2, ('col2', 'no'): 1, ('col2', 'yes'): 1}}).T
#             ]
#             self.output = self.inputs[0].div(self.inputs[0].sum(1, level=0), 1, 0).xs('yes', 1, 1)

#             self.funcs = ['df.sum', 'df.div', 'df.xs']
#             self.seqs = [[0, 1, 2]]

#     # https://stackoverflow.com/questions/12065885
#     class SO_12065885_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({'RPT_Date': {0: '1980-01-01', 1: '1980-01-02', 2: '1980-01-03', 3: '1980-01-04',
#                                            4: '1980-01-05', 5: '1980-01-06', 6: '1980-01-07', 7: '1980-01-08',
#                                            8: '1980-01-09', 9: '1980-01-10'},
#                               'STK_ID': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
#                               'STK_Name': {0: 'Arthur', 1: 'Beate', 2: 'Cecil', 3: 'Dana', 4: 'Eric', 5: 'Fidel',
#                                            6: 'George', 7: 'Hans', 8: 'Ingrid', 9: 'Jones'},
#                               'sales': {0: 0, 1: 4, 2: 2, 3: 8, 4: 4, 5: 5, 6: 4, 7: 7, 8: 7, 9: 4}}),
#                 [[4, 2, 6]]
#             ]
#             self.output = self.inputs[0][self.inputs[0].STK_ID.isin([4, 2, 6])]
#             self.funcs = ['df.isin', 'df.getitem', 'df.loc_getitem']
#             self.seqs = [[2, 0, 1]]

#     # https://stackoverflow.com/questions/13576164
#     class SO_13576164_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(columns=['col1', 'to_merge_on'],
#                              index=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']],
#                                                              names=['id1', 'id2']),
#                              data=[[1, 2], [3, 4], [1, 2], [3, 4]]),
#                 pd.DataFrame(columns=['col2', 'to_merge_on'],
#                              index=[0, 1, 2],
#                              data=[[1, 1], [2, 3], [3, 5]])
#             ]
#             self.output = self.inputs[0].reset_index().merge(self.inputs[1], how='left').set_index(
#                 ['id1', 'id2'])
#             self.funcs = ['df.reset_index', 'df.merge', 'df.set_index']
#             self.seqs = [[0, 1, 2]]

#     # https://stackoverflow.com/questions/14023037
#     class SO_14023037_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(
#                     {'id': [1, 2, 3, 4, 5, 6],
#                      'col1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2'],
#                      'col2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B2'],
#                      'col3': ['before', 'after', 'before', 'after', 'before', 'after'],
#                      'value': [20, 13, 11, 21, 18, 22]},
#                     columns=['id', 'col1', 'col2', 'col3', 'value'])
#             ]
#             self.output = self.inputs[0].pivot_table(values='value',
#                                                      index=['col1', 'col2'],
#                                                      columns=['col3']).fillna(method='bfill').dropna()
#             self.funcs = ['df.pivot_table', 'df.fillna', 'df.dropna']
#             self.seqs = [[0, 1, 2]]

#     # https://stackoverflow.com/questions/53762029/pandas-groupby-and-cumsum-on-a-column
#     class SO_53762029_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             data = """
# doc_created_month   doc_created_year    speciality      doc_id_count
# 8                   2016                Acupuncturist   1           
# 2                   2017                Acupuncturist   1           
# 4                   2017                Acupuncturist   1           
# 4                   2017                Allergist       1           
# 5                   2018                Allergist       1           
# 10                  2018                Allergist       2   
# """

#             df = pd.read_csv(StringIO(data), sep='\s+')
#             self.inputs = [df]
#             self.output = df.groupby(['doc_created_month', 'doc_created_year', 'speciality']).sum().cumsum()
#             self.funcs = ['df.groupby', 'dfgroupby.sum', 'df.cumsum']
#             self.seqs = [[0, 1, 2]]

#             # http://stackoverflow.com/questions/21982987/mean-per-group-in-a-data-frame

#     class SO_21982987_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({"Name": ["Aira", "Aira", "Ben", "Ben", "Cat", "Cat"], "Month": [1, 2, 1, 2, 1, 2],
#                               "Rate1": [12, 18, 53, 22, 22, 27], "Rate2": [23, 73, 19, 87, 87, 43]}),
#             ]
#             self.output = pd.DataFrame({'Name': {0: 'Aira', 1: 'Ben', 2: 'Cat'}, 'Rate1': {0: 15.0, 1: 37.5, 2: 24.5},
#                                         'Rate2': {0: 48.0, 1: 53.0, 2: 65.0}})
#             self.seqs = [[0, 1, 2]]
#             self.funcs = ['df.groupby', 'dfgroupby.mean', 'df.drop']

#     # http://stackoverflow.com/questions/39656670/pivot-table-on-r-using-dplyr-or-tidyr
#     class SO_39656670_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame(
#                     {"Player": ["Abdoun", "Abe", "Abidal", "Abreu"], "Team": ["Algeria", "Japan", "France", "Uruguay"],
#                      "Shots": [0, 3, 0, 5], "Passes": [6, 101, 91, 15], "Tackles": [0, 14, 6, 0]}),
#             ]
#             self.output = self.inputs[0].melt(value_vars=["Passes", "Tackles"], var_name="Var",
#                                               value_name="Mean").groupby(
#                 "Var", as_index=False).mean()
#             self.seqs = [[0, 1, 2]]
#             self.funcs = ['df.melt', 'df.groupby', 'dfgroupby.mean']

#     # http://stackoverflow.com/questions/23321300/efficient-method-to-filter-and-add-based-on-certain-conditions-3-conditions-in
#     class SO_23321300_depth3(Benchmark):
#         def __init__(self):
#             super().__init__()
#             self.inputs = [
#                 pd.DataFrame({"a": [1, 1, 1, 1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 2, 2, 2, 3],
#                               "d": [0, 200, 300, 0, 600, 0, 100, 200, 0]}),
#                 'd > 0'
#             ]
#             self.output = self.inputs[0].query('d > 0').groupby(['a', 'b']).mean()
#             self.funcs = ['df.query', 'df.groupby', 'dfgroupby.mean']
#             self.seqs = [[0, 1, 2]]
