#!/usr/bin/python
# -*- coding: utf-8 -*-

# BUILT-IN PACKAGES
import datetime as dt
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from scipy.interpolate import CubicSpline
from matplotlib.patches import Rectangle
from typing import Tuple, List, Any, Dict, Optional
from datetime import datetime

# PROJECT PACKAGES
from logs_generator import generate_logs

# TURN WARNINGS OFF
# warnings.filterwarnings("ignore")


# CLASSES
class PolynomialMethod:
    def __init__(self, logs: pd.DataFrame, case_id_text: str = 'case_id', task_text: str = 'task_name',
                 instance_text: str = 'instance', start_timestamp_text: str = 'start_timestamp',
                 complete_timestamp_text: str = 'complete_timestamp') -> None:
        """
        Constructor
        """

        # Initialize dataframe of logs
        self.logs: pd.DataFrame = logs
        self.two_logs_dataframe: Optional[pd.DataFrame] = None

        # Initialize the names of column in logs dataframe
        self.case_id_text: str = case_id_text
        self.task_text: str = task_text
        self.instance_text: str = instance_text
        self.start_timestamp_text: str = start_timestamp_text
        self.complete_timestamp_text: str = complete_timestamp_text

        # Calculate other useful parameters
        self.list_of_tasks: List[str] = list(self.logs[self.task_text].unique())
        self.number_of_traces: Dict[Any] = self.logs.groupby([self.task_text]).count().iloc[:, 0].to_dict()

        # self.minimum_start_timestamp: datetime = dt.datetime(2000, 1, 1)
        # self.maximum_complete_timestamp: datetime = dt.datetime(2010, 1, 1)
        self.minimum_start_timestamp: datetime = self.logs[self.start_timestamp_text].min()
        self.maximum_complete_timestamp: datetime = self.logs[self.complete_timestamp_text].max()

    def prepare_dataframe_with_two_tasks_to_compare(self, task1: str, task2: str, instance: str = None) -> None:
        is_first_task_logs = self.logs[self.task_text] == task1
        is_second_task_logs = self.logs[self.task_text] == task2
        is_selected_instance = self.logs[self.instance_text] == instance if instance else True
        self.two_logs_dataframe = self.logs.loc[is_first_task_logs | is_second_task_logs & is_selected_instance]

    def gets_tasks_names_in_time_relation(self) -> List[str]:
        # dodać funkcję sortującą taski w kolejności występowania
        task_means = self.two_logs_dataframe.groupby(self.task_text)[self.start_timestamp_text].mean()

        return task_means.sort_values().index.values

    def moved_logs_to_side(self) -> None:
        task1, task2 = self.gets_tasks_names_in_time_relation()
        case1 = self.two_logs_dataframe[self.task_text] == task1
        case2 = self.two_logs_dataframe[self.task_text] == task2

        # Calculate difference for each group
        diff = self.two_logs_dataframe.complete_timestamp.min() - self.two_logs_dataframe[case1].groupby(self.case_id_text).complete_timestamp.min()

        # merge the differences back into the original dataframe
        self.two_logs_dataframe = self.two_logs_dataframe.merge(diff.to_frame('diff'), left_on=self.case_id_text,
                                                                right_index=True)

        # Update start and complete timestamps for each group
        self.two_logs_dataframe.loc[case1, 'start_timestamp'] += self.two_logs_dataframe.loc[case1, 'diff']
        self.two_logs_dataframe.loc[case2, 'start_timestamp'] += self.two_logs_dataframe.loc[case2, 'diff']
        self.two_logs_dataframe.loc[case1, 'complete_timestamp'] += self.two_logs_dataframe.loc[case1, 'diff']
        self.two_logs_dataframe.loc[case2, 'complete_timestamp'] += self.two_logs_dataframe.loc[case2, 'diff']

        # Drop the diff column
        self.two_logs_dataframe = self.two_logs_dataframe.drop('diff', axis=1)

        self.minimum_start_timestamp = self.two_logs_dataframe[self.start_timestamp_text].min() - dt.timedelta(10)
        self.maximum_complete_timestamp = self.two_logs_dataframe[self.complete_timestamp_text].max() + dt.timedelta(10)

    def print_results(self, area1: float, area2: float, area_conv: float) -> None:
        task1, task2 = self.gets_tasks_names_in_time_relation()
        print(45 * "= ")
        print("{0} area:\033[1m {1} \033[0m".format(task1, round(area1, 2)))
        print("{0} area:\033[1m {1} \033[0m".format(task2, round(area2, 2)))
        print("conv area:\033[1m {} \033[0m".format(round(area_conv, 2)))
        print("\nconv/task1:\033[1m {} % \033[0m".format(round(100 * area_conv / area1), 2))
        print("conv/task2:\033[1m {} % \033[0m".format(round(100 * area_conv / area2), 2))
        print("")

    def plot_task_duration(self, ax: plt.Axes, color: Any = 'r', label=''):
        """
        Method to generate the time flow of one analyzed task
        """

        for i in range(self.two_logs_dataframe.shape[0] - 1):
            x1: float = np.array(self.two_logs_dataframe[self.start_timestamp_text])[i]
            x2: float = np.array(self.two_logs_dataframe[self.complete_timestamp_text])[i]
            y: float = np.array(self.two_logs_dataframe[self.case_id_text])[i]
            ax.plot([x1, x2], [y, y], color=color, alpha=0.5, label=label)

        ax.get_yaxis().set_visible(False)
        ax.set(xlim=[self.minimum_start_timestamp, self.maximum_complete_timestamp])

    def plot_tasks(self, colors: List[str]):
        """
        Method to generate the time flow of two analyzed tasks
        """

        fig, ax = plt.subplots(figsize=(14, 6))
        task1, task2 = self.gets_tasks_names_in_time_relation()
        self.plot_task_duration(ax=ax, color=colors[0], label=task1)
        self.plot_task_duration(ax=ax, color=colors[1], label=task2)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        legend_handles = []
        legend_labels = []

        for label in unique_labels:
            index = labels.index(label)
            legend_handles.append(handles[index])
            legend_labels.append(labels[index])

        plt.legend(legend_handles, legend_labels)
        plt.tight_layout()
        plt.show()

    def combine_task_intervals(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        df = self.two_logs_dataframe[self.two_logs_dataframe[self.task_text] == task].reset_index()
        timestamps = np.unique(np.array(sorted(list(df[self.start_timestamp_text]) + list(df[self.complete_timestamp_text]))))
        frequencies = np.array([0] * len(timestamps))

        for i in range(len(df)):
            start_ts = df[self.start_timestamp_text][i]
            complete_ts = df[self.complete_timestamp_text][i]
            start_idx = list(timestamps).index(start_ts)
            complete_idx = list(timestamps).index(complete_ts)

            for j in range(start_idx, complete_idx + 1):
                frequencies[j] = frequencies[j] + 1

        return np.array([t.timestamp() for t in timestamps]), frequencies

    def calculate_timestamp_metrics(self, task1, task2, timestamps1, frequencies1, timestamps2, frequencies2, colors,
                                    plot_result: bool = False) -> Tuple[float, float, float]:
        fig = plt.figure(figsize=(30, 10))
        task_list = [task1, task2]
        timestamps_list = [timestamps1, timestamps2]
        frequencies_list = [frequencies1, frequencies2]
        approx_list = []
        area_list = []

        for i in range(2):
            ax = fig.add_subplot(1, 3, i + 1)
            task = task_list[i]
            timestamps = timestamps_list[i]
            frequencies = frequencies_list[i]
            N = len(timestamps)
            approx = CubicSpline(np.array(timestamps), np.array(frequencies), bc_type='natural')
            approx_list.append(approx)
            area_rect = sum([((timestamps[i + 1] - timestamps[i]) * frequencies[i]) for i in range(N - 1)])
            area_poly = approx.integrate(timestamps[0], timestamps[-1])
            area_list.append(area_poly)

            if plot_result:
                for j in range(N - 1):
                    t1 = timestamps[j]
                    t2 = timestamps[j + 1]
                    freq = frequencies[j]
                    ax.add_patch(Rectangle((t1, 0), t2 - t1, freq, color=colors[i], label=task))

                ax.plot(timestamps, approx(np.array(timestamps)), label="Approximation")
                ax.set_xlim([min(timestamps), max(timestamps)])
                ax.set_title(f"Area calculated by rectangles: {area_rect}\nArea calculate by quad: {area_poly}\nRatio: {area_rect / area_poly}")

        t1 = (timestamps_list[0][0], timestamps_list[0][-1])
        t2 = (timestamps_list[1][0], timestamps_list[1][-1])
        conv_fun = lambda x: min([approx_list[0](x), approx_list[1](x)])
        area1 = float(area_list[0])
        area2 = float(area_list[1])
        area_conv = scipy.integrate.quad(conv_fun, max([t1[0], t2[0]]), min([t1[1], t2[1]]))[0]

        if plot_result:
            ax = fig.add_subplot(1, 3, 3)
            ax.plot(timestamps_list[0], approx_list[0](timestamps_list[0]), label=task_list[0])
            ax.plot(timestamps_list[1], approx_list[1](timestamps_list[1]), label=task_list[1])
            # show_legend(fig)
            plt.show()

        return area1, area2, area_conv

    def core_metod(self, task1: str, task2: str, instance: str = None, plot_steps: bool = False,
                   plot_results: bool = False, print_results: bool = False) -> List[str]:

        self.prepare_dataframe_with_two_tasks_to_compare(task1, task2, instance=instance)

        if plot_steps:
            colors = ['red', 'blue']
            self.plot_tasks(colors)
            self.moved_logs_to_side()
            self.plot_tasks(colors)

        else:
            self.moved_logs_to_side()

        timestamps1, frequencies1 = self.combine_task_intervals(task1)
        timestamps2, frequencies2 = self.combine_task_intervals(task2)
        area1, area2, area_conv = self.calculate_timestamp_metrics(task1, task2, timestamps1, frequencies1, timestamps2,
                                                                   frequencies2, ['red', 'blue'],
                                                                   plot_result=plot_results)

        if print_results:
            self.print_results(area1, area2, area_conv)

        return return_possible_relation(area_conv / area1, area_conv / area2)


def return_possible_relation(p_value1: float, p_value2: float) -> List[str]:
    if p_value1 == 0 and p_value2 == 0:
        possible_relations = ['meets', 'before']

    elif p_value1 == 1:
        if p_value2 == 1:
            possible_relations = ['equals']

        else:
            possible_relations = ['contains', 'starts']

    elif (p_value1 == 1 and p_value2 <= 0.9) or (p_value1 <= 0.9 and p_value2 == 1):
        possible_relations = ['contains', 'starts']

    elif 0 < p_value1 <= 0.9 and 0 < p_value2 <= 0.9:
        possible_relations = ['meets', 'starts', 'overlaps']

    else:
        possible_relations = ['meets', 'starts', 'before', 'overlaps', 'contains', 'equals']

    return possible_relations
