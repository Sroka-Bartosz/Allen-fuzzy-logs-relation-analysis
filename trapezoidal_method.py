import datetime as dt
from datetime import datetime
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Any, Dict
from logs_generator import generate_logs

warnings.filterwarnings("ignore")


class TrapezoidMethod:
    def __init__(self,
                 logs: pd.DataFrame,
                 case_id_column_name: str = 'case_id',
                 task_column_name: str = 'task_name',
                 instance_column_name: str = 'instance',
                 start_timestamp_column_name: str = 'start_timestamp',
                 complete_timestamp_column_name: str = 'complete_timestamp') -> None:
        # initialize dataframe of logs
        self.logs: pd.DataFrame = logs
        self.two_logs_dataframe: pd.DataFrame = pd.DataFrame()
        # initialize the names of column in logs dataframe
        self.case_id_column_name: str = case_id_column_name
        self.task_column_name: str = task_column_name
        self.instance_column_name: str = instance_column_name
        self.start_timestamp_column_name: str = start_timestamp_column_name
        self.complete_timestamp_column_name: str = complete_timestamp_column_name
        # calculate other useful parameters
        self.list_of_tasks: List[str] = self.logs[self.task_column_name].unique()
        self.number_of_traces: Dict[Any] = self.logs.groupby([self.task_column_name]).count().iloc[:, 0].to_dict()
        # self.minimum_start_timestamp: datetime = dt.datetime(2000, 1, 1)
        # self.maximum_complete_timestamp: datetime = dt.datetime(2010, 1, 1)
        self.minimum_start_timestamp: datetime = self.logs[self.start_timestamp_column_name].min()
        self.maximum_complete_timestamp: datetime = self.logs[self.complete_timestamp_column_name].max()

    def prepare_dataframe_with_two_tasks_to_compare(self,
                                                    first_task: str,
                                                    second_task: str,
                                                    instance: str = None) -> None:
        is_first_task_logs = self.logs[self.task_column_name] == first_task
        is_second_task_logs = self.logs[self.task_column_name] == second_task
        is_selected_instance = self.logs[self.instance_column_name] == instance if instance else True

        self.two_logs_dataframe = self.logs.loc[is_first_task_logs | is_second_task_logs & is_selected_instance]

    def gets_tasks_names_in_time_relation(self) -> List[str]:
        # dodać funkcję sortującą taski w kolejności występowania
        return self.two_logs_dataframe[self.task_column_name].unique()

    def update_range_of_timestamp(self) -> None:
        self.minimum_start_timestamp = self.two_logs_dataframe[
                                           self.start_timestamp_column_name].min() - dt.timedelta(10)
        self.maximum_complete_timestamp = self.two_logs_dataframe[
                                              self.complete_timestamp_column_name].max() + dt.timedelta(10)

    def moved_logs_to_side(self):
        # first_task, second_task = self.gets_tasks_names_in_time_relation()
        # min_first_task_complete_timestamp = np.min(self.two_logs_dataframe.complete_timestamp)
        #
        # for i in self.two_logs_dataframe[self.case_id_column_name].unique():
        #     case1 = ((self.two_logs_dataframe[self.case_id_column_name] == i) & (
        #             self.two_logs_dataframe[self.task_column_name] == first_task))
        #     case2 = ((self.two_logs_dataframe[self.case_id_column_name] == i) & (
        #             self.two_logs_dataframe[self.task_column_name] == second_task))
        #
        #     diff = min_first_task_complete_timestamp - pd.to_datetime(self.two_logs_dataframe[case1].complete_timestamp)
        #
        #     self.two_logs_dataframe.loc[case1, 'start_timestamp'] = self.two_logs_dataframe.loc[
        #                                                                 case1, 'start_timestamp'] + diff.item()
        #     self.two_logs_dataframe.loc[case2, 'start_timestamp'] = self.two_logs_dataframe.loc[
        #                                                                 case2, 'start_timestamp'] + diff.item()
        #     self.two_logs_dataframe.loc[case1, 'complete_timestamp'] = self.two_logs_dataframe.loc[
        #                                                                    case1, 'complete_timestamp'] + diff.item()
        #     self.two_logs_dataframe.loc[case2, 'complete_timestamp'] = self.two_logs_dataframe.loc[
        #                                                                    case2, 'complete_timestamp'] + diff.item()

        first_task, second_task = self.gets_tasks_names_in_time_relation()
        min_first_task_complete_timestamp = np.min(self.two_logs_dataframe.complete_timestamp)

        case1 = self.two_logs_dataframe[self.task_column_name] == first_task
        case2 = self.two_logs_dataframe[self.task_column_name] == second_task

        # calculate difference for each group
        diff = min_first_task_complete_timestamp - self.two_logs_dataframe[case1].groupby(
            self.case_id_column_name).complete_timestamp.min()

        # merge the differences back into the original dataframe
        self.two_logs_dataframe = self.two_logs_dataframe.merge(diff.to_frame('diff'), left_on=self.case_id_column_name,
                                                                right_index=True)

        # update start and complete timestamps for each group
        self.two_logs_dataframe.loc[case1, 'start_timestamp'] += self.two_logs_dataframe.loc[case1, 'diff']
        self.two_logs_dataframe.loc[case2, 'start_timestamp'] += self.two_logs_dataframe.loc[case2, 'diff']
        self.two_logs_dataframe.loc[case1, 'complete_timestamp'] += self.two_logs_dataframe.loc[case1, 'diff']
        self.two_logs_dataframe.loc[case2, 'complete_timestamp'] += self.two_logs_dataframe.loc[case2, 'diff']

        # drop the diff column
        self.two_logs_dataframe = self.two_logs_dataframe.drop('diff', axis=1)

        self.update_range_of_timestamp()

    def calculate_metrics(self) -> Dict[str, Any]:
        task_stats = {}
        for task in self.gets_tasks_names_in_time_relation():
            logs_task = self.two_logs_dataframe[self.two_logs_dataframe[self.task_column_name] == task]

            task_stats[task] = {
                'mean_start': np.mean(logs_task.start_timestamp),
                'std_start': np.std(logs_task.start_timestamp),
                'mean_complete': np.mean(logs_task.complete_timestamp),
                'std_complete': np.std(logs_task.complete_timestamp)
            }
        return task_stats

    def calculate_points(self, task_stats: Dict[str, Any], task: str) -> List[Tuple[Any, int]]:
        stats = task_stats[task]
        A = (stats['mean_start'] - stats['std_start'], 0)
        B = (stats['mean_start'], self.number_of_traces[task])
        C = (stats['mean_complete'], self.number_of_traces[task])
        D = (stats['mean_complete'] + stats['std_complete'], 0)
        return [A, B, C, D]

    def calculate_timestamp_metrics(self, plot_result: bool = False):
        first_task, second_task = self.gets_tasks_names_in_time_relation()
        task_stats = self.calculate_metrics()

        tasks_function_values = {}
        tasks_function_range = pd.date_range(self.minimum_start_timestamp, self.maximum_complete_timestamp, freq='D')
        for task in self.gets_tasks_names_in_time_relation():
            trapezoid_points = self.calculate_points(task_stats=task_stats, task=task)
            function_of_trapezoid = generate_function(points=trapezoid_points)
            tasks_function_values[task] = [function_of_trapezoid(_) for _ in tasks_function_range]

        if plot_result:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=[14, 6])
            ax.plot(tasks_function_range, tasks_function_values[first_task], 'r--')
            ax.plot(tasks_function_range, tasks_function_values[second_task], 'b--')
            plt.legend(self.gets_tasks_names_in_time_relation())
            plt.tight_layout()
            plt.show()

        area_first_task = np.sum(tasks_function_values[first_task])
        area_second_task = np.sum(tasks_function_values[second_task])
        area_splot = np.sum(np.minimum(tasks_function_values[first_task], tasks_function_values[second_task]))

        return area_first_task, area_second_task, area_splot

    def print_results(self, area_first_task, area_second_task, area_splot):

        first_task, second_task = self.gets_tasks_names_in_time_relation()
        print(45 * "= ")
        print("{0} area:\033[1m {1} \033[0m".format(first_task, round(area_first_task, 2)))
        print("{0} area:\033[1m {1} \033[0m".format(second_task, round(area_second_task, 2)))
        print("splot area:\033[1m {} \033[0m".format(round(area_splot, 2)))
        print("\nsplot/task1:\033[1m {} % \033[0m".format(round(100 * area_splot / area_first_task), 2))
        print("splot/task2:\033[1m {} % \033[0m".format(round(100 * area_splot / area_second_task), 2))
        print("")

    def plot_task_duration(self, ax: plt.Axes, color: Any = 'r', label=''):
        """
        generating a plot of one task logs, like a line in time.
        """
        for i in range(self.two_logs_dataframe.shape[0] - 1):
            ax.plot([np.array(self.two_logs_dataframe[self.start_timestamp_column_name])[i],
                     np.array(self.two_logs_dataframe[self.complete_timestamp_column_name])[i]],
                    [np.array(self.two_logs_dataframe[self.case_id_column_name])[i],
                     np.array(self.two_logs_dataframe[self.case_id_column_name])[i]], color=color, alpha=0.5,
                    label=label)
        ax.get_yaxis().set_visible(False)
        ax.set(xlim=[self.minimum_start_timestamp, self.maximum_complete_timestamp])

    def plot_tasks(self, colors: List = None):
        """
        generating plot_task_duration for two analyzed tasks
        """
        if not colors:
            colors = [generate_random_color() for _ in range(len(self.list_of_tasks))]

        fig, ax = plt.subplots(figsize=[14, 6])
        for task_idx in range(len(self.list_of_tasks)):
            self.plot_task_duration(ax=ax, color=colors[task_idx], label=self.list_of_tasks[task_idx])
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

    def core_metod(self, first_task: str, second_task: str, plot_steps: bool = False, plot_results: bool = False,
                   print_results: bool = False):
        self.prepare_dataframe_with_two_tasks_to_compare(first_task, second_task)
        if plot_steps: self.plot_tasks(colors=['red', 'blue'])
        self.moved_logs_to_side()
        if plot_steps: self.plot_tasks(colors=['red', 'blue'])
        area_first_task, area_second_task, area_splot = self.calculate_timestamp_metrics(plot_result=plot_results)
        if print_results: self.print_results(area_first_task, area_second_task, area_splot)
        return area_splot / area_first_task, area_splot / area_second_task


def generate_function(points):
    def function(x):
        if x < points[0][0]:
            return 0
        elif points[0][0] <= x < points[1][0]:
            return (x - points[0][0]) * (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]) + points[0][1]
        elif points[1][0] <= x < points[2][0]:
            return points[1][1]
        elif points[2][0] <= x < points[3][0]:
            return (x - points[2][0]) * (points[3][1] - points[2][1]) / (points[3][0] - points[2][0]) + points[2][1]
        else:
            return 0

    return function


def generate_random_color():
    r = random.randint(0, 255)
    g = random.choice([0, 255])
    b = random.choice([0, 255])
    return r / 255.0, g / 255.0, b / 255.0


if __name__ == "__main__":
    generated_logs = generate_logs('before')

    list_of_times = []
    list_of_fraction = [x for x in np.logspace(-2, 0, 100) if x > 0.05]
    for frac in list_of_fraction:
        t = TrapezoidMethod(logs=generated_logs.sample(frac=frac, random_state=42))
        start_time = time.time()
        p1, p2 = t.core_metod('A', 'B')
        execution_time = time.time() - start_time
        print(".", end="")
        list_of_times.append(execution_time)
    plt.plot(list_of_fraction, list_of_times)
    plt.show()