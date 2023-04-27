import datetime as dt
from datetime import datetime
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
from matplotlib.patches import Rectangle
from typing import Tuple, List, Any, Dict

warnings.filterwarnings("ignore")


class VerticalLogsAnalyzer:
    def __init__(self,
                 logs: pd.DataFrame,
                 case_id_column_name: str = 'case_id',
                 task_column_name: str = 'task_name',
                 instance_column_name: str = 'instance',
                 start_timestamp_column_name: str = 'start_timestamp',
                 complete_timestamp_column_name: str = 'complete_timestamp') -> None:
        """
        Initializes the VerticalLogAnalyzerViewer object with the logs dataframe and
        the names of columns in the logs dataframe.

        :param:
            logs (pd.DataFrame): The logs dataframe to be analyzed.
            case_id_column_name (str): The name of the column containing the case ids.
            task_column_name (str): The name of the column containing the task names.
            instance_column_name (str): The name of the column containing the instance ids.
            start_timestamp_column_name (str): The name of the column containing the start timestamps.
            complete_timestamp_column_name (str): The name of the column containing the complete timestamps.
        :return:
            None
        """

        # Initialize the logs dataframe and a new dataframe to store analysis results.
        self.logs: pd.DataFrame = logs
        self.analyzed_logs: pd.DataFrame = pd.DataFrame()

        # Initialize the names of columns in the logs dataframe.
        self.case_id_column_name: str = case_id_column_name
        self.task_column_name: str = task_column_name
        self.instance_column_name: str = instance_column_name
        self.start_timestamp_column_name: str = start_timestamp_column_name
        self.complete_timestamp_column_name: str = complete_timestamp_column_name

        # Calculate other useful parameters:
        self.list_of_tasks: List[str] = self.logs[self.task_column_name].unique()
        self.number_of_traces: Dict[Any] = self.logs.groupby([self.task_column_name]).count().iloc[:, 0].to_dict()

        # Calculate the range of timestamps in the logs dataframe.
        self.minimum_start_timestamp: datetime = self.logs[self.start_timestamp_column_name].min()
        self.maximum_complete_timestamp: datetime = self.logs[self.complete_timestamp_column_name].max()

    def select_data_for_analyze(self,
                                first_task: str,
                                second_task: str,
                                instance: str = None) -> None:
        """
        Prepare a subset of the DataFrame containing logs with two specific tasks to compare.

        :param:
            first_task (str): The name of the first task to compare.
            second_task (str): The name of the second task to compare.
            instance (str): The name of the instance to filter logs by (optional).
        :return:
            None
        """
        is_first_task_logs = self.logs[self.task_column_name] == first_task
        is_second_task_logs = self.logs[self.task_column_name] == second_task
        is_selected_instance = (self.logs[self.instance_column_name] == instance) if instance else True
        self.analyzed_logs = self.logs.loc[(is_first_task_logs | is_second_task_logs) & is_selected_instance]

    def select_time_order_tasks(self) -> List[str]:
        """
        Calculate the mean start timestamp for each task in the analyzed logs

        :return:
            A sorted list of tasks in the order of their mean timestamps
        """
        task_means = self.analyzed_logs.groupby(self.task_column_name)[self.start_timestamp_column_name].mean()
        return task_means.sort_values().index.values

    def update_range_of_timestamp(self) -> None:
        """
        Update the range of timestamps for the analyzed logs. The minimum start timestamp is set
        to the earliest start time in the analyzed logs minus 10 days, while the maximum complete
        timestamp is set to the latest complete time in the analyzed logs plus 10 days.

        :return:
            None
        """
        self.minimum_start_timestamp = self.analyzed_logs[self.start_timestamp_column_name].min() - dt.timedelta(10)
        self.maximum_complete_timestamp = self.analyzed_logs[self.complete_timestamp_column_name].max() + dt.timedelta(
            10)

    def moved_logs_to_side(self) -> None:
        """
        Adjusts start and complete timestamps of two tasks to align them with each other.
        The function finds the earliest task among the two, calculates the time difference
        between its completion timestamp and the completion timestamps of all cases in its
        group, and then adds this time difference to the start and complete timestamps of
        both tasks in the group. After the adjustments, it updates the range of the timestamp
        in the logs dataframe.

        :return:
            None
        """
        first_task, second_task = self.select_time_order_tasks()
        min_first_task_complete_timestamp = np.min(self.analyzed_logs.complete_timestamp)

        case1 = self.analyzed_logs[self.task_column_name] == first_task
        case2 = self.analyzed_logs[self.task_column_name] == second_task

        # calculate difference for each group
        diff = min_first_task_complete_timestamp - self.analyzed_logs[case1].groupby(
            self.case_id_column_name).complete_timestamp.min()

        # merge the differences back into the original dataframe
        self.analyzed_logs = self.analyzed_logs.merge(diff.to_frame('diff'), left_on=self.case_id_column_name,
                                                      right_index=True)

        # update start and complete timestamps for each group
        self.analyzed_logs.loc[case1, 'start_timestamp'] += self.analyzed_logs.loc[case1, 'diff']
        self.analyzed_logs.loc[case2, 'start_timestamp'] += self.analyzed_logs.loc[case2, 'diff']
        self.analyzed_logs.loc[case1, 'complete_timestamp'] += self.analyzed_logs.loc[case1, 'diff']
        self.analyzed_logs.loc[case2, 'complete_timestamp'] += self.analyzed_logs.loc[case2, 'diff']

        # drop the diff column
        self.analyzed_logs = self.analyzed_logs.drop('diff', axis=1)

        self.update_range_of_timestamp()

    def plot_task_duration(self, df: pd.DataFrame, ax: plt.Axes, color: Any = 'r', label=''):
        """
        Generate a plot of one task logs, like a line in time.

        :param:
            df (pd.DataFrame): The dataframe containing the task logs.
            ax (plt.Axes): The Matplotlib Axes object to plot on.
            color (Any, optional): The color to use for the plot. Defaults to 'r'.
            label (str, optional): The label for the plot. Defaults to ''.

        :return:
            None
        """
        for i in range(df.shape[0] - 1):
            ax.plot([np.array(df[self.start_timestamp_column_name])[i],
                     np.array(df[self.complete_timestamp_column_name])[i]],
                    [np.array(df[self.case_id_column_name])[i],
                     np.array(df[self.case_id_column_name])[i]], color=color, alpha=0.5,
                    label=label)
        ax.get_yaxis().set_visible(False)
        ax.set(xlim=[self.minimum_start_timestamp, self.maximum_complete_timestamp])

    def plot_all_tasks(self, colors: List = None):
        """
        Generates plot_task_duration for all tasks in logs

        :param
            colors: A list of colors to use for each task. If not provided, generates random colors.
        """
        if not colors:
            colors = [generate_random_color() for _ in range(len(self.list_of_tasks))]

        fig, ax = plt.subplots(figsize=[14, 6])
        list_of_tasks = list(self.list_of_tasks)
        for task in list_of_tasks:
            self.plot_task_duration(
                df=self.logs.loc[self.logs[self.task_column_name] == task],
                ax=ax,
                color=colors[list_of_tasks.index(task)],
                label=task)

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

    def plot_tasks(self, colors: List = None):
        """
        Generates plot_task_duration for two analyzed tasks.

        :param:
            colors: A list of colors to use for each task. If not provided, generates random colors.
        """
        if not colors:
            colors = [generate_random_color() for _ in range(len(self.list_of_tasks))]

        fig, ax = plt.subplots(figsize=[14, 6])
        list_of_tasks = list(self.select_time_order_tasks())
        for task in list_of_tasks:
            self.plot_task_duration(
                df=self.analyzed_logs.loc[self.analyzed_logs[self.task_column_name] == task],
                ax=ax,
                color=colors[list_of_tasks.index(task)],
                label=task)

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

    def print_results(self, area_first_task, area_second_task, area_splot):
        """
        Prints the results of area calculations and percentage ratios.

        :param:
            area_first_task (float): area of the first task
            area_second_task (float): area of the second task
            area_splot (float): area of the splot
        :return:
            None
        """
        first_task, second_task = self.select_time_order_tasks()
        print(45 * "= ")
        print("{0} area:\033[1m {1} \033[0m".format(first_task, round(area_first_task, 2)))
        print("{0} area:\033[1m {1} \033[0m".format(second_task, round(area_second_task, 2)))
        print("splot area:\033[1m {} \033[0m".format(round(area_splot, 2)))
        print("\nsplot/task1:\033[1m {} % \033[0m".format(round(100 * area_splot / area_first_task), 2))
        print("splot/task2:\033[1m {} % \033[0m".format(round(100 * area_splot / area_second_task), 2))
        print("")

    #
    #  TRAPEZOID METHODS
    #

    def calculate_trapezoid_metrics(self) -> Dict[str, Any]:
        """
        Calculate statistics for each task based on their start and complete timestamps.

        :return:
            A dictionary with task names as keys and their respective statistics as values.
            Statistics include mean and standard deviation of start and complete timestamps.
        """
        task_stats = {}
        for task in self.select_time_order_tasks():
            logs_task = self.analyzed_logs[self.analyzed_logs[self.task_column_name] == task]

            task_stats[task] = {
                'mean_start': np.mean(logs_task.start_timestamp),
                'std_start': np.std(logs_task.start_timestamp),
                'mean_complete': np.mean(logs_task.complete_timestamp),
                'std_complete': np.std(logs_task.complete_timestamp)
            }
        return task_stats

    def calculate_points(self, task_stats: Dict[str, Any], task: str) -> List[Tuple[Any, int]]:
        """
        Calculates four points list [A, B, C, D] for a given task based on its statistics.

        :param:
            task_stats: A dictionary containing statistics (mean and standard deviation) for each task.
            task: The task for which to calculate the points.
        :return:
            A list of four tuples representing the four points A, B, C, D for the given task, with each tuple
            containing two values: the x-coordinate (timestamp) and the y-coordinate (number of traces).
        """
        stats = task_stats[task]
        A = (stats['mean_start'] - stats['std_start'], 0)
        B = (stats['mean_start'], self.number_of_traces[task])
        C = (stats['mean_complete'], self.number_of_traces[task])
        D = (stats['mean_complete'] + stats['std_complete'], 0)
        return [A, B, C, D]

    def calculate_trapezoid_timestamp_metrics(self, plot_result: bool = False, is_big_dataframe: bool = False):
        """
        Calculates the area under the curves for two tasks and their intersection.
        The trapezoid function is calculated for each task using the mean start and
        complete times as well as their standard deviation. The function of the intersection
        is calculated as the minimum of the two trapezoid functions. The areas under
        each of the trapezoid functions and the intersection are then calculated.

        :param:
            plot_result (bool): Flag that determines whether to plot the resulting trapezoid function.
            is_big_dataframe (bool): Flag that determines whether to generate a big function using the .days.

        :return:
            Tuple of floats: A tuple containing the areas under the trapezoid function for the first task,
            the second task, and the intersection of the two tasks.
        """
        first_task, second_task = self.select_time_order_tasks()
        task_stats = self.calculate_trapezoid_metrics()

        tasks_function_values = {}
        tasks_function_range = pd.date_range(self.minimum_start_timestamp, self.maximum_complete_timestamp, freq='D')
        for task in self.select_time_order_tasks():
            trapezoid_points = self.calculate_points(task_stats=task_stats, task=task)
            if is_big_dataframe:
                function_of_trapezoid = generate_big_function(points=trapezoid_points)
            else:
                function_of_trapezoid = generate_function(points=trapezoid_points)
            tasks_function_values[task] = [function_of_trapezoid(_) for _ in tasks_function_range]

        if plot_result:
            self.plot_trapezoid(tasks_function_range=tasks_function_range, tasks_function_values=tasks_function_values)

        area_first_task = np.sum(tasks_function_values[first_task])
        area_second_task = np.sum(tasks_function_values[second_task])
        area_splot = np.sum(np.minimum(tasks_function_values[first_task], tasks_function_values[second_task]))

        return area_first_task, area_second_task, area_splot

    def core_trapezoid_method(self,
                              first_task: str,
                              second_task: str,
                              instance: str = None,
                              plot_steps: bool = False,
                              plot_results: bool = False,
                              print_results: bool = False,
                              is_big_dataframe: bool = False):
        """
        Perform core analysis for trapezoid method on the provided data and return a possible
        relation between the two tasks.

        :param:
        first_task: A string representing the name of the first task to compare.
            second_task: A string representing the name of the second task to compare.
            instance: An optional string representing the instance to analyze.
            plot_steps: A boolean flag indicating whether to generate plots of the tasks.
            plot_results: A boolean flag indicating whether to generate a plot of the results.
            print_results: A boolean flag indicating whether to print the results.
            is_big_dataframe: A boolean flag indicating whether the dataframe is big.
        :return:
            A list containing a possible relation between the splot/area of the first task
            and splot/area of the second task.
        """
        self.select_data_for_analyze(first_task, second_task, instance)
        if plot_steps:
            self.plot_tasks(colors=['red', 'blue'])
        self.moved_logs_to_side()
        if plot_steps:
            self.plot_tasks(colors=['red', 'blue'])
        area_first_task, area_second_task, area_splot = self.calculate_trapezoid_timestamp_metrics(
            plot_result=plot_results,
            is_big_dataframe=is_big_dataframe)

        if area_first_task == 0 or area_second_task == 0:
            print(45 * "= ")
            print("{0} area:\033[1m {1} \033[0m".format(first_task, round(area_first_task, 2)))
            print("{0} area:\033[1m {1} \033[0m".format(second_task, round(area_second_task, 2)))
            return []
        if print_results: self.print_results(area_first_task, area_second_task, area_splot)
        return return_possible_relation(area_splot / area_first_task, area_splot / area_second_task)

    def plot_trapezoid(self, tasks_function_range, tasks_function_values):
        """
        Plots the trapezoid function for two tasks.

        :param:
            tasks_function_range (np.ndarray): A numpy array containing the range of the tasks function.
            tasks_function_values (Dict): A dictionary containing the values of the tasks functions for each task.

        :return:
            None
        """
        first_task, second_task = self.select_time_order_tasks()
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=[14, 6])
        ax.plot(tasks_function_range, tasks_function_values[first_task], 'r--')
        ax.plot(tasks_function_range, tasks_function_values[second_task], 'b--')
        plt.legend(self.select_time_order_tasks())
        plt.tight_layout()
        plt.show()

    #
    # POLYNOMIAL METHOD
    #

    def combine_task_intervals(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        df = self.analyzed_logs[self.analyzed_logs[self.task_column_name] == task].reset_index()
        timestamps = np.unique(np.array(
            sorted(list(df[self.start_timestamp_column_name]) + list(df[self.complete_timestamp_column_name]))))
        frequencies = np.array([0] * len(timestamps))

        for i in range(len(df)):
            start_ts = df[self.start_timestamp_column_name][i]
            complete_ts = df[self.complete_timestamp_column_name][i]
            start_idx = list(timestamps).index(start_ts)
            complete_idx = list(timestamps).index(complete_ts)

            for j in range(start_idx, complete_idx + 1):
                frequencies[j] = frequencies[j] + 1

        return np.array([t.timestamp() for t in timestamps]), frequencies

    def calculate_polynomial_timestamp_metrics(self,
                                               task1,
                                               task2,
                                               timestamps1,
                                               frequencies1,
                                               timestamps2,
                                               frequencies2,
                                               colors,
                                               plot_result: bool = False) -> Tuple[float, float, float]:
        if plot_result:
            fig = plt.figure(figsize=(30, 10))

        task_list = [task1, task2]
        timestamps_list = [timestamps1, timestamps2]
        frequencies_list = [frequencies1, frequencies2]
        approx_list = []
        area_list = []

        for i in range(2):
            if plot_result:
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
                ax.set_title(
                    f"Area calculated by rectangles: {area_rect}\nArea calculate by quad: {area_poly}\nRatio: {area_rect / area_poly}")

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

    def core_polynomial_method(self,
                               first_task: str,
                               second_task: str,
                               instance: str = None,
                               plot_steps: bool = False,
                               plot_results: bool = False,
                               print_results: bool = False) -> List[str]:

        self.select_data_for_analyze(first_task, second_task, instance=instance)

        if plot_steps:
            colors = ['red', 'blue']
            self.plot_tasks(colors)
            self.moved_logs_to_side()
            self.plot_tasks(colors)

        else:
            self.moved_logs_to_side()

        timestamps1, frequencies1 = self.combine_task_intervals(first_task)
        timestamps2, frequencies2 = self.combine_task_intervals(second_task)
        area1, area2, area_conv = self.calculate_polynomial_timestamp_metrics(first_task,
                                                                              second_task,
                                                                              timestamps1,
                                                                              frequencies1,
                                                                              timestamps2,
                                                                              frequencies2,
                                                                              ['red', 'blue'],
                                                                              plot_result=plot_results)

        if print_results:
            self.print_results(area1, area2, area_conv)

        return return_possible_relation(area_conv / area1, area_conv / area2)

    def plot_polynomial(self):
        pass


def generate_function(points):
    """
    Returns a function that maps x to a value according to the given points.

    :param:
        points (List[Tuple]): A list of tuples representing (x, y) points on the function.
    :return:
        function (function): A function that takes a value of x and maps it to a corresponding y value.
    """

    def function(x):
        """
        Calculates the y value for a given x according to the points provided.

        :param:
            x (float): The x value to be mapped.
        :return:
            y (float): The corresponding y value.
        """
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


def generate_big_function(points):
    """
    The same function as generate_function but use a parametr .days for a big dataframe.
    It is because we must avoid overflowing the int capacity.

    :param:
        points (List[Tuple]): A list of tuples representing (x, y) points on the function.
    :return:
        function (function): A function that takes a value of x and maps it to a corresponding y value.
    """

    def function(x):
        if x < points[0][0]:
            return 0
        elif points[0][0] <= x < points[1][0]:
            return (x - points[0][0]).total_seconds() * (points[1][1] - points[0][1]) / (
                    points[1][0] - points[0][0]).total_seconds() + points[0][1]
        elif points[1][0] <= x < points[2][0]:
            return points[1][1]
        elif points[2][0] <= x < points[3][0]:
            return (x - points[2][0]).total_seconds() * (points[3][1] - points[2][1]) / (
                    points[3][0] - points[2][0]).total_seconds() + points[2][1]
        else:
            return 0

    return function


def generate_random_color():
    """
    Generate random colors for plotting function.
    :return:
        tuples of RGB colors
    """
    r = random.randint(0, 255)
    g = random.choice([0, 255])
    b = random.choice([0, 255])
    return r / 255.0, g / 255.0, b / 255.0


def return_possible_relation(p_value1: float, p_value2: float) -> List[str]:
    """
    This function takes two float values p_value1 and p_value2 as input and returns a list of possible relations
    between them. The output list contains the possible relations based on the values of p_value1 and p_value2.

    :param:
        p_value1: A float value representing the ratio of the intersection of the second task duration
        and the overlapping area to the first task duration.
        p_value2: A float value representing the ratio of the intersection of the first task duration
        and the overlapping area to the second task duration.
    :return:
        A list of possible relations between the two tasks based on the values of p_value1 and p_value2.
        Possible relations include 'meets', 'starts', 'before', 'overlaps', 'contains', and 'equals'.
    """
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
