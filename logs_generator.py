import os
import sys
from datetime import datetime
from typing import Union, Dict
import warnings
import abc
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


class LogsGenerator(abc.ABC):
    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        self.generated_df = pd.DataFrame(columns=['case_id', 'task_name', 'start_timestamp', 'complete_timestamp'])
        self.minimal_date_in_generated_logs = pd.to_datetime(
            datetime.strptime(minimal_date_in_generated_logs, '%Y-%m-%d'))
        self.maximal_date_in_generated_logs = pd.to_datetime(
            datetime.strptime(maximal_date_in_generated_logs, '%Y-%m-%d'))
        self.level_of_differentiation_of_events = level_of_differentiation_of_events
        self.number_of_traces = number_of_traces
        self.first_task_duration = first_task_duration
        self.second_task_duration = second_task_duration

    @abc.abstractmethod
    def make_logs(self):
        ...

    def calc_scale(self) -> float:
        return np.random.uniform(-self.level_of_differentiation_of_events, self.level_of_differentiation_of_events)


class Generate_meets(LogsGenerator):
    """A meets B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.first_task_duration * duration
            complete_a = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration
            complete_b = complete_a + (1 + self.calc_scale()) * self.second_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': start_a, 'complete_timestamp': complete_a},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': complete_a, 'complete_timestamp': complete_b},
                ignore_index=True)
        return self.generated_df


class Generate_starts(LogsGenerator):
    """A starts B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.first_task_duration * duration
            complete_a = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration
            complete_b = start_a + (1 + self.calc_scale()) * self.second_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': start_a,
                 'complete_timestamp': np.min([complete_a, complete_b])},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': start_a,
                 'complete_timestamp': np.max([complete_a, complete_b])},
                ignore_index=True)
        return self.generated_df


class Generate_contains(LogsGenerator):
    """A contains B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.second_task_duration * duration
            complete_a = self.maximal_date_in_generated_logs + self.calc_scale() * self.second_task_duration * duration

            start_b = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration
            complete_b = complete_a - (1 + self.calc_scale()) * self.first_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': np.min([start_a, start_b]),
                 'complete_timestamp': np.max([complete_a, complete_b])},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': np.max([start_a, start_b]),
                 'complete_timestamp': np.min([complete_a, complete_b])},
                ignore_index=True)
        return self.generated_df


class Generate_before(LogsGenerator):
    """A before B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.first_task_duration * duration
            complete_a = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration
            start_b = complete_a + np.abs(self.calc_scale()) * duration
            complete_b = start_b + (1 + self.calc_scale()) * self.second_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': start_a, 'complete_timestamp': complete_a},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': start_b, 'complete_timestamp': complete_b},
                ignore_index=True)
        return self.generated_df


class Generate_overlaps(LogsGenerator):
    """A overlaps B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.first_task_duration * duration
            complete_a = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration
            start_b = complete_a - np.abs(self.calc_scale()) * duration * self.first_task_duration
            complete_b = start_b + (1 + self.calc_scale()) * self.second_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': start_a, 'complete_timestamp': complete_a},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': start_b, 'complete_timestamp': complete_b},
                ignore_index=True)
        return self.generated_df


class Generate_equals(LogsGenerator):
    """A equals B"""

    def __init__(self,
                 minimal_date_in_generated_logs: str,
                 maximal_date_in_generated_logs: str,
                 level_of_differentiation_of_events: float,
                 number_of_traces: int,
                 first_task_duration: int,
                 second_task_duration: int):
        super().__init__(minimal_date_in_generated_logs,
                         maximal_date_in_generated_logs,
                         level_of_differentiation_of_events,
                         number_of_traces,
                         first_task_duration,
                         second_task_duration)

    def make_logs(self):
        duration = self.maximal_date_in_generated_logs - self.minimal_date_in_generated_logs
        for i in range(self.number_of_traces):
            start_a = self.minimal_date_in_generated_logs - np.abs(
                self.calc_scale()) * self.first_task_duration * duration
            complete_a = start_a + (1 + self.calc_scale()) * self.first_task_duration * duration

            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'A', 'start_timestamp': start_a, 'complete_timestamp': complete_a},
                ignore_index=True)
            self.generated_df = self.generated_df.append(
                {'case_id': i, 'task_name': 'B', 'start_timestamp': start_a, 'complete_timestamp': complete_a},
                ignore_index=True)
        return self.generated_df


Generator = {
    'meets': Generate_meets,
    'starts': Generate_starts,
    'before': Generate_before,
    'overlaps': Generate_overlaps,
    'contains': Generate_contains,
    'equals': Generate_equals
}


class WrongOptionException(Exception):
    def __init__(self, options):
        self.options = options

    def __str__(self):
        return f"Wrong option selected. Available options: {self.options}"


def create_logs_generator(logs_generation_type: str, params: Dict) -> Union[
        Generate_meets, Generate_starts, Generate_before, Generate_overlaps, Generate_contains, Generate_equals]:
    options = list(Generator.keys())

    try:
        if logs_generation_type not in options:
            raise WrongOptionException(options)
        else:
            print(f"Generate relation 'A' {logs_generation_type} 'B' ")
    except WrongOptionException as e:
        print(e)

    logs_generator = Generator[logs_generation_type]

    return logs_generator(**params)


def rootRelPath(file: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), file))


def generate_logs(logs_generation_type: str) -> pd.DataFrame:
    with open(rootRelPath('config.yaml')) as f:
        try:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            print(e)
            sys.exit(-1)

    LogsGenerator = create_logs_generator(logs_generation_type, params=config)
    return LogsGenerator.make_logs()
