import abc
import numpy as np
import pandas as pd


class AllenRelationsFinderNaiveMethod(abc.ABC):
    def __init__(self,
                 logs):
        self.logs = logs
        task_names = self.logs["task_name"].unique()
        self.relation_matrix = pd.DataFrame(np.full([len(task_names), len(task_names)], "-"), index=task_names,
                                            columns=task_names)

    @staticmethod
    def find_allens_algebra_relation_between_tasks(left_task, right_task):
        if left_task["complete_timestamp"] < right_task["start_timestamp"]:
            return "<;>"
        elif left_task["complete_timestamp"] == right_task["start_timestamp"]:
            return "m;mi"
        else:
            if left_task["start_timestamp"] == right_task["start_timestamp"]:
                if left_task["complete_timestamp"] == right_task["complete_timestamp"]:
                    return "=;="
                else:
                    return "si;si"
            else:
                if left_task["complete_timestamp"] < right_task["complete_timestamp"]:
                    return "o;oi"
                elif left_task["complete_timestamp"] == right_task["complete_timestamp"]:
                    return "fi;f"
                else:
                    return "di;d"

    def create_relation_matrix(self, number_of_following_tasks_considered):
        case_ids = self.logs["case_id"].unique()
        for case_id in case_ids:
            logs_per_case_id = self.logs[self.logs["case_id"] == case_id].reset_index()

            for left_task_idx, left_task in logs_per_case_id.iterrows():
                for right_task_idx in range(left_task_idx + 1,
                                            min(left_task_idx + number_of_following_tasks_considered + 1,
                                                logs_per_case_id.shape[
                                                    0])):  # min used to avoid errors at the end of the DataFrame
                    right_task = logs_per_case_id.iloc[right_task_idx]

                    relation = self.find_allens_algebra_relation_between_tasks(left_task, right_task)
                    self._update_relation_matrix(left_task["task_name"], right_task["task_name"], relation)

        return self.relation_matrix

    def find_relations(self, relation_threshold):
        relations_occurence_rates_df = pd.DataFrame(columns=['Task 1', 'Task 2', 'Relations occurence rates'])
        founded_relations = pd.DataFrame(columns=['Task 1', 'Relation', 'Task 2'])

        for left_task_name, left_task_row in self.relation_matrix.iterrows():
            for right_task_name, relations_for_pair_of_tasks in left_task_row.items():
                relations_occurence_rates = dict(
                    (relation, relations_for_pair_of_tasks.count(relation) / len(relations_for_pair_of_tasks)) \
                    for relation in set(relations_for_pair_of_tasks) if relation != "-")
                relations_occurence_rates_df_row = pd.DataFrame(
                    {'Task 1': [left_task_name], 'Task 2': [right_task_name],
                     'Relations occurence rates': [relations_occurence_rates]})
                relations_occurence_rates_df = pd.concat(
                    [relations_occurence_rates_df, relations_occurence_rates_df_row], axis=0, ignore_index=True)

                relations_occurence_rates_filtered = [relation for (relation, occurence_rate) in
                                                      relations_occurence_rates.items() \
                                                      if occurence_rate >= relation_threshold and relation != "-"]

                for filtered_relation in relations_occurence_rates_filtered:
                    founded_relation = pd.DataFrame(
                        {'Task 1': [left_task_name], 'Relation': [filtered_relation], 'Task 2': [right_task_name]})
                    founded_relations = pd.concat([founded_relations, founded_relation], axis=0, ignore_index=True)

        # export DataFrame to txt file
        path = r'relations_occurence_rates.txt'
        with open(path, 'w') as f:
            relations_occurence_rates_string = relations_occurence_rates_df.to_string(header=False, index=False)
            f.write(relations_occurence_rates_string)

        return founded_relations

    def _update_relation_matrix(self, left_task_name, right_task_name, relation):
        relation, inverse_relation = relation.split(";")

        self.relation_matrix[left_task_name][right_task_name] = self.relation_matrix[left_task_name][
                                                                    right_task_name] + [relation] \
            if self.relation_matrix[left_task_name][right_task_name] != "-" else [relation]
        self.relation_matrix[right_task_name][left_task_name] = self.relation_matrix[right_task_name][
                                                                    left_task_name] + [inverse_relation] \
            if self.relation_matrix[right_task_name][left_task_name] != "-" else [inverse_relation]


class AllenRelationsFinderNaiveMethodWithLimitedRelations(abc.ABC):
    def __init__(self,
                 logs,
                 possible_relations):
        self.logs = logs

        if set(possible_relations) == set(['meets', 'before']):
            self.find_allens_algebra_relation_between_tasks = AllenRelationsFinderNaiveMethodWithLimitedRelations.meets_before
        elif set(possible_relations) == set(['meets', 'starts', 'overlaps']):
            self.find_allens_algebra_relation_between_tasks = AllenRelationsFinderNaiveMethodWithLimitedRelations.meets_starts_overlaps
        else:
            raise ValueError("Unsupported relations group")

        task_names = self.logs["task_name"].unique()
        self.relation_matrix = pd.DataFrame(np.full([len(task_names), len(task_names)], "-"), index=task_names,
                                            columns=task_names)

    @staticmethod
    def meets_before(left_task, right_task):
        if left_task["complete_timestamp"] < right_task["start_timestamp"]:
            return "<;>"
        else:
            return "m;mi"

    @staticmethod
    def meets_starts_overlaps(left_task, right_task):
        if left_task["complete_timestamp"] == right_task["start_timestamp"]:
            return "m;mi"
        elif left_task["start_timestamp"] == right_task["start_timestamp"]:
            return "si;si"
        else:
            return "o;oi"

    def create_relation_matrix(self, number_of_following_tasks_considered):
        case_ids = self.logs["case_id"].unique()
        for case_id in case_ids:
            logs_per_case_id = self.logs[self.logs["case_id"] == case_id].reset_index()

            for left_task_idx, left_task in logs_per_case_id.iterrows():
                for right_task_idx in range(left_task_idx + 1,
                                            min(left_task_idx + number_of_following_tasks_considered + 1,
                                                logs_per_case_id.shape[
                                                    0])):  # min used to avoid errors at the end of the DataFrame
                    right_task = logs_per_case_id.iloc[right_task_idx]

                    relation = self.find_allens_algebra_relation_between_tasks(left_task, right_task)
                    self._update_relation_matrix(left_task["task_name"], right_task["task_name"], relation)

        return self.relation_matrix

    def find_relations(self, relation_threshold):
        relations_occurence_rates_df = pd.DataFrame(columns=['Task 1', 'Task 2', 'Relations occurence rates'])
        founded_relations = pd.DataFrame(columns=['Task 1', 'Relation', 'Task 2'])

        for left_task_name, left_task_row in self.relation_matrix.iterrows():
            for right_task_name, relations_for_pair_of_tasks in left_task_row.items():
                relations_occurence_rates = dict(
                    (relation, relations_for_pair_of_tasks.count(relation) / len(relations_for_pair_of_tasks)) \
                    for relation in set(relations_for_pair_of_tasks) if relation != "-")
                relations_occurence_rates_df_row = pd.DataFrame(
                    {'Task 1': [left_task_name], 'Task 2': [right_task_name],
                     'Relations occurence rates': [relations_occurence_rates]})
                relations_occurence_rates_df = pd.concat(
                    [relations_occurence_rates_df, relations_occurence_rates_df_row], axis=0, ignore_index=True)

                relations_occurence_rates_filtered = [relation for (relation, occurence_rate) in
                                                      relations_occurence_rates.items() \
                                                      if occurence_rate >= relation_threshold and relation != "-"]

                for filtered_relation in relations_occurence_rates_filtered:
                    founded_relation = pd.DataFrame(
                        {'Task 1': [left_task_name], 'Relation': [filtered_relation], 'Task 2': [right_task_name]})
                    founded_relations = pd.concat([founded_relations, founded_relation], axis=0, ignore_index=True)

        # export DataFrame to txt file
        path = r'relations_occurence_rates.txt'
        with open(path, 'w') as f:
            relations_occurence_rates_string = relations_occurence_rates_df.to_string(header=False, index=False)
            f.write(relations_occurence_rates_string)

        return founded_relations

    def _update_relation_matrix(self, left_task_name, right_task_name, relation):
        relation, inverse_relation = relation.split(";")

        self.relation_matrix[left_task_name][right_task_name] = self.relation_matrix[left_task_name][
                                                                    right_task_name] + [relation] \
            if self.relation_matrix[left_task_name][right_task_name] != "-" else [relation]
        self.relation_matrix[right_task_name][left_task_name] = self.relation_matrix[right_task_name][
                                                                    left_task_name] + [inverse_relation] \
            if self.relation_matrix[right_task_name][left_task_name] != "-" else [inverse_relation]
