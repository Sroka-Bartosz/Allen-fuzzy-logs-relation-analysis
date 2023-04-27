# Analysis of event relationships based on Allen’s algebra and fuzzy logic

This project was completed by a group of three people and involves the development of methods for analyzing
relationships that occur in business logs. The main goal was to create relation analysis of BPMN logs. The software also
can be used to optimize the task parallelization.

Three methods were created to determine which Allen relationships occur between two events in the log. The first method
is the trapezoid method based on the mean and standard deviation of task times. The second method uses approximation
function and the last method uses a matrix of the relationships. The first two methods allow elimination of incorrect
relationships and narrow the search range to a smaller number of relationships.

Although the methods require further work, they represent an innovative approach to this type of analysis.

## Usage

The project contains two main files, `vertical_logs_analyzer.py` and `relation_matrix.py`, which contain the three
methods described above. In addition, the Jupyter notebook files contain calls and analyses of the methods. The
file `method_times_comparison.jpynb` contains time analyses of the implemented methods.

## Authors

This project was created by Bartosz Sroka, Artur Mzyk, and Dominik Czyżyk, who are fourth-year students of Automation
and Robotics.

**Note:** This software is currently under development and requires several adjustments.