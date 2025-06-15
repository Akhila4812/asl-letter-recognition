import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Generate data with named columns
columns = [f"pixel_{i}" for i in range(28*28)]
X_train = pd.DataFrame(np.random.rand(100, 28*28), columns=columns)
user_inputs = pd.DataFrame(np.random.rand(10, 28*28), columns=columns)

# Run drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=user_inputs)
report.save_html("drift_report.html")