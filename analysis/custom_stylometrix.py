import stylo_metrix as sm
import numpy as np
import pandas as pd


class CustomStyloMetrix(sm.StyloMetrix):

    def transform(self, doc, text):
        columns = ["text"]
        m_columns = [m.code for m in self._metrics]
        values = []
        debugs = []

        try:
            metric_values, metric_debugs = [], []
            for metric in self._metrics:
                value, debug = doc._.metrics[metric]
                metric_values.append(value)
                metric_debugs.append(debug)
        except Exception:
            metric_values = [np.nan] * len(self._metrics)
            metric_debugs = [{} for _ in self._metrics]

        values.append([text, *metric_values])
        debugs.append([text, *metric_debugs])

        if self._save_path:
            values_temp = pd.DataFrame(values, columns=columns + m_columns)
            debugs_temp = pd.DataFrame(debugs, columns=columns + m_columns)

            self._save(values_temp, f"{self.output_name}{self._file_number}_temp")
            if self._debug:
                self._save(debugs_temp, f"{self.debug_name}{self._file_number}_temp")

        values_df = pd.DataFrame(values, columns=columns + m_columns)
        debugs_df = pd.DataFrame(debugs, columns=columns + m_columns)

        if self._debug:
            if self._save_path:
                self._save(values_df, f"{self.output_name}{self._file_number}")
                self._save(debugs_df, f"{self.debug_name}{self._file_number}")
            return values_df, debugs_df
        else:
            if self._save_path:
                self._save(values_df, f"{self.output_name}{self._file_number}")
            return values_df