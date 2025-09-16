# app/dashboard.py
import streamlit as st
import pandas as pd
from attendance_logger import get_today_file, export_range_to_excel
from datetime import datetime, timedelta

st.title("Attendance Dashboard")

st.header("Today's attendance")
today_file = get_today_file()
if os.path.exists(today_file):
    df = pd.read_csv(today_file)
    st.dataframe(df)
    st.download_button("Download Today's CSV", df.to_csv(index=False), file_name=f"{today_file}")
else:
    st.info("No attendance recorded for today yet.")

st.header("Export date range to Excel")
col1, col2 = st.columns(2)
start = col1.date_input("Start date", datetime.now().date())
end = col2.date_input("End date", datetime.now().date())
if start > end:
    st.error("Start date must be <= end date")
else:
    out_path = f"attendance_export_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.xlsx"
    if st.button("Export to Excel"):
        ok = export_range_to_excel(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), out_path)
        if ok:
            st.success(f"Exported to {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download Excel", f, file_name=out_path)
        else:
            st.warning("No data found in the selected date range.")