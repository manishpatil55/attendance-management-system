# lbph/attendance_logger.py
import os
import csv
from datetime import datetime, timedelta
import pandas as pd
from lbph import config

LOG_DIR = config.ATTENDANCE_LOG_DIR

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def get_today_file():
    ensure_log_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"attendance_{today}.csv")

def mark_attendance(name, rollno, status="Present"):
    """
    Append a row to today's CSV if name not already marked today.
    """
    file = get_today_file()
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "RollNo", "Date", "Time", "Status"])
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # check duplicates
    with open(file, "r") as f:
        for line in f:
            if name in line and date_str in line:
                return False  # already marked

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, rollno, date_str, time_str, status])
    return True

def export_range_to_excel(start_date, end_date, out_path):
    """
    Export attendance CSV files in date range to a single excel workbook, with a sheet per date.
    start_date/end_date strings 'YYYY-MM-DD'
    """
    ensure_log_dir()
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    writer = pd.ExcelWriter(out_path, engine="openpyxl")
    cur = s
    any_written = False
    while cur <= e:
        fname = os.path.join(LOG_DIR, f"attendance_{cur.strftime('%Y-%m-%d')}.csv")
        sheetname = cur.strftime('%Y-%m-%d')
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df.to_excel(writer, sheet_name=sheetname[:31], index=False)
            any_written = True
        cur += timedelta(days=1)
    if any_written:
        writer.save()
        return True
    return False