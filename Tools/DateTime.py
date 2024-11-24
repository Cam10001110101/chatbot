from datetime import datetime
from langchain.tools import tool

@tool
def get_current_datetime():
    """Get the current date and time."""
    current = datetime.now()
    return {
        "date": current.strftime("%Y-%m-%d"),
        "time": current.strftime("%H:%M:%S"),
        "day_of_week": current.strftime("%A"),
        "timezone": datetime.now().astimezone().tzname(),
        "timestamp": current.timestamp(),
        "iso_format": current.isoformat()
    }

def get_datetime_tool():
    """Initialize and return the DateTime tool."""
    return get_current_datetime
