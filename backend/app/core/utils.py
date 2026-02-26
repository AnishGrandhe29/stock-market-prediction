from datetime import date, timedelta


def get_next_trading_day(reference_date: date) -> date:
    """
    Returns the next trading day (skipping weekends).
    Note: Can be expanded to skip market holidays in production.
    """
    next_day = reference_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day
