from typing import List
from .models import SensorData, Event

# Dummy in-memory database
sensor_data_db: List[SensorData] = []
event_db: List[Event] = []

# Functions to interact with the database

def add_sensor_data(data: SensorData):
    sensor_data_db.append(data)

def get_sensor_data() -> List[SensorData]:
    return sensor_data_db

def add_event(event: Event):
    event_db.append(event)

def get_events() -> List[Event]:
    return event_db
